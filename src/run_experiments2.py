# -*- coding: utf-8 -*-
"""
第二轮实验：
 A. 逐条件消融（在完全相同的退出规则与成本下，比较每个条件的边际贡献）
 B. 候选策略的样本内外验证 + 成本敏感性 + 市场分层 + 净值曲线
 C. 随机基准的“前瞻收益分布”（用于判断是否有超越市场漂移的 alpha）
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

RES = os.path.join(E.BASE, "results")
os.makedirs(RES, exist_ok=True)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def C_(P, name):
    """信号组件（全部只用 T 日及之前数据）。"""
    C, L, O, V = P.C, P.L, P.O, P.V
    if name == "true":
        return np.ones_like(C, dtype=bool)
    if name == "up":                     # 收盘 ≥ 前收
        return C >= np.roll(C, 1, axis=0)
    if name == "nolow2":                 # 连续 2 日不创新低
        return (L >= np.roll(L, 1, axis=0)) & (np.roll(L, 1, axis=0) >= np.roll(L, 2, axis=0))
    if name == "nolow1":                 # 今日不创新低
        return L >= np.roll(L, 1, axis=0)
    if name == "ma5above":               # 站上 MA5
        return C > P.MA5
    if name == "green":                  # 收阳
        return C > O
    if name == "volA":
        return E.vol_cond(P, "A")
    if name == "volB":
        return E.vol_cond(P, "B")
    if name == "volC":
        return E.vol_cond(P, "C")
    if name == "volD":
        return E.vol_cond(P, "D")
    if name == "volE":
        return E.vol_cond(P, "E")
    raise KeyError(name)


def build(P, comps, rng="D2-D6", start=None, end=None, regime=None):
    dates = P.dates
    s = np.ones_like(P.C, dtype=bool)
    for c in comps:
        s &= C_(P, c)
    if rng != "ALL":
        s &= E.oversold_cond(P, rng)
    s &= P.WARM
    if start:
        s[:np.searchsorted(dates, start)] = False
    if end:
        s[np.searchsorted(dates, end) + 1:] = False
    s[:70] = False
    if regime == "above_ma60":
        s &= P.market_on["above_ma60"][:, None]
    elif regime == "below_ma60":
        s &= ~P.market_on["above_ma60"][:, None]
    elif regime in ("牛市", "熊市", "震荡"):
        s &= (P.market_on["state"] == regime)[:, None]
    return s


def run(P, comps, **kw):
    st = kw.pop("start", None)
    en = kw.pop("end", None)
    rg = kw.pop("regime", None)
    rng = kw.pop("rng", "D2-D6")
    sig = build(P, comps, rng=rng, start=st, end=en, regime=rg)
    return E.simulate(P, sig, start=st, end=en, **kw)


def M(tr):
    return E.metrics(tr)


def row(m, **ex):
    d = {k: v for k, v in m.items() if k != "label"}
    d.update(ex)
    return d


EXIT = dict(sl=0.03, sl_mode="fixed", tp=0.06, exit_mode=E.EXIT_FIXED_TP,
            max_hold=20, slippage=0.001)


def main():
    t0 = time.time()
    P = E.Panel()
    log(f"面板 {len(P.dates)} 交易日 × {len(P.codes)} 只  "
        f"({P.dates[0]} ~ {P.dates[-1]})")

    # ================================================= A. 逐条件消融
    log("\n[A] 逐条件消融（统一退出：止损-3% / 止盈+6% / 最长20日 / 滑点10bp）")
    variants = [
        ("随机全市场（基准）", ["true"], "ALL", "random"),
        ("超跌D2-D6（基准）", ["true"], "D2-D6", "oversold_base"),
        ("超跌+缩量A", ["volA"], "D2-D6", "+volA"),
        ("超跌+企稳A(收≥前收)", ["up"], "D2-D6", "+up"),
        ("超跌+不创新低(2日)", ["nolow2"], "D2-D6", "+nolow2"),
        ("超跌+站上MA5", ["ma5above"], "D2-D6", "+ma5here"),
        ("超跌+不创新低+收阳", ["nolow2", "green"], "D2-D6", "+nolow2+green"),
        ("超跌+不创新低+收涨", ["nolow2", "up"], "D2-D6", "+nolow2+up"),
        ("超跌+不创新低+站上MA5", ["nolow2", "ma5above"], "D2-D6", "+nolow2+ma5"),
        ("超跌+缩量A+不创新低+收涨【规则B】", ["volA", "nolow2", "up"], "D2-D6", "ruleB"),
        ("超跌+缩量A+不创新低+站上MA5【规则C】", ["volA", "nolow2", "ma5above"], "D2-D6", "ruleC"),
        ("超跌+缩量C(地量)+不创新低+收涨", ["volC", "nolow2", "up"], "D2-D6", "ruleB_volC"),
        ("超跌+缩量D+不创新低+收涨", ["volD", "nolow2", "up"], "D2-D6", "ruleB_volD"),
        ("超跌+缩量B+不创新低+收涨", ["volB", "nolow2", "up"], "D2-D6", "ruleB_volB"),
        ("超跌+缩量E+不创新低+收涨", ["volE", "nolow2", "up"], "D2-D6", "ruleB_volE"),
        ("不加超跌：缩量A+不创新低+收涨", ["volA", "nolow2", "up"], "ALL", "noOversold"),
        ("全市场+不创新低+收涨", ["nolow2", "up"], "ALL", "noOversold_noVol"),
    ]
    rows = []
    trs = {}
    for lbl, comps, rng, key in variants:
        if "随机" in lbl:
            sig = E.sample_signal(P, P.WARM & P.ALIVE, 150000, seed=11)
            tr = E.simulate(P, sig, **EXIT)
        else:
            tr = run(P, comps, rng=rng, **EXIT)
        trs[key] = tr
        m = M(tr)
        rows.append(row(m, variant=lbl, key=key))
        log(f"  {lbl:<34} n={m.get('n',0):>9} 胜率{m.get('win_rate',0)*100:5.1f}% "
            f"均值{m.get('avg_ret',0)*100:7.3f}% t={m.get('t_stat',float('nan')):7.2f} "
            f"PF={m.get('profit_factor',0):5.2f}")
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(RES, "f1_ablation.csv"), index=False, encoding="utf-8-sig")

    # ================================================= B. 候选策略深度验证
    log("\n[B] 候选策略深度验证")
    cands = {
        "规则B(缩量A)": ["volA", "nolow2", "up"],
        "规则C(缩量A)": ["volA", "nolow2", "ma5above"],
        "规则B(缩量不走缩量条件)": ["nolow2", "up"],
    }
    rows = []
    for lbl, comps in cands.items():
        for pn, s, e in [("样本内16-20", "2016-04-01", "2020-12-31"),
                         ("样本外21-26", "2021-01-01", "2026-12-31"),
                         ("全样本", None, None)]:
            tr = run(P, comps, start=s, end=e, **EXIT)
            m = M(tr)
            rows.append(row(m, cand=lbl, period=pn))
            log(f"  {lbl:<22}{pn:<12} n={m.get('n',0):>8} 均值{m.get('avg_ret',0)*100:7.3f}% "
                f"t={m.get('t_stat',float('nan')):6.2f} 胜率{m.get('win_rate',0)*100:5.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f2_cand_is_oos.csv"),
                              index=False, encoding="utf-8-sig")

    # 成本敏感性（候选 vs 基准）
    log("\n[B2] 成本/滑点敏感性（零滑点 ~ 30bp）")
    rows = []
    for lbl, comps in [("随机全市场", None), ("超跌基准", ["true"]),
                       ("规则B(缩量A)", ["volA", "nolow2", "up"]),
                       ("规则C(缩量A)", ["volA", "nolow2", "ma5above"]),
                       ("规则B(无缩量)", ["nolow2", "up"])]:
        for slip in [0.0, 0.0005, 0.001, 0.002, 0.003]:
            ex = dict(EXIT); ex["slippage"] = slip
            if comps is None:
                sig = E.sample_signal(P, P.WARM & P.ALIVE, 150000, seed=11)
                tr = E.simulate(P, sig, **ex)
            else:
                rng = "D2-D6"
                tr = run(P, comps, rng=rng, **ex)
            m = M(tr)
            rows.append(row(m, cand=lbl, slippage=slip))
    cs = pd.DataFrame(rows)
    cs.to_csv(os.path.join(RES, "f3_cost.csv"), index=False, encoding="utf-8-sig")
    for lbl in cs.cand.unique():
        d = cs[cs.cand == lbl]
        log(f"  {lbl:<16} " + "  ".join(
            f"{r.slippage*1e4:.0f}bp:{r.avg_ret*100:+.3f}%" for r in d.itertuples()))

    # 市场分层
    log("\n[B3] 候选策略市场分层")
    rows = []
    for lbl, comps in [("规则B(缩量A)", ["volA", "nolow2", "up"]),
                       ("规则C(缩量A)", ["volA", "nolow2", "ma5above"])]:
        for rg, nm in [(None, "全部"), ("above_ma60", "指数>MA60"),
                       ("below_ma60", "指数<MA60"), ("牛市", "牛市"),
                       ("熊市", "熊市"), ("震荡", "震荡市")]:
            tr = run(P, comps, rng="D2-D6", regime=rg, **EXIT)
            m = M(tr)
            rows.append(row(m, cand=lbl, regime=nm))
            log(f"  {lbl:<16}{nm:<12} n={m.get('n',0):>8} "
                f"均值{m.get('avg_ret',0)*100:7.3f}% t={m.get('t_stat',float('nan')):6.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f4_regime.csv"),
                              index=False, encoding="utf-8-sig")

    # 退出方式（针对候选）
    log("\n[B4] 候选策略的退出方式比较")
    exits = [
        ("止损-3% + 止盈+6%", dict(sl=0.03, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损-5% + 止盈+6%", dict(sl=0.05, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损ATR×2 + 止盈+6%", dict(sl=0.03, sl_mode="atr", atr_k=2.0, tp=0.06,
                                     exit_mode=E.EXIT_FIXED_TP)),
        ("止损-3% + 止盈+10%", dict(sl=0.03, tp=0.10, exit_mode=E.EXIT_FIXED_TP)),
        ("止损-3% + 不设止盈(最长20日)", dict(sl=0.03, tp=None, exit_mode=E.EXIT_HOLD)),
        ("止损-3% + MACD死叉", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MACD)),
        ("止损-3% + 跌破MA10", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MA10)),
        ("止损-3% + +6%卖50%后跌破MA5", dict(sl=0.03, tp=None,
                                             exit_mode=E.EXIT_TP_HALF_MA5, tp_half=0.06)),
    ]
    rows = []
    best = {}
    for lbl, comps in [("规则B(缩量A)", ["volA", "nolow2", "up"]),
                       ("规则C(缩量A)", ["volA", "nolow2", "ma5above"])]:
        for en, kw in exits:
            tr = run(P, comps, rng="D2-D6", slippage=0.001, max_hold=20, **kw)
            m = M(tr)
            rows.append(row(m, cand=lbl, exit=en))
            log(f"  {lbl:<16}{en:<26} 均值{m.get('avg_ret',0)*100:7.3f}% "
                f"t={m.get('t_stat',float('nan')):6.2f} 持有{m.get('avg_hold',0):4.1f}日")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f5_exits.csv"),
                              index=False, encoding="utf-8-sig")

    # 基准的“前瞻收益”分布（判断 alpha vs 漂移）
    log("\n[B5] 市场漂移基准：随机买入并持有 N 日的平均收益")
    rows = []
    for h in [1, 3, 5, 10, 20]:
        sig = E.sample_signal(P, P.WARM & P.ALIVE, 120000, seed=5)
        tr = E.simulate(P, sig, sl=0.99, sl_mode="fixed", tp=None,
                        exit_mode=E.EXIT_HOLD, max_hold=h, slippage=0.0)
        m = M(tr)
        rows.append(row(m, hold=h))
        log(f"  随机买入持有{h:>3}日  均值{m.get('avg_ret',0)*100:7.3f}% "
            f"胜率{m.get('win_rate',0)*100:5.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f6_drift.csv"),
                              index=False, encoding="utf-8-sig")

    # 候选策略净值曲线（计入成本）
    log("\n[B6] 候选策略净值曲线（已计入手续费/印花税/滑点）")
    curves = {}
    for key, comps in [("ruleB", ["volA", "nolow2", "up"]),
                       ("ruleC", ["volA", "nolow2", "ma5above"]),
                       ("ruleB_noVol", ["nolow2", "up"])]:
        tr = run(P, comps, rng="D2-D6", **EXIT)
        nav = E.equity_curve(tr, P, slippage=0.001)
        st = E.curve_stats(nav)
        curves[key] = st
        log(f"  {key:<12} 总收益{st.get('total_ret',0)*100:8.1f}%  "
            f"年化{st.get('cagr',0)*100:6.2f}%  回撤{st.get('max_dd',0)*100:7.1f}%  "
            f"Sharpe{st.get('sharpe',0):5.2f}")
        nav.iloc[::5].to_csv(os.path.join(RES, f"f7_nav_{key}.csv"),
                             encoding="utf-8-sig")
    # 基准净值
    sig = E.sample_signal(P, P.WARM & P.ALIVE, 150000, seed=11)
    trr = E.simulate(P, sig, **EXIT)
    nav = E.equity_curve(trr, P, slippage=0.001)
    st = E.curve_stats(nav)
    curves["random"] = st
    log(f"  {'random':<12} 总收益{st.get('total_ret',0)*100:8.1f}%  "
        f"年化{st.get('cagr',0)*100:6.2f}%  回撤{st.get('max_dd',0)*100:7.1f}%  "
        f"Sharpe{st.get('sharpe',0):5.2f}")
    nav.iloc[::5].to_csv(os.path.join(RES, "f7_nav_random.csv"), encoding="utf-8-sig")
    json.dump(curves, open(os.path.join(RES, "f8_curves.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2,
              default=str)

    open(os.path.join(RES, "run2.log"), "w", encoding="utf-8").write("\n".join(LOG))
    log(f"\n完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
