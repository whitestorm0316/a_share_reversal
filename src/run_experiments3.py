# -*- coding: utf-8 -*-
"""第三轮：成本敏感性 / 市场分层 / 退出方式 / 市场漂移基准 / 净值曲线（精简版）"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402
from run_experiments2 import build, M, row, EXIT  # noqa: E402

RES = os.path.join(E.BASE, "results")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


CANDS = {
    "规则B(超跌+缩量A+不创新低+收涨)": ["volA", "nolow2", "up"],
    "规则C(超跌+缩量A+不创新低+站上MA5)": ["volA", "nolow2", "ma5above"],
    "精简版(超跌+不创新低+收涨，无缩量)": ["nolow2", "up"],
}


def main():
    t0 = time.time()
    P = E.Panel()
    log(f"面板 {len(P.dates)} × {len(P.codes)}")

    # ---------------- 成本敏感性
    log("\n[B2] 成本/滑点敏感性")
    rows = []
    for lbl, comps in CANDS.items():
        for slip in [0.0, 0.0005, 0.001, 0.002, 0.003]:
            ex = dict(EXIT); ex["slippage"] = slip
            tr = E.simulate(P, build(P, comps, rng="D2-D6"), **ex)
            m = M(tr)
            rows.append(row(m, cand=lbl, slippage=slip))
    for slip in [0.0, 0.0005, 0.001, 0.002, 0.003]:
        ex = dict(EXIT); ex["slippage"] = slip
        sig = E.sample_signal(P, P.WARM & P.ALIVE, 150000, seed=11)
        m = M(E.simulate(P, sig, **ex))
        rows.append(row(m, cand="随机全市场(基准)", slippage=slip))
    cs = pd.DataFrame(rows)
    cs.to_csv(os.path.join(RES, "f3_cost.csv"), index=False, encoding="utf-8-sig")
    for lbl in cs.cand.unique():
        d = cs[cs.cand == lbl].sort_values("slippage")
        log(f"  {lbl:<34} " + "  ".join(
            f"{r.slippage*1e4:>2.0f}bp:{r.avg_ret*100:+.3f}%" for r in d.itertuples()))

    # ---------------- 市场分层
    log("\n[B3] 市场分层")
    rows = []
    for lbl, comps in CANDS.items():
        for rg, nm in [(None, "全部"), ("above_ma60", "指数>MA60"),
                       ("below_ma60", "指数<MA60"), ("牛市", "牛市"),
                       ("熊市", "熊市"), ("震荡", "震荡市")]:
            tr = E.simulate(P, build(P, comps, rng="D2-D6", regime=rg), **EXIT)
            m = M(tr)
            rows.append(row(m, cand=lbl, regime=nm))
            log(f"  {lbl[:18]:<18}{nm:<11} n={m.get('n',0):>8} "
                f"均值{m.get('avg_ret',0)*100:7.3f}% t={m.get('t_stat',float('nan')):6.2f} "
                f"胜率{m.get('win_rate',0)*100:5.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f4_regime.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------------- 退出方式
    log("\n[B4] 退出方式（规则B）")
    exits = [
        ("止损-2% + 止盈+6%", dict(sl=0.02, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损-3% + 止盈+6%", dict(sl=0.03, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损-5% + 止盈+6%", dict(sl=0.05, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损ATR×1.5 + 止盈+6%", dict(sl=0.03, sl_mode="atr", atr_k=1.5,
                                       tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损ATR×2 + 止盈+6%", dict(sl=0.03, sl_mode="atr", atr_k=2.0,
                                     tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("止损-3% + 止盈+10%", dict(sl=0.03, tp=0.10, exit_mode=E.EXIT_FIXED_TP)),
        ("止损-3% + 止损/止盈(不设止盈,最长20日)",
         dict(sl=0.03, tp=None, exit_mode=E.EXIT_HOLD)),
        ("止损-3% + MACD死叉", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MACD)),
        ("止损-3% + 跌破MA10", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MA10)),
        ("止损-3% + 跌破MA5", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MA5)),
        ("止损-3% + +6%卖50%后MA5", dict(sl=0.03, tp=None,
                                         exit_mode=E.EXIT_TP_HALF_MA5, tp_half=0.06)),
    ]
    rows = []
    for lbl, comps in [("规则B(缩量A)", CANDS["规则B(超跌+缩量A+不创新低+收涨)"])]:
        for en, kw in exits:
            tr = E.simulate(P, build(P, comps, rng="D2-D6"),
                            slippage=0.001, max_hold=20, **kw)
            m = M(tr)
            rows.append(row(m, cand=lbl, exit=en))
            log(f"  {en:<34} 均值{m.get('avg_ret',0)*100:7.3f}% "
                f"t={m.get('t_stat',float('nan')):6.2f} 持有{m.get('avg_hold',0):5.1f}日 "
                f"胜率{m.get('win_rate',0)*100:5.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f5_exits.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------------- 市场漂移基准
    log("\n[B5] 市场漂移基准：随机买入并持有 N 个交易日（零成本）")
    rows = []
    for h in [1, 2, 3, 5, 10, 20]:
        sig = E.sample_signal(P, P.WARM & P.ALIVE, 120000, seed=5)
        tr = E.simulate(P, sig, sl=0.99, sl_mode="fixed", tp=None,
                        exit_mode=E.EXIT_HOLD, max_hold=h, slippage=0.0)
        m = M(tr)
        rows.append(row(m, hold=h))
        log(f"  随机持有{h:>3}日  均值{m.get('avg_ret',0)*100:7.3f}% "
            f"胜率{m.get('win_rate',0)*100:5.1f}% n={m.get('n',0)}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "f6_drift.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------------- 净值曲线
    log("\n[B6] 净值曲线（计入全部成本+滑点）")
    curves = {}
    items = list(CANDS.items()) + [("随机全市场(基准)", None)]
    for lbl, comps in items:
        if comps is None:
            sig = E.sample_signal(P, P.WARM & P.ALIVE, 150000, seed=11)
            tr = E.simulate(P, sig, **EXIT)
            key = "random"
        else:
            tr = E.simulate(P, build(P, comps, rng="D2-D6"), **EXIT)
            key = {"规则B": "ruleB", "规则C": "ruleC", "精简版": "simple"}[lbl[:3]]
        nav = E.equity_curve(tr, P, slippage=0.001)
        st = E.curve_stats(nav)
        curves[key] = st
        log(f"  {lbl:<34} 总收益{st.get('total_ret',0)*100:8.1f}% "
            f"年化{st.get('cagr',0)*100:6.2f}% 回撤{st.get('max_dd',0)*100:7.1f}% "
            f"Sharpe{st.get('sharpe',0):5.2f}")
        nav.iloc[::5].to_csv(os.path.join(RES, f"f7_nav_{key}.csv"),
                             encoding="utf-8-sig")
        # 分年净值
        yr = pd.Series(nav.values, index=pd.to_datetime(nav.index))
        curves[key + "_yearly"] = {
            str(y): float(g.iloc[-1] / g.iloc[0] - 1)
            for y, g in yr.groupby(yr.index.year)}
    json.dump(curves, open(os.path.join(RES, "f8_curves.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2,
              default=str)

    # ---------------- 候选策略 收益分布
    log("\n[B7] 收益分布（候选 vs 随机）")
    out = []
    for lbl, comps in CANDS.items():
        tr = E.simulate(P, build(P, comps, rng="D2-D6"), **EXIT)
        b = E.bucket_report(tr)
        b["group"] = lbl
        out.append(b)
        r = tr.net_ret.to_numpy()
        log(f"  {lbl}: n={len(r)} ≥10%:{int((r>=0.1).sum())} "
            f"≥20%:{int((r>=0.2).sum())} ≥30%:{int((r>=0.3).sum())} "
            f"最大{r.max()*100:.1f}%")
    pd.concat(out).to_csv(os.path.join(RES, "f9_dist.csv"), index=False,
                          encoding="utf-8-sig")

    open(os.path.join(RES, "run3.log"), "w", encoding="utf-8").write("\n".join(LOG))
    log(f"\n完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
