# -*- coding: utf-8 -*-
"""
实验脚本：逐项验证“缩量 + 企稳 + 超跌”短线反转策略

输出（results/ 目录）
  e1_volume.csv / e1_volume.json      缩量定义比较
  e2_stable.csv                       企稳定义比较
  e3_oversold.csv                     超跌区间比较
  e4_buyrule.csv                      买入规则比较
  e5_stoploss.csv                     止损方式比较
  e6_takeprofit.csv                   止盈比较
  e7_exit.csv                         退出方式比较
  e8_distribution.csv                 收益分布（+6% 止盈 vs 让利润奔跑）
  e9_regime.csv                       市场分层
  e10_is_oos.csv                      样本内 / 样本外
  e0_baseline_null.csv                基准与消融（核心：是否有统计优势）
  summary.json                        汇总
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

BASE = E.BASE
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

IS_START, IS_END = "2016-04-01", "2020-12-31"
OOS_START, OOS_END = "2021-01-01", "2026-12-31"

BASE_PARAMS = dict(vol="A", stab="A", rng="D2-D6", buy_rule="A",
                   sl=0.03, sl_mode="fixed", tp=0.06,
                   exit_mode=E.EXIT_FIXED_TP, max_hold=20, slippage=0.001)

SUMMARY = {}
LOGS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGS.append(s)


def M(tr, label):
    m = E.metrics(tr, label)
    return m


def row(m, **extra):
    d = dict(m)
    d.pop("label", None)
    d.update(extra)
    return d


def save(df, name, label=None):
    p = os.path.join(RES, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  → {name} ({len(df)} 行)")
    return df


def main():
    t_start = time.time()
    log("加载数据面板 ...")
    P = E.Panel()
    log(f"  dates={len(P.dates)} ({P.dates[0]}~{P.dates[-1]})  codes={len(P.codes)}")
    log(f"  指数分层可用: {'hs300' in P.idx_close}  状态分布: "
        f"{pd.Series(P.market_on['state']).value_counts().to_dict()}")

    B = BASE_PARAMS

    # =============================================================== E0 基准与消融
    log("\n[E0] 基准与消融（判断是否存在统计优势）")
    rows = []

    # 基准1：全市场可交易样本（等权随机抽样），同退出规则
    elig = P.WARM & P.ALIVE
    sig_rand = E.sample_signal(P, elig, 120000, seed=7)
    tr_rand = E.simulate(P, sig_rand, sl=B["sl"], sl_mode="fixed", tp=B["tp"],
                         exit_mode=B["exit_mode"], max_hold=B["max_hold"],
                         slippage=B["slippage"])
    rows.append(row(M(tr_rand, ""), exp="随机全市场(基准)", variant="random_all"))

    # 基准2：仅超跌 D2-D6（无缩量、无企稳）
    sig_o = E.build_signal(P, vol="NONE", stab="NONE", rng="D2-D6", buy_rule="A")
    tr_o = E.simulate(P, sig_o, sl=B["sl"], tp=B["tp"], exit_mode=B["exit_mode"],
                      max_hold=B["max_hold"], slippage=B["slippage"])
    rows.append(row(M(tr_o, ""), exp="基准对照", variant="仅超跌D2-D6"))

    # 基准3：超跌 + 缩量（无企稳）
    sig_ov = E.build_signal(P, vol="A", stab="NONE", rng="D2-D6", buy_rule="A")
    tr_ov = E.simulate(P, sig_ov, sl=B["sl"], tp=B["tp"], exit_mode=B["exit_mode"],
                       max_hold=B["max_hold"], slippage=B["slippage"])
    rows.append(row(M(tr_ov, ""), exp="基准对照", variant="超跌D2-D6+缩量A"))

    # 基准4：超跌 + 企稳（无缩量）
    sig_os = E.build_signal(P, vol="NONE", stab="A", rng="D2-D6", buy_rule="A")
    tr_os = E.simulate(P, sig_os, sl=B["sl"], tp=B["tp"], exit_mode=B["exit_mode"],
                       max_hold=B["max_hold"], slippage=B["slippage"])
    rows.append(row(M(tr_os, ""), exp="基准对照", variant="超跌D2-D6+企稳A"))

    # 基准5：缩量 + 企稳（无超跌）
    sig_vs = E.build_signal(P, vol="A", stab="A", rng="ALL", buy_rule="A")
    tr_vs = E.simulate(P, sig_vs, sl=B["sl"], tp=B["tp"], exit_mode=B["exit_mode"],
                       max_hold=B["max_hold"], slippage=B["slippage"])
    rows.append(row(M(tr_vs, ""), exp="基准对照", variant="缩量A+企稳A(不限超跌)"))

    # 完整策略
    log("  运行完整策略 ...")
    tr_base = E.run_backtest(P, **B)
    rows.append(row(M(tr_base, ""), exp="完整策略", variant="超跌+缩量A+企稳A"))

    df0 = pd.DataFrame(rows)
    cols = ["variant", "exp", "n", "win_rate", "avg_ret", "avg_gross", "median_ret",
            "avg_win", "avg_loss", "profit_factor", "t_stat", "std", "avg_hold",
            "best", "worst"]
    save(df0[cols], "e0_baseline_null.csv")

    nav = E.equity_curve(tr_base, P)
    cs = E.curve_stats(nav)
    SUMMARY["baseline_curve"] = cs
    log(f"  完整策略净值: 总收益 {cs.get('total_ret', 0)*100:.1f}%  "
        f"CAGR {cs.get('cagr', 0)*100:.1f}%  最大回撤 {cs.get('max_dd', 0)*100:.1f}%  "
        f"Sharpe {cs.get('sharpe', 0):.2f}")

    # =============================================================== E1 缩量定义
    log("\n[E1] 缩量定义比较")
    rows = []
    vol_defs = {
        "A": "VOL_t<VOL_t-1<VOL_t-2",
        "B": "近5日量连续下降",
        "C": "VOL<VOL20*0.5(地量)",
        "D": "VOL<VOL20*0.7",
        "E": "近3日均量<VOL20*0.6",
        "NONE": "不设缩量条件",
    }
    for k, desc in vol_defs.items():
        p = dict(B); p["vol"] = k
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, vol=k, desc=desc))
        log(f"  {k:<5} {desc:<22} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e1_volume.csv")

    # =============================================================== E2 企稳定义
    log("\n[E2] 企稳定义比较")
    rows = []
    stab_defs = {
        "A": "Close>=前收",
        "B": "连续2日不创新低",
        "C": "5日低点后2日未破",
        "D": "Low>=前低 且 收盘>开盘",
        "E": "收盘站上MA5",
        "NONE": "不设企稳条件",
    }
    for k, desc in stab_defs.items():
        p = dict(B); p["stab"] = k
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, stab=k, desc=desc))
        log(f"  {k:<5} {desc:<20} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e2_stable.csv")

    # =============================================================== E3 超跌区间
    log("\n[E3] 前置超跌（横截面 ret20 分位）区间比较")
    rows = []
    for rng in ["D1-D3", "D2-D6", "D3-D7", "D4-D8", "D1-D10", "D5-D8"]:
        p = dict(B); p["rng"] = rng
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, rng=rng))
        log(f"  {rng:<7} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e3_oversold.csv")

    # =============================================================== E4 买入规则
    log("\n[E4] 买入规则比较")
    rows = []
    buy_defs = {
        "A": "超跌+缩量+企稳",
        "B": "超跌+缩量+不创新低+收盘上涨",
        "C": "超跌+缩量+不创新低+站上MA5",
    }
    for k, desc in buy_defs.items():
        p = dict(B); p["buy_rule"] = k
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, rule=k, desc=desc))
        log(f"  {k} {desc:<28} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e4_buyrule.csv")

    # =============================================================== E5 止损
    log("\n[E5] 止损方式比较")
    rows = []
    cfgs = [("fixed", 0.02, None, "-2%"), ("fixed", 0.03, None, "-3%"),
            ("fixed", 0.04, None, "-4%"), ("fixed", 0.05, None, "-5%"),
            ("atr", None, 1.0, "ATR×1"), ("atr", None, 1.5, "ATR×1.5"),
            ("atr", None, 2.0, "ATR×2")]
    for mode, sl, k, desc in cfgs:
        p = dict(B); p.update(sl_mode=mode, sl=sl or 0.03, atr_k=k)
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, sl_mode=mode, sl=sl, atr_k=k, desc=desc))
        log(f"  {desc:<8} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  盈亏比{m.get('profit_factor',0):5.2f}  "
            f"t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e5_stoploss.csv")

    # =============================================================== E6 止盈
    log("\n[E6] 止盈比较")
    rows = []
    for tp, desc in [(0.03, "+3%"), (0.04, "+4%"), (0.05, "+5%"), (0.06, "+6%"),
                     (0.08, "+8%"), (0.10, "+10%"), (0.15, "+15%"), (None, "不设止盈")]:
        p = dict(B); p["tp"] = tp
        p["exit_mode"] = E.EXIT_FIXED_TP if tp else E.EXIT_HOLD
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, tp=tp, desc=desc))
        log(f"  {desc:<8} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  平均持有{m.get('avg_hold',0):4.1f}日  "
            f"t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e6_takeprofit.csv")

    # =============================================================== E7 退出方式
    log("\n[E7] 重点退出方式比较")
    rows = []
    exits = [
        ("策略1 -3%/+6%", dict(sl=0.03, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("策略2 -3%/+6%卖100%", dict(sl=0.03, tp=0.06, exit_mode=E.EXIT_FIXED_TP)),
        ("策略3 -3%/+6%卖50%+MA5", dict(sl=0.03, tp=None, exit_mode=E.EXIT_TP_HALF_MA5, tp_half=0.06)),
        ("策略4 -3%/MA10", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MA10)),
        ("策略5 -3%/MA10", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MA10)),
        ("策略6 -3%/MACD死叉", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MACD)),
        ("对照 -3%/MA5", dict(sl=0.03, tp=None, exit_mode=E.EXIT_MA5)),
        ("对照 -3%/仅最长持有", dict(sl=0.03, tp=None, exit_mode=E.EXIT_HOLD)),
    ]
    tr_store = {}
    for desc, kw in exits:
        p = dict(B); p.update(kw)
        tr = E.run_backtest(P, **p)
        tr_store[desc] = tr
        m = M(tr, "")
        rows.append(row(m, desc=desc))
        log(f"  {desc:<24} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  持有{m.get('avg_hold',0):4.1f}日  "
            f"t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e7_exit.csv")

    # =============================================================== E8 收益分布
    log("\n[E8] 收益分布：固定+6%止盈 vs 让利润奔跑")
    tr_fix = tr_store["策略1 -3%/+6%"]
    tr_run = tr_store["策略4 -3%/MA10"]
    b_fix = E.bucket_report(tr_fix)
    b_run = E.bucket_report(tr_run)
    b_fix["group"] = "固定+6%止盈"
    b_run["group"] = "不设止盈(MA10)"
    save(pd.concat([b_fix, b_run], ignore_index=True), "e8_distribution.csv")

    for nm, tr in [("固定+6%止盈", tr_fix), ("不设止盈(MA10)", tr_run)]:
        r = tr.net_ret.to_numpy()
        big = r[r >= 0.10]
        huge = r[r >= 0.20]
        log(f"  {nm}: n={len(r)}  >=10%单数 {len(big)}({len(big)/len(r)*100:.2f}%) "
            f"合计贡献 {big.sum()/r.sum()*100:.1f}% |  >=20%单数 {len(huge)} "
            f"贡献 {huge.sum()/r.sum()*100:.1f}%")
        SUMMARY[f"dist_{nm}"] = {
            "n": int(len(r)),
            "cnt_ge10": int(len(big)), "sum_share_ge10": float(big.sum() / r.sum()),
            "cnt_ge20": int(len(huge)), "sum_share_ge20": float(huge.sum() / r.sum()),
            "cnt_ge30": int((r >= 0.30).sum()),
            "sum_share_ge30": float(r[r >= 0.30].sum() / r.sum()),
            "total_sum": float(r.sum()), "max": float(r.max()),
        }
    # 若不设止盈，超过6%的部分被保留了多少
    big_fix = (tr_run.net_ret >= 0.10).sum()
    log(f"  不设止盈时 >=10% 的交易数 {big_fix}，而固定6%止盈下为 "
        f"{(tr_fix.net_ret >= 0.10).sum()}")

    # =============================================================== E9 市场分层
    log("\n[E9] 市场环境分层")
    rows = []
    for desc, kw in [("全部", {}),
                     ("指数>MA60", dict(regime_filter="above_ma60")),
                     ("指数<MA60", dict(regime_filter="below_ma60")),
                     ("牛市", dict(regime_filter="牛市")),
                     ("熊市", dict(regime_filter="熊市")),
                     ("震荡市", dict(regime_filter="震荡"))]:
        p = dict(B); p.update(kw)
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, desc=desc))
        log(f"  {desc:<10} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e9_regime.csv")
    # 按交易日归属统计（同一批交易，按信号日市场状态分组，样本更完整）
    grp = tr_base.groupby("mkt_state").net_ret.agg(["count", "mean", "median",
                                                    lambda x: (x > 0).mean()])
    grp.columns = ["n", "avg_ret", "median", "win_rate"]
    grp = grp.reset_index().rename(columns={"mkt_state": "desc"})
    save(grp, "e9_regime_by_signal_day.csv")
    log("  按信号日市场状态分组：\n" + grp.to_string(index=False))

    # =============================================================== E10 样本内/外
    log("\n[E10] 样本内 / 样本外")
    rows = []
    for nm, s, e in [("样本内 2016-2020", IS_START, IS_END),
                     ("样本外 2021-2026", OOS_START, OOS_END),
                     ("全样本 2016-2026", None, None)]:
        tr = E.run_backtest(P, start=s, end=e, **B)
        m = M(tr, "")
        rows.append(row(m, period=nm))
        nav = E.equity_curve(tr, P, s, e)
        cs = E.curve_stats(nav)
        SUMMARY[f"curve_{nm}"] = cs
        log(f"  {nm:<18} n={m.get('n',0):>7}  胜率{m.get('win_rate',0)*100:5.1f}%  "
            f"均值{m.get('avg_ret',0)*100:6.3f}%  t={m.get('t_stat',float('nan')):6.2f}  "
            f"| 净值 {cs.get('total_ret',0)*100:7.1f}%  回撤 {cs.get('max_dd',0)*100:6.1f}%  "
            f"Sharpe {cs.get('sharpe',float('nan')):5.2f}")
    save(pd.DataFrame(rows), "e10_is_oos.csv")

    # 参数稳健性：在固定随机子样本（1200 只）上扫参数网格，样本内/外分别看
    log("\n[E10b] 样本内外参数稳健性（1200 只随机子样本 · 网格扫描）")
    rng_rng = np.random.default_rng(20260923)
    sub_idx = rng_rng.choice(len(P.codes), size=min(1200, len(P.codes)),
                             replace=False)
    sub_mask = np.zeros(len(P.codes), dtype=bool)
    sub_mask[sub_idx] = True

    def run_sub(**kw):
        s = E.build_signal(P, vol=kw["vol"], stab=kw["stab"], rng=kw["rng"],
                           buy_rule="A", start=kw.get("start"), end=kw.get("end"))
        s = s & sub_mask[None, :]
        return E.simulate(P, s, sl=B["sl"], tp=B["tp"],
                          exit_mode=B["exit_mode"], max_hold=B["max_hold"],
                          slippage=B["slippage"], start=kw.get("start"),
                          end=kw.get("end"))

    rows = []
    for vol in ["A", "B", "C", "D", "E"]:
        for stab in ["A", "B", "C", "D", "E"]:
            tri = run_sub(vol=vol, stab=stab, rng="D2-D6", start=IS_START, end=IS_END)
            tro = run_sub(vol=vol, stab=stab, rng="D2-D6", start=OOS_START, end=OOS_END)
            mi, mo = M(tri, ""), M(tro, "")
            rows.append({
                "vol": vol, "stab": stab, "rng": "D2-D6",
                "n_is": mi.get("n", 0), "avg_is": mi.get("avg_ret", np.nan),
                "t_is": mi.get("t_stat", np.nan),
                "n_oos": mo.get("n", 0), "avg_oos": mo.get("avg_ret", np.nan),
                "t_oos": mo.get("t_stat", np.nan),
                "win_oos": mo.get("win_rate", np.nan),
            })
            log(f"  vol={vol} stab={stab} | IS n={mi.get('n',0):>5} "
                f"avg={mi.get('avg_ret',0)*100:6.3f}% t={mi.get('t_stat',0):5.2f} | "
                f"OOS n={mo.get('n',0):>5} avg={mo.get('avg_ret',0)*100:6.3f}% "
                f"t={mo.get('t_stat',0):5.2f}")
    grid = pd.DataFrame(rows)
    save(grid, "e10b_grid.csv")
    n_sig = int(((grid.t_oos > 2)).sum())
    SUMMARY["grid_oos_sig"] = {"combos": len(grid), "t_oos_gt2": n_sig,
                               "best_avg_oos": float(grid.avg_oos.max()),
                               "median_avg_oos": float(grid.avg_oos.median())}
    log(f"  → 网格 {len(grid)} 组合中，样本外 t>2 的有 {n_sig} 个；"
        f"样本外平均收益中位数 {grid.avg_oos.median()*100:.3f}%")

    # =============================================================== 稳健性：成本与滑点
    log("\n[E11] 成本敏感性")
    rows = []
    for slip, nm in [(0.0, "零滑点"), (0.0005, "5bp滑点"), (0.001, "10bp滑点"),
                     (0.002, "20bp滑点"), (0.003, "30bp滑点")]:
        p = dict(B); p["slippage"] = slip
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, desc=nm, slippage=slip))
        log(f"  {nm:<9} n={m.get('n',0):>7}  均值{m.get('avg_ret',0)*100:6.3f}%  "
            f"t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e11_cost.csv")

    # =============================================================== 最长持有敏感性
    log("\n[E12] 最长持有期敏感性")
    rows = []
    for mh in [5, 10, 15, 20, 30, 40, 60]:
        p = dict(B); p["max_hold"] = mh
        tr = E.run_backtest(P, **p)
        m = M(tr, "")
        rows.append(row(m, max_hold=mh))
        log(f"  最长持有{mh:>3}日  n={m.get('n',0):>7}  均值{m.get('avg_ret',0)*100:6.3f}%  "
            f"t={m.get('t_stat',float('nan')):6.2f}")
    save(pd.DataFrame(rows), "e12_maxhold.csv")

    # ===============================================================
    SUMMARY["base_params"] = BASE_PARAMS
    SUMMARY["elapsed"] = time.time() - t_start
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(RES, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOGS))
    log(f"\n全部完成，用时 {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
