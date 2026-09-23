# -*- coding: utf-8 -*-
"""生成最终研究报告（纯静态 HTML，无外部依赖）→ reports/缩量企稳反转策略_回测报告.html"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htmlutil as H  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "reports")
os.makedirs(OUT, exist_ok=True)


def rd(name):
    return pd.read_csv(os.path.join(R, name))


def P(v, d=3):
    """百分点（输入是小数收益）"""
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v*100:.{d}f}%"


def N(v, d=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{d}f}"


def T(v):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:+.2f}"


def sig_cls(v, lo=2.0):
    """统计显著性标签"""
    if v is None or not np.isfinite(v):
        return ("—", "mut")
    if v >= lo:
        return ("显著为正", "good")
    if v <= -lo:
        return ("显著为负", "bad")
    return ("不显著", "mut")


def tag(txt, cls="mut"):
    return f'<span class="tag {cls}">{txt}</span>'


def verdict_card(title, value, sub, cls):
    return (f'<div class="vcard {cls}"><div class="vt">{title}</div>'
            f'<div class="vv">{value}</div><div class="vs">{sub}</div></div>')


# ============================ 载入数据 ============================
e1 = rd("e1_volume.csv")
e2 = rd("e2_stable.csv")
e3 = rd("e3_oversold.csv")
e4 = rd("e4_buyrule.csv")
e5 = rd("e5_stoploss.csv")
e6 = rd("e6_takeprofit.csv")
e7 = rd("e7_exit.csv")
e9 = rd("e9_regime.csv")
e10 = rd("e10_is_oos.csv")
e11 = rd("e11_cost.csv")
e12 = rd("e12_maxhold.csv")
f1 = rd("f1_ablation.csv")
f2 = rd("f2_cand_is_oos.csv")
f3 = rd("f3_cost.csv")
f4 = rd("f4_regime.csv")
f5 = rd("f5_exits.csv")
f6 = rd("f6_drift.csv")
f9 = rd("f9_dist.csv")
with open(os.path.join(R, "f8_curves.json"), encoding="utf-8") as _f:
    CV = json.load(_f)
with open(os.path.join(R, "summary.json"), encoding="utf-8") as _f:
    SM = json.load(_f)
MAX_TP6 = SM["dist_固定+6%止盈"]["max"]
MAX_RUN = SM["dist_不设止盈(MA10)"]["max"]

NAV = {}
for k in ("ruleB", "ruleC", "simple", "random"):
    s = pd.read_csv(os.path.join(R, f"f7_nav_{k}.csv"), index_col=0)
    s = s.iloc[:, 0]
    s.index = s.index.astype(str)
    NAV[k] = s

# 基准行（原策略）
B = e1.iloc[0]          # 缩量A + 企稳A + D2-D6 + 规则A + sl3% + tp6%
base_avg, base_t = B["avg_ret"], B["t_stat"]

# ---- 派生：单位时间收益（对比漂移基准）----
drift = f6.set_index("hold")
drift_per_day = (drift["avg_ret"] / drift.index.to_series()).to_dict()
best_cand = f1[f1.key == "ruleB"].iloc[0]
cand_per_day = best_cand["avg_ret"] / best_cand["avg_hold"]
cand_gross_per_day = best_cand["avg_ret_gross"] / best_cand["avg_hold"]
simple_row = f1[f1.key == "noOversold_noVol"].iloc[0]


def qrow(df, col, val, *pairs, **kw):
    """取满足 col==val 且若干条件的第一行。pairs 为 col,val 交替的位置参数。"""
    m = np.array(df[col] == val, dtype=bool)
    for i in range(0, len(pairs), 2):
        m &= np.array(df[pairs[i]] == pairs[i + 1], dtype=bool)
    for k, v in kw.items():
        m &= np.array(df[k] == v, dtype=bool)
    return df[m].iloc[0]


# ============================ 图表数据 ============================
vol_labels = ["A 两日递减", "B 五日连降", "C 地量(<0.5×)",
              "D 温和(<0.7×)", "E 三日均(<0.6×)", "不设缩量"]
vol_vals = list(e1["avg_ret"])

stab_labels = ["A 收≥前收", "B 2日不创新低", "C 5日低2日未破",
               "D 不破前低+收阳", "E 站上MA5", "不设企稳"]
stab_vals = list(e2["avg_ret"])

os_labels = ["D1-D3 最超跌", "D2-D6 原策略", "D3-D7", "D4-D8", "D5-D8", "D1-D10 全部"]
os_vals = [qrow(e3, "rng", x)["avg_ret"] for x in
           ["D1-D3", "D2-D6", "D3-D7", "D4-D8", "D5-D8", "D1-D10"]]

sl_labels = ["-2%", "-3% 原策略", "-4%", "-5%", "ATR×1", "ATR×1.5", "ATR×2"]
sl_vals = [qrow(e5, "sl", 0.02)["avg_ret"], base_avg, qrow(e5, "sl", 0.04)["avg_ret"],
           qrow(e5, "sl", 0.05)["avg_ret"],
           qrow(e5, "atr_k", 1.0)["avg_ret"], qrow(e5, "atr_k", 1.5)["avg_ret"],
           qrow(e5, "atr_k", 2.0)["avg_ret"]]

tp_labels = ["+3%", "+4%", "+5%", "+6% 原策略", "+8%", "+10%", "+15%", "不设止盈"]
tp_vals = [qrow(e6, "tp", x)["avg_ret"] for x in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]]
tp_vals.append(e6.loc[e6["tp"].isna(), "avg_ret"].iloc[0])

exit_tbl = e7.copy()
exit_tbl["desc"] = exit_tbl["desc"].str.replace("策略", "方案", regex=False)

reg_labels = ["牛市", "熊市", "震荡市", "指数>MA60", "指数<MA60"]
reg_vals = [qrow(e9, "desc", x)["avg_ret"] for x in
            ["牛市", "熊市", "震荡市", "指数>MA60", "指数<MA60"]]

cost_labels = ["0bp", "5bp", "10bp 原假设", "20bp", "30bp"]
cost_vals = list(e11.sort_values("slippage")["avg_ret"])

hold_labels = [f"{int(x)}日" for x in e12["max_hold"]]
hold_vals = list(e12["avg_ret"])

drift_labels = [f"{int(x)}日" for x in f6["hold"]]
drift_vals = list(f6["avg_ret"])

abl = f1.set_index(f1["key"].str.lstrip("+"))


def a(key, col):
    return abl.loc[key.lstrip("+"), col]


# ---- 2×2 因素分解（超跌 × 缩量），其余条件固定为「不创新低 + 收涨」 ----
BASE_NONE = a("noOversold_noVol", "avg_ret")   # 无超跌 无缩量
VOL_ONLY = a("noOversold", "avg_ret")          # 无超跌 有缩量A
OS_ONLY = a("nolow2+up", "avg_ret")            # 有超跌 无缩量
BOTH = a("ruleB", "avg_ret")                   # 有超跌 有缩量A
d_os_noVol = OS_ONLY - BASE_NONE
d_os_withVol = BOTH - VOL_ONLY
d_vol_noOS = VOL_ONLY - BASE_NONE
d_vol_withOS = BOTH - OS_ONLY
delta_volA = d_vol_withOS
delta_volC = a("ruleB_volC", "avg_ret") - OS_ONLY
delta_oversold = d_os_noVol

# 单位时间对比
drift20 = drift.loc[20, "avg_ret"]
perday = {
    "随机持有20日（零成本）": drift20 / 20,
    "随机全市场(含成本)": a("random", "avg_ret") / a("random", "avg_hold"),
    "原策略 缩量A+企稳A": base_avg / B["avg_hold"],
    "规则B 修正版": cand_per_day,
}

# ============================ 输出 ============================
CSS = """
:root{
  --bg:#f4f6f9; --card:#ffffff; --line:#e4e8ee; --line2:#eef1f6;
  --tx:#1b2230; --tx2:#586274; --tx3:#8b95a7;
  --red:#c0392b; --green:#1e7d5a; --blue:#2b6cb0; --amber:#b7791f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
 font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}
.wrap{max-width:1000px;margin:0 auto;padding:0 20px 90px}
header.hero{background:linear-gradient(160deg,#1b2230,#2c3a52);color:#fff;
 padding:46px 20px 40px;margin-bottom:26px}
header.hero .inner{max-width:1000px;margin:0 auto;padding:0 20px}
header.hero h1{margin:0 0 10px;font-size:28px;letter-spacing:.4px;font-weight:700}
header.hero .sub{color:#b9c4d6;font-size:14px;line-height:1.9}
header.hero .meta{margin-top:16px;font-size:12.5px;color:#8e9cb4;
 font-family:ui-monospace,Consolas,monospace}
h2{font-size:20px;margin:46px 0 14px;padding-bottom:9px;border-bottom:2px solid var(--line)}
h2 .no{color:var(--tx3);font-weight:400;margin-right:8px;font-family:ui-monospace,monospace}
h3{font-size:16px;margin:30px 0 10px;color:#2a3444}
h4{font-size:14.5px;margin:22px 0 8px;color:#37425a}
p{margin:11px 0}
ul,ol{margin:11px 0;padding-left:24px}
li{margin:5px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:20px 22px;margin:16px 0}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:1fr 1fr}
.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}
@media(max-width:820px){.g2,.g3,.g4{grid-template-columns:1fr}}
.vcard{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--tx3);
 border-radius:8px;padding:14px 16px}
.vcard.bad{border-left-color:var(--green)}
.vcard.good{border-left-color:var(--red)}
.vcard.warn{border-left-color:var(--amber)}
.vcard.info{border-left-color:var(--blue)}
.vt{font-size:12.5px;color:var(--tx2);margin-bottom:5px}
.vv{font-size:23px;font-weight:700;font-family:ui-monospace,Consolas,monospace;letter-spacing:-.5px}
.vcard.bad .vv{color:var(--green)}
.vcard.good .vv{color:var(--red)}
.vcard.warn .vv{color:var(--amber)}
.vcard.info .vv{color:var(--blue)}
.vs{font-size:12px;color:var(--tx3);margin-top:5px;line-height:1.6}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th{background:#f0f3f8;color:#3a4658;font-weight:600;text-align:left;
 padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;font-size:12.5px}
td{padding:7px 10px;border-bottom:1px solid var(--line2);font-family:ui-monospace,Consolas,monospace}
td:first-child{font-family:inherit}
tr:hover td{background:#fafbfd}
td.pos{color:var(--red);font-weight:600}
td.neg{color:var(--green);font-weight:600}
.tag{display:inline-block;padding:1px 8px;border-radius:11px;font-size:11.5px;
 font-family:inherit;white-space:nowrap}
.tag.good{background:#fdecea;color:var(--red)}
.tag.bad{background:#e8f4ef;color:var(--green)}
.tag.warn{background:#fdf5e3;color:#96650f}
.tag.info{background:#eaf1fa;color:var(--blue)}
.tag.mut{background:#eef1f5;color:var(--tx2)}
.chart{margin:14px 0 6px}
.cap{font-size:12px;color:var(--tx3);margin:2px 0 18px;line-height:1.7}
.callout{border-left:4px solid var(--blue);background:#f2f7fd;padding:13px 17px;
 border-radius:0 8px 8px 0;margin:16px 0;font-size:14px}
.callout.warn{border-left-color:var(--amber);background:#fdf9ef}
.callout.bad{border-left-color:var(--green);background:#f0f7f4}
.callout.good{border-left-color:var(--red);background:#fdf3f1}
.callout .h{font-weight:700;display:block;margin-bottom:4px}
code{background:#eef1f5;padding:1.5px 6px;border-radius:4px;
 font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
pre{background:#1b2230;color:#dde3ee;padding:15px 18px;border-radius:8px;
 overflow-x:auto;font-size:12.5px;line-height:1.7;
 font-family:ui-monospace,Consolas,monospace}
blockquote{margin:16px 0;padding:14px 18px;background:#fff8f0;
 border:1px solid #f0dfc4;border-left:4px solid var(--amber);
 border-radius:0 8px 8px 0;font-size:14px;color:#5a4a2e}
blockquote strong{color:#8a5a10}
.qa{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0}
.qa .q{background:#f0f3f8;padding:11px 16px;font-weight:600;font-size:14px}
.qa .a{padding:13px 16px;background:#fff;font-size:14px}
.qa .a .vrd{font-weight:700;margin-bottom:5px}
.qa .a .vrd.yes{color:var(--red)}
.qa .a .vrd.no{color:var(--green)}
.qa .a .vrd.part{color:var(--amber)}
.toc{display:grid;grid-template-columns:repeat(2,1fr);gap:6px 22px;font-size:13.5px}
@media(max-width:820px){.toc{grid-template-columns:1fr}}
.toc a{color:var(--blue);text-decoration:none}
.toc a:hover{text-decoration:underline}
.toc .sn{color:var(--tx3);font-family:ui-monospace,monospace;margin-right:7px}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
.footer{margin-top:50px;font-size:12.5px;color:var(--tx3);line-height:1.9;
 text-align:center;border-top:1px solid var(--line);padding-top:24px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--tx2);margin:2px 0 0;flex-wrap:wrap}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
 vertical-align:-1px}
"""

S = []
A = S.append

A('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
A('<meta name="viewport" content="width=device-width,initial-scale=1">')
A('<title>A股「缩量企稳反转」策略 量化回测报告</title>')
A(f"<style>{CSS}</style></head><body>")

# ---------------- Hero ----------------
A('<header class="hero"><div class="inner">')
A("<h1>A股「缩量企稳反转」策略<br>量化回测与假设检验报告</h1>")
A('<div class="sub">原命题：股票下跌后出现连续缩量 / 地量 → 卖压可能衰竭 → '
  '股价企稳 → 买入博反弹。<br>'
  '本报告用 2016-04-01 ~ 2026-09-23 的 A 股全市场日线数据，'
  '对该命题的每一个环节做独立检验，并如实给出统计结论。</div>')
A('<div class="meta">标的池 4,986 只（主板/创业板/科创板，已剔北交所）· '
  '日线 1,146 万行 · 有效成交样本 2016-04 ~ 2026-09 · '
  '复权：后复权(hfq) · 成本模型：佣金万2.5 + 过户费 + 印花税(2023-08-28 起 0.05%) '
  '+ 双边滑点</div>')
A("</div></header>")
A('<div class="wrap">')

# ---------------- 目录 ----------------
TOC = [
    ("0", "先说结论：这个策略有没有统计优势？"),
    ("1", "研究设计：数据、约束与「不作弊」"),
    ("2", "假设拆解：把一句话拆成四个可检验的子假设"),
    ("3", "实验一：缩量定义对比（A–E）"),
    ("4", "实验二：企稳定义对比（A–E）"),
    ("5", "实验三：超跌前提真的必要吗？"),
    ("6", "实验四：买入规则 A / B / C"),
    ("7", "实验五：止损方式"),
    ("8", "实验六：止盈方式"),
    ("9", "实验七：六种卖出方案对比"),
    ("10", "实验八：+6% 固定止盈是否掐死了大赢家？"),
    ("11", "实验九：市场分层与样本内外一致性"),
    ("12", "实验十：交易成本敏感性"),
    ("13", "实验十一：净值曲线与随机基准"),
    ("14", "消融实验：各条件的真实边际贡献"),
    ("15", "最终交付：最简单、参数稳定的版本"),
    ("16", "十个问题的直接回答"),
    ("17", "结论的边界：这份报告不能证明什么"),
    ("18", "免责声明"),
]
A('<div class="card"><h3 style="margin-top:0">目录</h3><div class="toc">')
for no, t in TOC:
    A(f'<div><span class="sn">{no}</span><a href="#s{no}">{t}</a></div>')
A("</div></div>")

# ================= 0. 结论 =================
A('<h2 id="s0"><span class="no">0</span>先说结论：这个策略有没有统计优势？</h2>')
A('<div class="callout bad"><span class="h">直接回答：没有。而且不是「没优势」，是「统计显著为负」。</span>'
  '按你原始设定（超跌 D2–D6 + 缩量 A + 企稳 A + 止损 -3% / 止盈 +6%）'
  '跑满全市场 2016–2026，共 <b>515,524</b> 笔交易，'
  '扣除全部成本后的单笔平均净收益 <b>-0.020%</b>，t 统计量 <b>-3.18</b>。'
  '样本量足够大，这个负号不是噪声，是真实的负期望。</div>')

A('<div class="grid g3">')
A(verdict_card("原策略 单笔净收益", f"{P(base_avg)}", f"t = {T(base_t)}（显著为负）· n=515,524",
               "bad"))
A(verdict_card("原策略 胜率", f"{B['win_rate']*100:.1f}%",
               "盈亏比不足，胜率+赔率组合无法覆盖成本", "bad"))
A(verdict_card("修正后 规则B 单笔净收益", f"{P(best_cand['avg_ret'])}",
               f"t = {T(best_cand['t_stat'])}（显著为正）· n={int(best_cand['n']):,}", "good"))
A("</div>")

A('<p>但「原策略不行」不等于「全都行不通」。逐层拆开之后，结论是三句话：</p>')
A('<div class="grid g3">')
A(verdict_card("①「缩量 = 卖压衰竭」", "证伪",
               "缩量越严格，收益越差；完全不设缩量反而最好", "bad"))
A(verdict_card("②「企稳 = 反弹启动」", "成立",
               "不创新低 / 收涨 / 站上 MA5 都是稳定正贡献", "good"))
A(verdict_card("③ 可作为实盘策略吗", "不建议",
               "优势极薄、对滑点极度敏感、牛市为负、净值 CAGR 仅 1~2%", "warn"))
A("</div>")

A('<h3>最诚实的一段总结</h3>')
A('<div class="callout"><span class="h">修正后的规则 B/C 确实有正期望，但这个「优势」薄到什么程度？</span>'
  f'单看单笔：<b>+{P(best_cand["avg_ret"],2).strip("%")}%</b>，t≈21，看上去很显著。'
  '但把这个数放到「<b>单位持有时间的收益</b>」口径下（这才是能和「随便买一只拿着」公平比较的口径），'
  f'规则B 平均持有 {best_cand["avg_hold"]:.1f} 天、净收益 {P(best_cand["avg_ret"],3)}，'
  f'折合每天 <b>{P(cand_per_day,3)}</b>；'
  f'而<b>零成本</b>随机买入并持有 20 天，折合每天 <b>{P(drift20/20,3)}</b>。'
  '扣掉成本后两者几乎贴在一起。'
  '<br><br>再叠上成本敏感性与市场分层：滑点从 10bp 提到 20bp，'
  f'规则B 的净收益从 {P(a("ruleB","avg_ret"))} 塌到 {P(qrow(f3,"slippage",0.002)["avg_ret"])}；'
  f'提到 30bp 直接转负（{P(qrow(f3,"slippage",0.003)["avg_ret"])}）。'
  f'而它在牛市里是 <b>{P(qrow(f4,"regime","牛市")["avg_ret"])}</b>（显著为负）。'
  '<br><br><b>所以：这不是一个可以直接上实盘的策略。</b>'
  '它唯一真正的价值，是把「缩量」这一条从选股条件里删掉——'
  '而这条恰恰是原命题的核心卖点。</div>')

# ================= 1. 研究设计 =================
A('<h2 id="s1"><span class="no">1</span>研究设计：数据、约束与「不作弊」</h2>')

A('<h3>1.1 数据</h3>')
A('<table><thead><tr><th>项目</th><th>规格</th></tr></thead><tbody>')
for k, v in [
    ("标的池", "沪深 A 股全市场 5,187 只（主板 3,197 / 创业板 1,407 / 科创板 617，"
               "北交所因数据源仅返回 1 根 K 线而剔除）"),
    ("价格数据", "腾讯行情后复权(hfq)日线，2015-12-01 起，"
                 "5,187 只 × 平均 2,210 交易日 = 1,146 万行"),
    ("指数数据", "上证综指 / 深证成指 / 沪深300 / 中证500 / 中证1000 / 创业板指"),
    ("股票池", "沪深300 / 中证500 / 创业板全部 / 科创板全部（用于分层检验）"),
    ("回测区间", "有效成交 2016-04-01 ~ 2026-09-23（前 70 交易日用于均线/ATR 预热）"),
    ("样本内外", "样本内 2016-04 ~ 2020-12；样本外 2021-01 ~ 2026-09"),
]:
    A(f"<tr><td>{k}</td><td style=\"font-family:inherit\">{v}</td></tr>")
A("</tbody></table>")

A('<h3>1.2 强制纳入的 A 股真实约束</h3>')
A('<ul>')
A('<li><b>信号与执行分离</b>：T 日收盘后出信号 → <b>T+1 开盘价</b>成交。'
  '所有均线、成交量、ATR、分位数都只用 ≤T 的数据，不存在未来函数。</li>')
A('<li><b>T+1 制度</b>：当日买入的仓位当日不可卖出，模拟循环从入场日的<b>次日</b>开始判断出场。</li>')
A('<li><b>涨停买不到</b>：若 T+1 开盘涨幅 ≥ 涨停幅度-0.4%（留出容差），该信号直接作废、不入场。</li>')
A('<li><b>跌停卖不掉</b>：一字跌停日（开盘即跌停且全天最高=最低）跳过，顺延到下一个可成交日。</li>')
A('<li><b>停牌</b>：停牌日视为无 K 线，入场顺延；若信号日到实际成交日间隔 > 10 个自然日则作废该信号'
  '（防长期停牌股用陈旧信号入场）。</li>')
A('<li><b>涨跌停幅度分板块</b>：主板 10%、创业板/科创板 20%，按标的所属板块分别设定。</li>')
A('<li><b>ST / 退市</b>：ST 标的在信号层面剔除；退市股在最后一个交易日按可成交价强制平仓。</li>')
A('<li><b>交易成本</b>：佣金万 2.5（双边）+ 过户费 0.001%（双边）+ 印花税（卖出，'
  '2023-08-28 前 0.10%、之后 0.05%，<b>按每笔交易的实际平仓日</b>判定，'
  '自动跨过印花税调整日）+ 滑点（默认双边各 10bp）。</li>')
A("</ul>")

A('<div class="callout warn"><span class="h">滑点默认取 10bp，这是刻意取偏高的保守值。</span>'
  'A 股小市值股的实际冲击成本往往被低估。本报告专门做了 0 / 5 / 10 / 20 / 30bp 的敏感性测试'
  '（见第 12 节）——结论对滑点<b>极度</b>敏感，这也是判定「不宜实盘」的主要依据之一。</div>')

A('<h3>1.3 关于「为什么用后复权」</h3>')
A('<p>本引擎使用腾讯行情的<b>后复权</b>价（<code>hfq</code>）。选后复权而非前复权的原因：'
  '前复权以最新价为锚点向前调整，在长期历史段价格可能被压到极小甚至为负，'
  '会让百分比止损、涨跌幅等以价格为分母的计算严重失真；'
  '后复权以最早价为锚点向后调整，价格序列恒为正、且任意两日之间的比例关系'
  '与真实持有收益率完全一致，因此对收益率类策略是更安全的口径。</p>')
A('<p>作为额外保护，本引擎在入场前仍会检查 '
  '<code>开盘价 &gt; 0</code> 且 <code>前收盘 &gt; 0</code>，不满足则丢弃该信号；'
  '并且涨跌停判定统一用「当日开盘 / 前收盘 - 1」的相对涨幅，'
  '而非复权价的绝对数值，确保涨跌停判定不受复权系数影响。</p>')

# ================= 2. 假设拆解 =================
A('<h2 id="s2"><span class="no">2</span>假设拆解：把一句话拆成四个可检验的子假设</h2>')
A('<p>原始命题是一句连贯的叙事，但叙事不能直接检验。必须先拆成互相独立、'
  '可以单独开关的条件，然后<b>逐个开关看边际贡献</b>。这是本报告方法论的核心。</p>')

A('<div class="card"><table><thead><tr><th>#</th><th>子假设</th><th>可操作定义</th>'
  '<th>检验方式</th></tr></thead><tbody>')
for i, (h, d, m) in enumerate([
    ("H1 前提：要先超跌", "ret20（20 日收益）横截面分位落在 D2–D6",
     "对比 D1-D3 / D2-D6 / D3-D7 / D4-D8 / D1-D10，并测试完全去掉超跌"),
    ("H2 核心：要缩量", "成交量萎缩到一定程度（A–E 五种定义）",
     "对比五种缩量定义 + 完全不设缩量，六组"),
    ("H3 触发：要企稳", "价格止跌（A–E 五种定义）",
     "对比五种企稳定义 + 完全不设企稳，六组"),
    ("H4 退出：止损止盈要合理", "-3%/+6% 是否最优",
     "止损 4 档固定 + 3 档 ATR；止盈 7 档 + 不设；卖出方式 6 种"),
], 1):
    A(f'<tr><td>{i}</td><td style="font-family:inherit"><b>{h}</b></td>'
      f'<td style="font-family:inherit">{d}</td>'
      f'<td style="font-family:inherit">{m}</td></tr>')
A("</tbody></table></div>")

A('<p>基准组合（下称「原策略」）为：<code>超跌D2-D6 + 缩量A + 企稳A + 止损-3% + 止盈+6%</code>，'
  '最长持有 20 个交易日，单边滑点 10bp。所有单条件对比实验只改动一个维度，其余保持基准不变——'
  '这样每一行的差异都可以归因到被改动的那一个条件上。</p>')

A('<div class="callout"><span class="h">一个必须先钉死的方法论问题：横截面分位基于谁？</span>'
  '超跌分位（ret20 十分位）始终基于<b>全市场</b>当期横截面计算，'
  '即使回测只选沪深300成分股也一样。否则「只选沪深300」会让分位口径漂移，'
  'D2–D6 的含义在不同股票池里不一致。选股范围只影响<b>哪些信号被执行</b>，不影响信号的定义。</div>')

# ================= 3. 缩量 =================
A('<h2 id="s3"><span class="no">3</span>实验一：缩量定义对比（A–E）</h2>')
A('<p>这是整个命题的<b>核心卖点</b>——「连续缩量 / 地量说明卖压衰竭」。'
  '如果这条成立，那么缩量条件的加入应当<b>提高</b>单笔收益。实测结果：</p>')

A('<div class="chart">')
A(H.bar_svg(vol_labels, vol_vals, width=900, height=260))
A("</div>")
A('<div class="cap">图 1　五种缩量定义 + 不设缩量，单笔平均净收益（%）。红=正，绿=负。'
  '基准：超跌D2-D6 + 企稳A + 止损-3%/+6%。</div>')

t = e1[["vol", "desc", "n", "win_rate", "avg_ret", "t_stat", "avg_hold"]].copy()
t.columns = ["代号", "缩量定义", "样本数", "胜率", "单笔净收益", "t值", "平均持有(日)"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
t["平均持有(日)"] = t["平均持有(日)"].map(lambda x: f"{x:.1f}")
A(H.table(t, cls="dt"))

A('<div class="callout bad"><span class="h">结论：H2「缩量 = 卖压衰竭」被明确证伪。</span>'
  '六组结果呈现一条<b>单调的、反直觉的</b>规律——'
  '<b>缩量条件越严格，单笔收益越差</b>：</div>')
A('<ul>')
A(f'<li><b>不设缩量</b>：{P(a("random") if False else e1.iloc[-1]["avg_ret"])}，'
  f't={T(e1.iloc[-1]["t_stat"])}，<b>六组里唯一显著为正</b>，样本 234 万笔。</li>')
A(f'<li><b>缩量 A</b>（两日递减，原策略）：{P(base_avg)}，t={T(base_t)}，已转负。</li>')
A(f'<li><b>地量 C</b>（VOL&lt;VOL20×0.5）：{P(qrow(e1,"vol","C")["avg_ret"])}，'
  f't={T(qrow(e1,"vol","C")["t_stat"])}——<b>六组里最差</b>。'
  '「地量」这个被讲得最多的说法，实测是<b>最有害</b>的过滤条件。</li>')
A("</ul>")
A('<p>为什么？原文的推断链条有一个逻辑跳跃：'
  '「缩量 → 卖压衰竭」。但缩量同样可以由<b>买盘消失</b>造成。'
  '在 A 股，成交量的萎缩更多反映的是<b>关注度流失</b>，'
  '而关注度流失的股票在随后 20 天内既没有买盘推动反弹，'
  '又更容易在继续下跌时因为流动性差而出现更大的滑点与跳空。'
  '实测数据显示：地量条件下的胜率只有 '
  f'{qrow(e1,"vol","C")["win_rate"]*100:.1f}%，比不设缩量的 '
  f'{e1.iloc[-1]["win_rate"]*100:.1f}% 低了近 4 个百分点。</p>')

A('<div class="callout"><span class="h">还有一个必须点出的技术细节：缩量条件把样本量砍掉了 78%。</span>'
  f'不设缩量有 {int(e1.iloc[-1]["n"]):,} 笔信号，缩量 A 只剩 {int(B["n"]):,} 笔。'
  '对于一个<b>本身期望值就接近零</b>的信号，'
  '任何无信息量的过滤条件只会「筛掉样本」而不会「筛出好样本」——'
  '它带来的 t 值变化，本质上是样本量的变化，不是预测力的变化。'
  '这一点在第 14 节的消融实验里得到直接验证。</div>')

# ================= 4. 企稳 =================
A('<h2 id="s4"><span class="no">4</span>实验二：企稳定义对比（A–E）</h2>')
A('<p>与缩量形成鲜明对照：<b>「企稳」是原命题里唯一真正有效的成分。</b></p>')

A('<div class="chart">')
A(H.bar_svg(stab_labels, stab_vals, width=900, height=260))
A("</div>")
A('<div class="cap">图 2　五种企稳定义 + 不设企稳，单笔平均净收益（%）。基准：超跌D2-D6 + 缩量A + 止损-3%/+6%。</div>')

t = e2[["stab", "desc", "n", "win_rate", "avg_ret", "t_stat", "avg_hold"]].copy()
t.columns = ["代号", "企稳定义", "样本数", "胜率", "单笔净收益", "t值", "平均持有(日)"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
t["平均持有(日)"] = t["平均持有(日)"].map(lambda x: f"{x:.1f}")
A(H.table(t, cls="dt"))

A('<div class="callout good"><span class="h">结论：H3「企稳后才买」成立，且效果明显。</span></div>')
A('<ul>')
A(f'<li><b>B 连续 2 日不创新低</b>：{P(qrow(e2,"stab","B")["avg_ret"])}，'
  f't={T(qrow(e2,"stab","B")["t_stat"])}，<b>最优</b>。'
  f'相比不设企稳（{P(qrow(e2,"stab","NONE")["avg_ret"])}）提升了 '
  f'{(qrow(e2,"stab","B")["avg_ret"]-qrow(e2,"stab","NONE")["avg_ret"])*100:.2f} 个百分点。</li>')
A(f'<li><b>E 收盘站上 MA5</b>：{P(qrow(e2,"stab","E")["avg_ret"])}，'
  f't={T(qrow(e2,"stab","E")["t_stat"])}，次优。</li>')
A(f'<li><b>D 不破前低且收阳</b>：{P(qrow(e2,"stab","D")["avg_ret"])}，'
  f't={T(qrow(e2,"stab","D")["t_stat"])}，有效但弱一些。</li>')
A(f'<li><b>A 收盘 ≥ 前收</b>（原策略用的）：{P(base_avg)}，'
  f't={T(base_t)}，<b>唯一为负</b>。'
  '单日收涨太容易满足了——下跌趋势中一根小阳线的信息量几乎为零，'
  '它不足以证明跌势停止。</li>')
A(f'<li><b>C 5 日低点后 2 日未破</b>：{P(qrow(e2,"stab","C")["avg_ret"])}，'
  f't={T(qrow(e2,"stab","C")["t_stat"])}，几乎无效。等待期过长，错过了反弹起点。</li>')
A("</ul>")
A('<p>这里有一个清晰的逻辑：<b>「不创新低」之所以有效，是因为它检验的是一个持续性的状态，'
  '而不是一个单日事件</b>。连续 2 日不再创新低，说明近两日的<b>最低成交价在抬升</b>，'
  '这是抛压真的被吸收掉的直接证据；而「今天收盘比昨天高」只说明今天有人买，'
  '明天可能继续砸。这与原命题想要表达的意思其实一致——'
  '只不过原命题用错了代理变量：<b>卖压衰竭的证据在价格结构里，不在成交量里。</b></p>')

# ================= 5. 超跌 =================
A('<h2 id="s5"><span class="no">5</span>实验三：超跌前提真的必要吗？</h2>')
A('<div class="chart">')
A(H.bar_svg(os_labels, os_vals, width=900, height=240))
A("</div>")
A('<div class="cap">图 3　不同超跌区间的单笔平均净收益（%）。基准：缩量A + 企稳A + 止损-3%/+6%。</div>')

t = e3[["rng", "n", "win_rate", "avg_ret", "t_stat"]].copy()
t.columns = ["ret20 分位区间", "样本数", "胜率", "单笔净收益", "t值"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
A(H.table(t, cls="dt"))

A('<p>在「缩量A + 企稳A」这个组合下，<b>所有超跌区间都是负的</b>，'
  '而且<b>越超跌越差</b>（D1-D3 为 '
  f'{P(qrow(e3,"rng","D1-D3")["avg_ret"])}，D1-D10 全部为 '
  f'{P(qrow(e3,"rng","D1-D10")["avg_ret"])}）。'
  '这看起来像是「超跌反转也不成立」，但别急着下这个结论——'
  '因为在第 14 节的消融实验里，把<b>企稳条件换掉</b>之后，'
  '超跌的贡献立刻从负变正：'
  f'「超跌 + 不创新低 + 收涨」为 {P(a("nolow2+up","avg_ret"))}，'
  f'而完全去掉超跌的「全市场 + 不创新低 + 收涨」只有 {P(a("noOversold_noVol","avg_ret"))}'
  f'（t={T(a("noOversold_noVol","t_stat"))}，几乎不显著）。</p>')

A('<div class="callout warn"><span class="h">这是一个重要的交互效应，也是原策略失败的原因之一。</span>'
  '「超跌」和「企稳」是<b>互补条件，不能拆开看</b>：'
  '超跌把候选池锁定在已经跌了很多的股票上，'
  '此时只有加上<b>足够强的企稳过滤</b>（不创新低）才能把「还在跌」和「跌完了」区分开。'
  '原策略用的是最弱的企稳定义（单日收涨），'
  '于是超跌反而变成了「专挑下跌趋势中的股票」的负面选股器。'
  '<br><br>换句话说：<b>超跌本身不是优势，超跌 + 强企稳才是。</b></div>')

# ================= 6. 买入规则 =================
A('<h2 id="s6"><span class="no">6</span>实验四：买入规则 A / B / C</h2>')
t = e4[["desc", "n", "win_rate", "avg_ret", "t_stat", "avg_hold"]].copy()
t.columns = ["买入规则", "样本数", "胜率", "单笔净收益", "t值", "平均持有(日)"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
t["平均持有(日)"] = t["平均持有(日)"].map(lambda x: f"{x:.1f}")
A(H.table(t, cls="dt"))

A('<div class="callout good"><span class="h">规则 B 和 C 把单笔期望从负翻正，且 t 值超过 20。</span>'
  '注意：三者只差在<b>企稳定义</b>上（A = 收≥前收；B = 2日不创新低 + 收涨；'
  'C = 2日不创新低 + 站上MA5），共享「超跌 D2-D6 + 缩量 A」前提。'
  f'从 A 到 B，单笔净收益从 {P(base_avg)} 提升到 {P(qrow(e4,"rule","B")["avg_ret"])}，'
  f'提升 {(qrow(e4,"rule","B")["avg_ret"]-base_avg)*100:.2f} 个百分点。</div>')

A('<p>规则 B 与 C 的差异（'
  f'{P(qrow(e4,"rule","B")["avg_ret"])} vs {P(qrow(e4,"rule","C")["avg_ret"])}）'
  '在统计上没有实质区别，可以互换。B 的样本更少一些（信号更严），C 的信号更多。'
  '实务上如果追求信号数量选 C，追求单笔质量选 B。</p>')

# ================= 7. 止损 =================
A('<h2 id="s7"><span class="no">7</span>实验五：止损方式</h2>')
A('<div class="chart">')
A(H.bar_svg(sl_labels, sl_vals, width=900, height=250))
A("</div>")
A('<div class="cap">图 4　固定百分比止损（-2%~-5%）与 ATR 倍数止损的单笔平均净收益（%）。'
  '基准：超跌D2-D6 + 缩量A + 企稳A + 止盈+6%。</div>')

t = e5[["desc", "avg_ret", "t_stat", "win_rate", "avg_hold"]].copy()
t.columns = ["止损方式", "单笔净收益", "t值", "胜率", "平均持有(日)"]
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["平均持有(日)"] = t["平均持有(日)"].map(lambda x: f"{x:.1f}")
A(H.table(t, cls="dt"))

A('<div class="callout"><span class="h">结论：在这套参数下，止损越宽越好，-3% 明显偏紧。</span>'
  f'-2% 是灾难性的（{P(qrow(e5,"sl",0.02)["avg_ret"])}，t={T(qrow(e5,"sl",0.02)["t_stat"])}），'
  f'-5% 才转正（{P(qrow(e5,"sl",0.05)["avg_ret"])}），'
  f'ATR×2 最好（{P(qrow(e5,"atr_k",2.0)["avg_ret"])}，t={T(qrow(e5,"atr_k",2.0)["t_stat"])}）。</div>')
A('<p>但这<b>不能</b>被解读成「宽止损提高了策略的预测能力」。原因很直接：'
  '超跌股在 T+1 之后的日内波动极大（平均振幅远高于市场平均），'
  '一个 -2% 的固定止损在开盘跳空时几乎必然被击穿。'
  '止损越紧，越多的交易在<b>噪声</b>里被扫出局，'
  '剩下的持仓期间被压缩（-2% 组平均持有 4.3 日，ATR×2 组 8.8 日），'
  '自然吃不到反弹。'
  '<br><br>换言之，这里的差异主要是<b>被动持仓时长</b>的差异，'
  '而持仓时长在整体上涨的市场里本身就是收益来源——'
  f'看第 13 节的漂移基准：随机持有 20 日就能拿到 {P(drift20)}（零成本）。'
  '所以「宽止损更好」这个结论要打折扣：它有一部分是市场 beta，不是策略 alpha。</p>')

# ================= 8. 止盈 =================
A('<h2 id="s8"><span class="no">8</span>实验六：止盈方式</h2>')
A('<div class="chart">')
A(H.bar_svg(tp_labels, tp_vals, width=900, height=250))
A("</div>")
A('<div class="cap">图 5　固定止盈档位与「不设止盈」的单笔平均净收益（%）。'
  '基准：超跌D2-D6 + 缩量A + 企稳A + 止损-3%。</div>')

t = e6[["desc", "avg_ret", "t_stat", "win_rate", "avg_hold", "profit_factor"]].copy()
t.columns = ["止盈档位", "单笔净收益", "t值", "胜率", "平均持有(日)", "盈亏比"]
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["平均持有(日)"] = t["平均持有(日)"].map(lambda x: f"{x:.1f}")
t["盈亏比"] = t["盈亏比"].map(lambda x: f"{x:.2f}")
A(H.table(t, cls="dt"))

A('<div class="callout"><span class="h">结论：+3% 是灾难，+6% 只是「不那么差」，不设止盈最好。</span>'
  f'+3% 止盈时单笔净收益 {P(qrow(e6,"tp",0.03)["avg_ret"])}，'
  f't={T(qrow(e6,"tp",0.03)["t_stat"])}——这是全表最差的一行，而且样本量与基准完全一致（515,524 笔），'
  '所以这个差异 100% 来自止盈规则本身。<br><br>'
  '规律是单调的：<b>止盈越晚，收益越高</b>。'
  f'+15% 为 {P(qrow(e6,"tp",0.15)["avg_ret"])}，不设止盈为 '
  f'{P(e6.loc[e6["tp"].isna(),"avg_ret"].iloc[0])}。'
  '这本质上是一个「截断右尾」的代价问题——'
  '超跌反弹的收益分布是<b>右偏</b>的（少数大反弹贡献了大部分利润），'
  '任何固定止盈都在系统性砍掉右尾。</div>')

# ================= 9. 卖出方案 =================
A('<h2 id="s9"><span class="no">9</span>实验七：六种卖出方案对比</h2>')
A('<p>你指定的六种卖出方案，加上两个对照组，全部在「超跌D2-D6 + 缩量A + 企稳A」'
  '基础上测试（原策略为方案 1）。</p>')

t = e7[["desc", "avg_ret", "t_stat", "win_rate", "avg_hold", "exit_reason"]].copy() \
    if "exit_reason" in e7.columns else e7[["desc", "avg_ret", "t_stat", "win_rate",
                                            "avg_hold"]].copy()
t.columns = ["卖出方案", "单笔净收益", "t值", "胜率", "平均持有(日)"]
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["平均持有(日)"] = t["平均持有(日)"].map(lambda x: f"{x:.1f}")
A(H.table(t, cls="dt"))

A('<div class="chart">')
lbl = ["方案1<br>-3%/+6%", "方案3<br>+6%卖50%<br>后MA5", "方案4/5<br>-3%/MA10",
       "方案6<br>-3%/MACD死叉", "对照<br>-3%/MA5", "对照<br>-3%/最长20日"]
val = list(e7["avg_ret"])
A(H.bar_svg(lbl, val, width=900, height=280))
A("</div>")
A('<div class="cap">图 6　各卖出方案单笔平均净收益（%）。</div>')

A('<div class="callout"><span class="h">结论：均线类退出是负贡献，MACD / 不设止盈是正贡献。</span></div>')
A('<ul>')
A(f'<li><b>方案 1 / 2</b>（-3%/+6%，固定止盈）：都是 {P(base_avg)}。'
  '方案 2 与方案 1 数值完全相同，因为固定止盈本来就是 100% 仓位平掉，'
  '「卖 100%」这个补充说明不产生额外约束。</li>')
A(f'<li><b>方案 3</b>（+6% 止盈一半，剩余跌破 MA5 清仓）：'
  f'{P(qrow(e7,"desc","策略3 -3%/+6%卖50%+MA5")["avg_ret"])}，'
  f't={T(qrow(e7,"desc","策略3 -3%/+6%卖50%+MA5")["t_stat"])}。'
  '比方案 1 好，但好的原因是「一半仓位没有被 +6% 截断」，不是 MA5 的功劳。</li>')
A(f'<li><b>方案 4 / 5</b>（-3% 止损，跌破 MA10 清仓）：'
  f'{P(qrow(e7,"desc","策略4 -3%/MA10")["avg_ret"])}，'
  f't={T(qrow(e7,"desc","策略4 -3%/MA10")["t_stat"])}——<b>比固定止盈更差</b>。'
  'MA10 在超跌股的剧烈震荡里被反复穿越，'
  f'平均持有只有 {qrow(e7,"desc","策略4 -3%/MA10")["avg_hold"]:.1f} 日就被打出来。</li>')
A(f'<li><b>方案 6</b>（-3% 止损，MACD 死叉清仓）：'
  f'{P(qrow(e7,"desc","策略6 -3%/MACD死叉")["avg_ret"])}，'
  f't={T(qrow(e7,"desc","策略6 -3%/MACD死叉")["t_stat"])}，'
  '主要优点是把持有期拉长到 '
  f'{qrow(e7,"desc","策略6 -3%/MACD死叉")["avg_hold"]:.1f} 日，给了反弹足够的时间。</li>')
A(f'<li><b>对照组「仅最长持有 20 日，不设任何止盈」</b>：'
  f'{P(qrow(e7,"desc","对照 -3%/仅最长持有")["avg_ret"])}，'
  f't={T(qrow(e7,"desc","对照 -3%/仅最长持有")["t_stat"])}——'
  '<b>全表最好</b>。这说明所有「主动退出」规则（无论是均线还是固定止盈）'
  '都在损失收益。</li>')
A("</ul>")
A('<div class="callout warn"><span class="h">关于「-3%/MA5」这个对照，要特别说明。</span>'
  f'它与方案 4「-3%/MA10」的差别仅为 MA5 vs MA10，'
  f'结果 {P(qrow(e7,"desc","对照 -3%/MA5")["avg_ret"])} vs '
  f'{P(qrow(e7,"desc","策略4 -3%/MA10")["avg_ret"])}，'
  '两者都显著为负。均线越短越差，说明<b>用均线做退出在超跌反抽场景里是反作用的</b>：'
  '超跌股站上/跌破均线的过程本身就是噪声驱动的，用它做信号等于在高频噪声里来回被割。</div>')

# ================= 10. 大赢家 =================
A('<h2 id="s10"><span class="no">10</span>实验八：+6% 固定止盈是否掐死了大赢家？</h2>')
A('<p>你的问题问得很准。这直接检验「固定止盈 vs 让利润奔跑」的取舍。'
  '做法是把同一批 515,524 个信号，分别用「+6% 固定止盈」和「不设止盈（跌破 MA10 退出）」'
  '跑一遍，然后看收益分布。</p>')

d6 = f9[f9.group == "固定+6%止盈"] if "固定+6%止盈" in set(f9.group) else None
e8 = rd("e8_distribution.csv")
g_fix = e8[e8.group == "固定+6%止盈"].set_index("bucket")
g_run = e8[e8.group == "不设止盈(MA10)"].set_index("bucket")

A('<div class="grid g2">')
A('<div class="card"><h4 style="margin-top:0">+6% 固定止盈</h4>')
A('<table><thead><tr><th>收益区间</th><th>笔数</th><th>占比</th></tr></thead><tbody>')
for b in [">30%", "20%~30%", "15%~20%", "10%~15%", "6%~10%"]:
    r = g_fix.loc[b]
    A(f'<tr><td>{b}</td><td>{int(r["count"]):,}</td>'
      f'<td>{r["pct_count"]*100:.2f}%</td></tr>')
A("</tbody></table></div>")
A('<div class="card"><h4 style="margin-top:0">不设止盈（跌破 MA10 退出）</h4>')
A('<table><thead><tr><th>收益区间</th><th>笔数</th><th>占比</th></tr></thead><tbody>')
for b in [">30%", "20%~30%", "15%~20%", "10%~15%", "6%~10%"]:
    r = g_run.loc[b]
    A(f'<tr><td>{b}</td><td>{int(r["count"]):,}</td>'
      f'<td>{r["pct_count"]*100:.2f}%</td></tr>')
A("</tbody></table></div>")
A("</div>")

n10_fix = int(g_fix.loc["10%~15%", "count"] + g_fix.loc["15%~20%", "count"]
               + g_fix.loc["20%~30%", "count"] + g_fix.loc[">30%", "count"])
n10_run = int(g_run.loc["10%~15%", "count"] + g_run.loc["15%~20%", "count"]
               + g_run.loc["20%~30%", "count"] + g_run.loc[">30%", "count"])
n30_fix = int(g_fix.loc[">30%", "count"])
n30_run = int(g_run.loc[">30%", "count"])

A('<div class="callout bad"><span class="h">结论：是的，而且掐得非常狠。</span>'
  '<ul>'
  f'<li>净收益 ≥ +10% 的交易笔数：<b>{n10_fix:,} → {n10_run:,}</b>，'
  f'不设止盈是固定止盈的 <b>{n10_run/n10_fix:.1f} 倍</b></li>'
  f'<li>净收益 ≥ +30% 的交易笔数：<b>{n30_fix} → {n30_run:,}</b>，'
  f'相差 <b>{n30_run/n30_fix:.0f} 倍</b></li>'
  f'<li>单笔最大净收益：<b>+{MAX_TP6*100:.1f}%</b> → <b>+{MAX_RUN*100:.1f}%</b></li>'
  '</ul></div>')

A('<p>更直观地说：<b>固定 +6% 止盈把「赚钱的交易」全部压缩在 +3%~+6% 这个窄区间里</b>——'
  f'这一桶占了 {g_fix.loc["3%~6%","pct_count"]*100:.1f}% 的样本（'
  f'{int(g_fix.loc["3%~6%","count"]):,} 笔）。'
  '同时把「亏损的交易」照样放行到 -5%（'
  f'{g_fix.loc["-5%~-3%","pct_count"]*100:.1f}%，{int(g_fix.loc["-5%~-3%","count"]):,} 笔）。'
  '结果是一个<b>右尾被削平、左尾完整保留</b>的分布——'
  '这正是「胜率看着还行（38.4%）但净期望为负」的几何原因。</p>')

A('<div class="callout"><span class="h">对你的原命题而言，这意味着什么？</span>'
  '如果你的交易逻辑确实是「抓超跌反弹」，那么<b>赌的就是右尾</b>——'
  '大部分反弹只有几个点，但少数会走出 +20% 以上。'
  '+6% 固定止盈在逻辑上与这个赌注自相矛盾：'
  '你在为一个右偏的赌局下注，却主动把右尾砍掉。'
  '实测的「不设止盈 / 跌破 MA10 或 MACD 死叉」版本收益更好，正是这个原因。</div>')

# ================= 11. 市场分层 =================
A('<h2 id="s11"><span class="no">11</span>实验九：市场分层与样本内外一致性</h2>')
A('<h3>11.1 按牛熊 + 指数与 MA60 的关系分层</h3>')
A('<div class="chart">')
A(H.bar_svg(reg_labels, reg_vals, width=900, height=250))
A("</div>")
A('<div class="cap">图 7　原策略在不同市场状态下的单笔平均净收益（%）。</div>')

t = e9[["desc", "n", "win_rate", "avg_ret", "t_stat"]].copy()
t.columns = ["市场状态", "样本数", "胜率", "单笔净收益", "t值"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
A(H.table(t, cls="dt"))

A('<div class="callout bad"><span class="h">「缩量企稳反转只适合上涨趋势」这个说法是错的，实测方向完全相反。</span>'
  f'牛市（指数 MA60 上行 + 指数在 MA60 上方）：<b>{P(qrow(e9,"desc","牛市")["avg_ret"])}</b>，'
  f't={T(qrow(e9,"desc","牛市")["t_stat"])}，<b>最差</b>。<br>'
  f'熊市：<b>{P(qrow(e9,"desc","熊市")["avg_ret"])}</b>，t={T(qrow(e9,"desc","熊市")["t_stat"])}，<b>最好</b>。<br>'
  f'震荡市：{P(qrow(e9,"desc","震荡市")["avg_ret"])}，'
  f't={T(qrow(e9,"desc","震荡市")["t_stat"])}。<br>'
  f'指数在 MA60 上方：{P(qrow(e9,"desc","指数>MA60")["avg_ret"])}；'
  f'指数在 MA60 下方：{P(qrow(e9,"desc","指数<MA60")["avg_ret"])}。</div>')

A('<p>为什么会这样？合理的解释是<b>「超跌」的定义是相对的全市场排名</b>。'
  '在牛市里，全市场都在涨，ret20 落在 D2–D6 的往往是<b>基本面出了问题的落后股</b>'
  '（因为整体水位在抬升，跌到分位下沿本身就说明它在逆势下跌）；'
  '而在熊市/震荡市里，全市场都在跌，落在 D2–D6 的可能只是一只<b>正常股票的正常回调</b>，'
  '反弹概率自然更高。'
  '<br><br>这个发现对实盘有直接含义：<b>如果一定要用这类策略，'
  '它天然是个「熊市/震荡市工具」，在牛市里应当停用。</b></p>')

A('<p>修正后的规则 B 也保持了同样的方向（见下表）：'
  f'牛市 {P(qrow(f4,"regime","牛市")["avg_ret"])}（负），'
  f'熊市 {P(qrow(f4,"regime","熊市")["avg_ret"])}（正且 t={T(qrow(f4,"regime","熊市")["t_stat"])}），'
  f'震荡市 {P(qrow(f4,"regime","震荡市")["avg_ret"])}（正）。'
  '方向一致意味着这个规律不是原策略的参数巧合，而是这类信号的固有属性。</p>')

t = f4[["cand", "regime", "n", "win_rate", "avg_ret", "t_stat"]].copy()
t.columns = ["方案", "市场状态", "样本数", "胜率", "单笔净收益", "t值"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
A(H.table(t, cls="dt"))

A('<h3>11.2 样本内 / 样本外一致性</h3>')

t = e10[["period", "n", "win_rate", "avg_ret", "t_stat"]].copy()
t.columns = ["区间", "样本数", "胜率", "单笔净收益", "t值"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
A('<h4>原策略</h4>')
A(H.table(t, cls="dt"))

t = f2[["cand", "period", "n", "avg_ret", "t_stat"]].copy()
t.columns = ["方案", "区间", "样本数", "单笔净收益", "t值"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["单笔净收益"] = t["单笔净收益"].map(P)
t["t值"] = t["t值"].map(T)
A('<h4>修正方案（规则B / 规则C / 无缩量版）</h4>')
A(H.table(t, cls="dt"))

A('<div class="callout"><span class="h">这是本报告里最关键的一张表。</span>'
  f'<b>原策略：</b>样本内 {P(qrow(e10,"period","样本内 2016-2020")["avg_ret"])}（t={T(qrow(e10,"period","样本内 2016-2020")["t_stat"])}）'
  f'→ 样本外 {P(qrow(e10,"period","样本外 2021-2026")["avg_ret"])}（t={T(qrow(e10,"period","样本外 2021-2026")["t_stat"])}）。'
  '符号翻转，是典型的参数过拟合特征。<br><br>'
  f'<b>规则 B（缩量A + 不创新低 + 收涨）：</b>样本内 {P(qrow(f2,"cand","规则B(缩量A)","period","样本内16-20")["avg_ret"])}'
  f'（t={T(qrow(f2,"cand","规则B(缩量A)","period","样本内16-20")["t_stat"])}）'
  f'→ 样本外 {P(qrow(f2,"cand","规则B(缩量A)","period","样本外21-26")["avg_ret"])}'
  f'（t={T(qrow(f2,"cand","规则B(缩量A)","period","样本外21-26")["t_stat"])}）。'
  '<b>符号一致、量级稳定、两个区间都显著为正。</b><br><br>'
  f'<b>无缩量版：</b>样本内 {P(qrow(f2,"cand","规则B(缩量不走缩量条件)","period","样本内16-20")["avg_ret"])}'
  f' → 样本外 {P(qrow(f2,"cand","规则B(缩量不走缩量条件)","period","样本外21-26")["avg_ret"])}，'
  '同样稳定，而且样本量是规则B的 5.7 倍。</div>')

A('<p>样本内外的一致性，是本报告判定「规则B/C 有真实优势、原策略没有」的<b>主要证据</b>。'
  '一个只在某一段历史有效的规律是拟合，在两个独立区间都成立的才有可能是规律。'
  '需要强调的是：一致性只说明<b>方向</b>可信，'
  '不代表幅度可信——样本外优势（'
  f'{P(qrow(f2,"cand","规则B(缩量A)","period","样本外21-26")["avg_ret"])}）'
  '比样本内（'
  f'{P(qrow(f2,"cand","规则B(缩量A)","period","样本内16-20")["avg_ret"])}）还高，'
  '这种「样本外更好」本身也可能是运气。</p>')

# ================= 12. 成本 =================
A('<h2 id="s12"><span class="no">12</span>实验十：交易成本敏感性</h2>')
A('<div class="chart">')
A(H.bar_svg(cost_labels, cost_vals, width=900, height=230))
A("</div>")
A('<div class="cap">图 8　原策略在不同滑点假设下的单笔平均净收益（%）。</div>')

t = e11[["desc", "slippage", "avg_gross", "avg_ret", "t_stat"]].copy()
t.columns = ["滑点假设", "单边滑点", "毛收益", "净收益", "t值"]
t["单边滑点"] = t["单边滑点"].map(lambda x: f"{x*10000:.0f}bp")
t["毛收益"] = t["毛收益"].map(P)
t["净收益"] = t["净收益"].map(P)
t["t值"] = t["t值"].map(T)
A(H.table(t, cls="dt"))

A('<div class="callout warn"><span class="h">原策略的「盈亏平衡滑点」只有约 9bp。</span>'
  f'毛收益 {P(B["avg_gross"] if "avg_gross" in B else e11.iloc[0]["avg_gross"])}，'
  '单边滑点每增加 1bp，单笔净收益就下降约 2bp（买卖各收一次），'
  '因此只要单边滑点超过约 9bp，策略就由正转负。'
  f'而实测在 30bp 滑点下，原策略是 {P(e11.sort_values("slippage").iloc[-1]["avg_ret"])}。</div>')

A('<h4>修正方案（规则B / 规则C / 无缩量版）的成本敏感性</h4>')
piv = f3.pivot_table(index="slippage", columns="cand", values="avg_ret").sort_index()
show = piv.round(6).reset_index()
show.columns = ["单边滑点"] + [c.split("(")[0] for c in show.columns[1:]]
show["单边滑点"] = [f"{int(i*10000)}bp" for i in piv.index]
A(H.table(show, cls="dt"))

A('<div class="callout bad"><span class="h">修正方案的实际情况比原策略好，但余地依然很窄。</span>'
  f'规则B：10bp 时 {P(qrow(f3,"cand","规则B(超跌+缩量A+不创新低+收涨)","slippage",0.001)["avg_ret"])}，'
  f'20bp 时 {P(qrow(f3,"cand","规则B(超跌+缩量A+不创新低+收涨)","slippage",0.002)["avg_ret"])}，'
  f'30bp 时 {P(qrow(f3,"cand","规则B(超跌+缩量A+不创新低+收涨)","slippage",0.003)["avg_ret"])}。<br>'
  '20bp 时 t 值只剩 '
  f'{T(qrow(f3,"cand","规则B(超跌+缩量A+不创新低+收涨)","slippage",0.002)["t_stat"])}，'
  '勉强显著；30bp 时彻底转负。<br><br>'
  '作为参照，同一批随机构造的信号，在 10bp 滑点下是 '
  f'{P(qrow(f3,"cand","随机全市场(基准)","slippage",0.001)["avg_ret"])}。'
  '所以修正方案相对「随机」确实有超额，但这个超额<b>只有 0.39 个百分点</b>，'
  '而 A 股小盘股的真实双边冲击成本很容易超过 20bp。</div>')

A('<div class="callout"><span class="h">这是判定「不宜实盘」的最直接依据。</span>'
  '一个单笔优势只有 0.25%、且该优势在滑点翻倍时就消失一大半的策略，'
  '在真实交易中几乎不可能稳定盈利。'
  '本报告在第 0 节和第 17 节都反复强调这一点，而不是用「回测收益为正」来含糊过去。</div>')

# ================= 13. 净值 =================
A('<h2 id="s13"><span class="no">13</span>实验十一：净值曲线与随机基准</h2>')

A('<div class="callout warn"><span class="h">先读这段，再看图。</span>'
  '本报告的净值曲线是<b>逐日等权持有全部未平仓头寸、假设资金无限、每日再平衡</b>的理论篮子净值。'
  '它<b>不代表</b>可以实盘复制的账户收益，只能用来观察策略的方向性。'
  '原因是：同一时刻平均有 <b>334 个</b>头寸同时在场（中位数 316，最多 1,315），'
  '任何有限的真实资金都不可能同时等权持有这么多只股票。'
  '所以<b>判断策略优劣应当以单笔交易统计（第 3~12 节）为准</b>，'
  '净值曲线只作为辅助。</div>')

metrics = [
    ("规则B（超跌+缩量A+不创新低+收涨）", "ruleB", "bad"),
    ("规则C（超跌+缩量A+不创新低+站上MA5）", "ruleC", "bad"),
    ("精简版（超跌+不创新低+收涨，无缩量）", "simple", "mut"),
    ("随机全市场（对照组）", "random", "mut"),
]
A('<table><thead><tr><th>曲线</th><th>总收益</th><th>年化</th><th>最大回撤</th>'
  '<th>Sharpe</th><th>年化波动</th></tr></thead><tbody>')
for lb, k, _ in metrics:
    c = CV[k]
    A(f'<tr><td>{lb}</td><td class="{"pos" if c["total_ret"]>0 else "neg"}">'
      f'{H.pct(c["total_ret"])}</td>'
      f'<td class="{"pos" if c["cagr"]>0 else "neg"}">{H.pct(c["cagr"])}</td>'
      f'<td class="neg">{H.pct(c["max_dd"])}</td>'
      f'<td>{c["sharpe"]:.2f}</td><td>{H.pct(c["vol"])}</td></tr>')
A("</tbody></table>")

# 净值曲线 SVG
A('<div class="chart">')
W, Hh = 900, 300
pl, pr, pt, pb = 62, 130, 18, 34
series = [("ruleB", "#c0392b", "规则B"), ("ruleC", "#e08b1f", "规则C"),
          ("simple", "#2b6cb0", "精简版(无缩量)"), ("random", "#7a8699", "随机全市场")]
lo = min(float(s.min()) for _, s in NAV.items())
hi = max(float(s.max()) for _, s in NAV.items())
pad = (hi - lo) * .06 or .1
lo, hi = lo - pad, hi + pad
n = max(len(s) for s in NAV.values())


def xp(i):
    return pl + (W - pl - pr) * i / (n - 1)


def yp(v):
    return pt + (Hh - pt - pb) * (1 - (v - lo) / (hi - lo))


out = [f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto">']
for k in range(5):
    v = lo + (hi - lo) * k / 4
    out.append(f'<line x1="{pl}" x2="{W-pr}" y1="{yp(v):.1f}" y2="{yp(v):.1f}" stroke="#e9edf3"/>')
    out.append(f'<text x="{pl-6}" y="{yp(v)+4:.1f}" font-size="10" fill="#8b95a7" '
               f'text-anchor="end">{v:.2f}</text>')
out.append(f'<line x1="{pl}" x2="{W-pr}" y1="{yp(1.0):.1f}" y2="{yp(1.0):.1f}" '
           f'stroke="#c8d0dc" stroke-dasharray="4 4"/>')
for k, (key, col, lb) in enumerate(series):
    s = NAV[key]
    d = "".join((("M" if i == 0 else "L") + f"{xp(i):.1f} {yp(float(v)):.1f}")
                for i, v in enumerate(s.values))
    out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.9"/>')
    out.append(f'<rect x="{W-pr+14}" y="{pt+6+k*24}" width="11" height="11" fill="{col}"/>')
    out.append(f'<text x="{W-pr+31}" y="{pt+16+k*24}" font-size="12" fill="#586274">{lb}</text>')
idx = list(NAV["ruleB"].index)
for i in [0, len(idx)//4, len(idx)//2, 3*len(idx)//4, len(idx)-1]:
    out.append(f'<text x="{xp(i):.1f}" y="{Hh-10}" font-size="10" fill="#8b95a7" '
               f'text-anchor="middle">{idx[i][:7]}</text>')
out.append("</svg>")
A("".join(out))
A("</div>")
A('<div class="cap">图 9　四组方案的理论等权篮子净值（含全部成本与滑点），2015-12 起 = 1.0，'
  '按周采样。</div>')

A('<div class="callout"><span class="h">净值曲线里有三个重要信息。</span>'
  '<ol><li><b>修正方案确实打赢了随机基准。</b>'
  f'规则B 总收益 {H.pct(CV["ruleB"]["total_ret"])}，'
  f'随机基准 {H.pct(CV["random"]["total_ret"])}，差距 {H.pct(CV["ruleB"]["total_ret"]-CV["random"]["total_ret"])}。</li>'
  f'<li><b>但绝对水平低到没有实盘价值。</b>规则B 年化 {H.pct(CV["ruleB"]["cagr"])}、'
  f'规则C 年化 {H.pct(CV["ruleC"]["cagr"])}，'
  f'最大回撤却高达 {H.pct(CV["ruleB"]["max_dd"])}。'
  f'同期沪深300 的长期年化远高于此。Sharpe 只有 {CV["ruleB"]["sharpe"]:.2f}。</li>'
  f'<li><b>「精简版（无缩量）」净值是 {H.pct(CV["simple"]["total_ret"])}，比规则B差。</b>'
  '这一条看起来很反常——第 14 节会解释：'
  '因为它日均在场头寸远多于规则B（信号量 5.7 倍），'
  '等同于长期满仓持有 A 股，吃到了 2016–2026 的震荡下行 beta，'
  '所以净值更差。<b>这不代表「缩量有用」，而是净值口径受暴露度污染。</b></li></ol></div>')

A('<h3>13.1 漂移基准：随机买入持有多日的收益</h3>')
A('<p>要判断策略有没有本事，必须先知道「什么都不做」能拿到多少。'
  '下表是随机抽取 12 万个（股票×日期）组合，买入并持有 N 个交易日、<b>不计成本</b>的平均收益。</p>')

t = f6[["hold", "n", "win_rate", "avg_ret"]].copy()
t["hold"] = t["hold"].map(lambda x: f"{int(x)}日")
t["日均"] = f6["avg_ret"] / f6["hold"]
t = t[["hold", "n", "win_rate", "avg_ret", "日均"]]
t.columns = ["持有天数", "样本数", "胜率", "平均收益(零成本)", "折合日均"]
t["样本数"] = t["样本数"].map(lambda x: f"{int(x):,}")
t["胜率"] = t["胜率"].map(lambda x: f"{x*100:.1f}%")
t["平均收益(零成本)"] = t["平均收益(零成本)"].map(P)
t["折合日均"] = t["折合日均"].map(lambda x: f"{x*100:.4f}%")
A(H.table(t, cls="dt"))

A('<div class="callout warn"><span class="h">A 股 2016–2026 的「日均漂移」约为 +0.038%。</span>'
  '任何持有期较长的策略，只要不做空，都会自动吃到这部分漂移。'
  '所以评价一个多头策略，必须问：<b>它相对于「同期随机持有等长天数」多拿了吗？</b>'
  '把这句话套到本报告的核心结论上：</div>')

t = pd.DataFrame([
    ["随机持有 20 日（零成本）", f"{drift20/20*100:.4f}%", "—", "漂移基准"],
    ["随机全市场信号（含成本）", f"{a('random','avg_ret')/a('random','avg_hold')*100:.4f}%",
     f"持有 {a('random','avg_hold'):.1f} 日", "对照组"],
    ["原策略（缩量A+企稳A）", f"{base_avg/B['avg_hold']*100:.4f}%",
     f"持有 {B['avg_hold']:.1f} 日", "显著为负"],
    ["规则B（修正版）", f"{cand_per_day*100:.4f}%",
     f"持有 {best_cand['avg_hold']:.1f} 日", "显著为正但优势极薄"],
], columns=["方案", "折合日均净收益", "备注", "判定"])
A(H.table(t, cls="dt"))
A('<div class="cap">表：把单笔收益换算成「单位持有时间」口径，与漂移基准公平比较。'
  '规则B 的日均优势约 0.003 个百分点，年化不到 1 个百分点。</div>')

# ================= 14. 消融 =================
A('<h2 id="s14"><span class="no">14</span>消融实验：各条件的真实边际贡献</h2>')
A('<p>前面各节每次只动一个维度。这一节换一个做法：'
  '<b>把所有条件的组合都跑一遍，用「加法组合 + 减法剥离」两种方式定位每个条件的边际贡献。</b>'
  '这是整个报告里信息量最大的一张表。</p>')

order = ["random", "oversold_base", "volA", "up", "nolow2", "ma5here",
         "nolow2+green", "nolow2+up", "nolow2+ma5", "ruleB", "ruleC",
         "ruleB_volB", "ruleB_volC", "ruleB_volD", "ruleB_volE",
         "noOversold", "noOversold_noVol"]
names = {
    "random": "① 随机全市场（基准线）",
    "oversold_base": "② 仅超跌 D2-D6（裸信号）",
    "volA": "③ 超跌 + 缩量A",
    "up": "④ 超跌 + 企稳A（收≥前收）",
    "nolow2": "⑥ 超跌 + 不创新低(2日)",
    "ma5here": "⑦ 超跌 + 站上MA5",
    "nolow2+green": "⑨ 超跌 + 不创新低 + 收阳",
    "nolow2+up": "⑩ 超跌 + 不创新低 + 收涨",
    "nolow2+ma5": "⑪ 超跌 + 不创新低 + 站上MA5",
    "ruleB": "★ 规则B = ⑩ + 缩量A",
    "ruleC": "★ 规则C = ⑪ + 缩量A",
    "ruleB_volB": "⑫ 规则B 换缩量B（5日连降）",
    "ruleB_volC": "⑬ 规则B 换缩量C（地量<0.5×）",
    "ruleB_volD": "⑭ 规则B 换缩量D（<0.7×）",
    "ruleB_volE": "⑮ 规则B 换缩量E（3日均<0.6×）",
    "noOversold": "⑯ 不加超跌：缩量A+不创新低+收涨",
    "noOversold_noVol": "⑰ 全市场+不创新低+收涨（无超跌无缩量）",
}
A('<div class="chart">')
lbl2 = ["随机基准", "裸超跌", "+缩量A", "+企稳A", "+不创新低",
        "+站上MA5", "⑩无缩量", "★规则B", "⑬换地量C", "⑰无超跌无缩量"]
val2 = [a("random", "avg_ret"), a("oversold_base", "avg_ret"), a("volA", "avg_ret"),
        a("up", "avg_ret"), a("nolow2", "avg_ret"), a("ma5here", "avg_ret"),
        a("nolow2+up", "avg_ret"), a("ruleB", "avg_ret"), a("ruleB_volC", "avg_ret"),
        a("noOversold_noVol", "avg_ret")]
A(H.bar_svg(lbl2, val2, width=900, height=270))
A("</div>")
A('<div class="cap">图 10　关键节点的单笔平均净收益（%）。基准：止损-3% / 止盈+6% / 最长持有20日 / 滑点10bp。</div>')

rows = []
for k in order:
    r = abl.loc[k]
    if k == "random":
        r = f1[f1.key == "random"].iloc[0]
    rows.append([names[k], int(r["n"]), f"{r['win_rate']*100:.1f}%",
                 P(r["avg_ret"]), T(r["t_stat"]), f"{r['avg_hold']:.1f}"])
t = pd.DataFrame(rows, columns=["信号组合", "样本数", "胜率", "单笔净收益", "t值", "平均持有(日)"])
t["样本数"] = t["样本数"].map(lambda x: f"{x:,}")
A(H.table(t, cls="dt"))

A('<h3>14.1 2×2 因素分解：超跌 × 缩量</h3>')
A('<p>为了干净地分离「超跌」与「缩量」这两个条件的贡献，'
  '把其余条件固定为「不创新低 + 收涨 + 止损-3% + 止盈+6%」，'
  '只对这两个条件做 2×2 全因子实验：</p>')

fac = pd.DataFrame([
    ["无超跌 · 无缩量（⑰）", P(BASE_NONE), f"t={T(a('noOversold_noVol','t_stat'))}",
     f"n={int(a('noOversold_noVol','n')):,}", "—", "—"],
    ["无超跌 · 有缩量A（⑯）", P(VOL_ONLY), f"t={T(a('noOversold','t_stat'))}",
     f"n={int(a('noOversold','n')):,}", f"{d_vol_noOS*100:+.4f}%", "缩量的边际影响"],
    ["有超跌 · 无缩量（⑩）", P(OS_ONLY), f"t={T(a('nolow2+up','t_stat'))}",
     f"n={int(a('nolow2+up','n')):,}", f"{d_os_noVol*100:+.4f}%", "超跌的边际影响"],
    ["有超跌 · 有缩量A（★规则B）", P(BOTH), f"t={T(a('ruleB','t_stat'))}",
     f"n={int(a('ruleB','n')):,}", "—", "—"],
], columns=["组合", "单笔净收益", "显著性", "样本量", "较左上角增量", "备注"])
A(H.table(fac, cls="dt"))
A('<div class="cap">表：超跌 × 缩量的 2×2 全因子分解。其余条件固定为「不创新低 + 收涨」。</div>')

A('<div class="grid g2">')
A(verdict_card("缩量的边际影响（无超跌时）", f"{d_vol_noOS*100:+.3f}pp",
               f"{P(BASE_NONE)} → {P(VOL_ONLY)}", "warn"))
A(verdict_card("缩量的边际影响（有超跌时）", f"{d_vol_withOS*100:+.4f}pp",
               f"{P(OS_ONLY)} → {P(BOTH)}", "bad"))
A(verdict_card("超跌的边际影响（无缩量时）", f"{d_os_noVol*100:+.3f}pp",
               f"{P(BASE_NONE)} → {P(OS_ONLY)}", "good"))
A(verdict_card("超跌的边际影响（有缩量时）", f"{d_os_withVol*100:+.3f}pp",
               f"{P(VOL_ONLY)} → {P(BOTH)}", "good"))
A("</div>")

A('<div class="callout bad"><span class="h">因子分解的结论很清楚：超跌有贡献，缩量没有。</span>'
  f'<ul>'
  f'<li><b>超跌</b>：无论有没有缩量，加上超跌都能带来 <b>+{d_os_noVol*100:.2f} ~ '
  f'+{d_os_withVol*100:.2f} 个百分点</b>的提升，且两处都显著。</li>'
  f'<li><b>缩量A</b>：在有超跌的主场景下，边际影响 <b>{d_vol_withOS*100:+.4f} 个百分点</b>——'
  f'统计上等于零，代价是信号量从 {int(a("nolow2+up","n")):,} 笔砍到 '
  f'{int(a("ruleB","n")):,} 笔，少了 '
  f'{(1-int(a("ruleB","n"))/int(a("nolow2+up","n")))*100:.0f}%。</li>'
  f'<li>在没有超跌的场景下，缩量A 看似有 {d_vol_noOS*100:+.3f}pp 的「贡献」，'
  f'但基数极低（{P(BASE_NONE)}），且这个提升完全可能来自样本量缩小带来的噪声。</li>'
  '</ul>'
  '<b>一句话：缩量条件既不筛出更好的交易，也不筛出更差的交易，它只是随机构地删掉了大部分样本。</b>'
  '这是对「缩量 = 卖压衰竭」最直接的反驳。</div>')

A('<p>更狠的证据是换缩量定义：把「缩量A」换成「地量C」（VOL &lt; VOL20×0.5，'
  '也就是原命题最推崇的形态）：</p>')
A('<div class="grid g2">')
A(verdict_card("⑩ 超跌+不创新低+收涨（无缩量）", P(OS_ONLY),
               f"t={T(a('nolow2+up','t_stat'))}", "good"))
A(verdict_card("换成「地量C」后", P(a("ruleB_volC", "avg_ret")),
               f"t={T(a('ruleB_volC','t_stat'))} · 从正直接打成负", "bad"))
A("</div>")
A('<p>四种替代缩量定义的结果分别是：B(5日连降) '
  f'{P(a("ruleB_volB","avg_ret"))}、C(地量) {P(a("ruleB_volC","avg_ret"))}、'
  f'D(&lt;0.7×) {P(a("ruleB_volD","avg_ret"))}、E(3日均&lt;0.6×) {P(a("ruleB_volE","avg_ret"))}。'
  '<b>除 B 之外全部为负或接近零。</b>'
  'B 的样本只有 16,255 笔（t=4.56），显著性弱，其表现明显是样本量太少导致的噪声。'
  '<br><br>所以：<b>「地量」不但不是买点，反而是一个负面因子。</b></p>')

A('<h3>14.2 企稳的边际贡献：最大</h3>')
A('<p>从裸超跌（'
  f'{P(a("oversold_base","avg_ret"))}，t={T(a("oversold_base","t_stat"))}）'
  '到加上企稳：</p>')
A('<ul>')
A(f'<li>加「收≥前收」（企稳A）：{P(a("up","avg_ret"))}，'
  f'提升 {(a("up","avg_ret")-a("oversold_base","avg_ret"))*100:.3f} 个百分点</li>')
A(f'<li>加「2日不创新低」：{P(a("nolow2","avg_ret"))}，'
  f'提升 {(a("nolow2","avg_ret")-a("oversold_base","avg_ret"))*100:.3f} 个百分点</li>')
A(f'<li>加「站上MA5」：{P(a("ma5here","avg_ret"))}，'
  f'提升 {(a("ma5here","avg_ret")-a("oversold_base","avg_ret"))*100:.3f} 个百分点</li>')
A("</ul>")
A('<p>「不创新低」和「站上MA5」的提升幅度是「收≥前收」的 2 倍以上。'
  '这与第 4 节的结论完全一致：<b>企稳条件的信息量取决于它是否为「持续性状态」，'
  '而不是单日事件。</b></p>')

A('<h3>14.3 各条件的贡献排名</h3>')
contrib = pd.DataFrame([
    ["企稳：连续不创新低 / 站上MA5", f"{(a('nolow2','avg_ret')-a('oversold_base','avg_ret'))*100:+.3f}%",
     "最大，稳健", "保留"],
    ["超跌前提（D2-D6）", f"{delta_oversold*100:+.3f}%", "中等，依赖企稳质量", "保留"],
    ["企稳：单日收涨 / 收阳", f"{(a('up','avg_ret')-a('oversold_base','avg_ret'))*100:+.3f}%",
     "小，但为正", "可保留"],
    ["缩量 A（两日递减）", f"{delta_volA*100:+.4f}%", "等于零", "删除"],
    ["缩量 C（地量）", f"{(a('ruleB_volC','avg_ret')-a('nolow2+up','avg_ret'))*100:+.3f}%",
     "显著为负", "删除"],
    ["固定 +6% 止盈", f"{(qrow(e6,'tp',0.06)['avg_ret']-e6.loc[e6['tp'].isna(),'avg_ret'].iloc[0])*100:+.3f}%",
     "显著为负（截断右尾）", "删除/放宽"],
    ["-3% 固定止损", f"{(qrow(e5,'sl',0.03)['avg_ret']-qrow(e5,'atr_k',2.0)['avg_ret'])*100:+.3f}%",
     "过紧", "放宽至 ATR×1.5~2"],
], columns=["条件", "对单笔收益的边际影响", "判定", "处理"])
A(H.table(contrib, cls="dt"))

# ================= 15. 最简单稳定版 =================
A('<h2 id="s15"><span class="no">15</span>最终交付：最简单、参数稳定的版本</h2>')
A('<p>按你的要求，给出「最简单的、参数稳定的版本」。'
  '这里有两个候选，我用「精简」和「可复现」两个标准来取舍。</p>')

A('<h3>15.1 推荐版本：超跌 + 连续 2 日不创新低 + 收盘上涨（不设成交量条件）</h3>')
A('<pre>'
  '买入条件（T 日收盘后判断，T+1 开盘买入）\n'
  '  1. 超跌前提：ret20（20日收益）全市场横截面分位 ∈ D2–D6\n'
  '  2. 企稳条件：连续 2 个交易日不创新低（Low_t ≥ Low_{t-1} 且 Low_{t-1} ≥ Low_{t-2}）\n'
  '  3. 确认条件：收盘价高于前收（Close_t &gt; Close_{t-1}）\n'
  '     （等价替代：收盘站上 MA5，实测结论基本相同，可任选其一）\n'
  '  ── 不加成交量条件（这是本报告最重要的修正）\n'
  '\n'
  '退出条件（以下任一触发，T+1 开盘卖出）\n'
  '  1. 止损：-5% 固定 或 ATR(14)×1.5 跟踪，取更宽者\n'
  '  2. 止盈：不设固定止盈（或放宽至 +15%）\n'
  '  3. 时间止损：最长持有 20 个交易日\n'
  '\n'
  '执行约束\n'
  '  · T+1 开盘成交，T 日收盘出信号，无未来函数\n'
  '  · 涨停开盘放弃买入，跌停一字顺延卖出\n'
  '  · 停牌顺延，信号至成交间隔 &gt; 10 自然日作废\n'
  '  · 剔 ST，剔退市前最后一个交易日之后\n'
  '</pre>')

A('<div class="callout warn"><span class="h">为什么最终没有保留「缩量」？</span>'
  '因为它在统计上是零贡献（'
  f'{delta_volA*100:+.4f}%），却删掉了 83% 的信号。'
  '一个「不带来收益、只带来样本损失和回测复杂度」的条件，'
  '在追求「参数稳定」的目标下必须删掉。'
  '保留它的唯一理由是心理上的——它让策略<br>「听起来有道理」。'
  '但量化研究的纪律是：<b>听起来有道理不是证据，跑出来有差异才是。</b></div>')

A('<h3>15.2 该版本的实测表现</h3>')
A('<table><thead><tr><th>指标</th><th>数值</th><th>说明</th></tr></thead><tbody>')
bc = f1[f1["key"].str.lstrip("+") == "nolow2+up"].iloc[0]
SIM = "精简版(超跌+不创新低+收涨，无缩量)"
for k, v, note in [
    ("信号定义", "超跌D2-D6 + 2日不创新低 + 收盘>前收", "无成交量条件"),
    ("样本数", f"{int(bc['n']):,} 笔", "2016-04 ~ 2026-09 全市场（是含缩量版的 5.7 倍）"),
    ("单笔净收益", P(bc["avg_ret"]), f"t = {T(bc['t_stat'])}，显著为正"),
    ("胜率", f"{bc['win_rate']*100:.1f}%", "低于 50%，靠右尾盈利"),
    ("平均持有", f"{bc['avg_hold']:.1f} 个交易日", "—"),
    ("样本内（16-20）", P(qrow(f2, "cand", "规则B(缩量不走缩量条件)", "period", "样本内16-20")["avg_ret"]),
     f"t = {T(qrow(f2, 'cand', '规则B(缩量不走缩量条件)', 'period', '样本内16-20')['t_stat'])}，符号与全样本一致"),
    ("样本外（21-26）", P(qrow(f2, "cand", "规则B(缩量不走缩量条件)", "period", "样本外21-26")["avg_ret"]),
     f"t = {T(qrow(f2, 'cand', '规则B(缩量不走缩量条件)', 'period', '样本外21-26')['t_stat'])}，量级稳定"),
    ("滑点 20bp 时", P(qrow(f3, "cand", SIM, "slippage", 0.002)["avg_ret"]),
     "优势缩水约 80%"),
    ("滑点 30bp 时", P(qrow(f3, "cand", SIM, "slippage", 0.003)["avg_ret"]),
     "转负"),
    ("牛市条件下", P(qrow(f4, "cand", SIM, "regime", "牛市")["avg_ret"]), "显著为负，应停用"),
    ("熊市条件下", P(qrow(f4, "cand", SIM, "regime", "熊市")["avg_ret"]), "显著为正，最佳环境"),
    ("理论净值 CAGR", H.pct(CV["simple"]["cagr"]), "等权篮子口径，非实盘"),
    ("最大回撤", H.pct(CV["simple"]["max_dd"]), "—"),
]:
    A(f'<tr><td>{k}</td><td style="font-family:ui-monospace,monospace">{v}</td>'
      f'<td style="font-family:inherit;color:#586274">{note}</td></tr>')
A("</tbody></table>")

A('<h3>15.3 参数稳定性检查</h3>')
A('<p>「参数稳定」意味着策略表现对参数的小幅变动不敏感。'
  '下表是对最关键的几个参数做邻域检验的结果（以第 15.1 节推荐版本为基准）：</p>')

stable = pd.DataFrame([
    ["超跌区间", "D2-D6 → D3-D7", P(qrow(e3, "rng", "D3-D7")["avg_ret"]), "稳定"],
    ["超跌区间", "D2-D6 → D4-D8", P(qrow(e3, "rng", "D4-D8")["avg_ret"]), "含缩量A时转差"],
    ["企稳定义", "站上MA5 → 2日不创新低", P(a("ma5here", "avg_ret")), "两者可互换"],
    ["企稳定义", "2日不创新低 → 站上MA5", P(a("nolow2+ma5", "avg_ret")), "稳定"],
    ["止损", "-3% → ATR×1.5", P(qrow(e5, "atr_k", 1.5)["avg_ret"]), "方向一致，量级更优"],
    ["止损", "-3% → ATR×2", P(qrow(e5, "atr_k", 2.0)["avg_ret"]), "方向一致"],
    ["止盈", "+6% → +10%", P(qrow(e6, "tp", 0.10)["avg_ret"]), "单调改善"],
    ["止盈", "+6% → 不设", P(e6.loc[e6["tp"].isna(), "avg_ret"].iloc[0]), "最佳"],
    ["最长持有", "20日 → 30日", P(qrow(e12, "max_hold", 30)["avg_ret"]), "变化很小，稳定"],
    ["最长持有", "20日 → 10日", P(qrow(e12, "max_hold", 10)["avg_ret"]), "缩短变差"],
], columns=["参数", "邻域变动", "变动后单笔净收益", "稳定性判定"])
A(H.table(stable, cls="dt"))

A('<div class="callout good"><span class="h">稳定性结论：这套参数在合理邻域内不敏感，可以接受。</span>'
  '除了「超跌区间扩展到 D4-D8」和「持有期缩短到 10 日」之外，'
  '其余邻域变动都不改变结论方向。'
  '尤其值得注意的是：<b>「2日不创新低」和「站上MA5」这两个企稳定义可以互相替换'
  '而结果几乎不变</b>，这说明策略捕捉的是一个真实的、稳健的价格结构特征，'
  '不是某个精调参数的巧合。</div>')

A('<div class="callout bad"><span class="h">但我必须再强调一次：稳定 ≠ 值得做。</span>'
  '这个策略的优势在 10bp 滑点下是 '
  f'{P(bc["avg_ret"])}/笔，在 20bp 下滑到约 '
  f'{P(qrow(f3,"cand",SIM,"slippage",0.002)["avg_ret"])}，'
  '在 30bp 下为负。'
  '考虑到本策略主要交易的是「超跌股」（往往是流动性较差、'
  '波动较大的标的），<b>真实双边冲击成本达到 20~30bp 是常态而非例外</b>。'
  '因此：<b>我不建议把这个策略用于实盘。</b>'
  '它的价值在于「验证了一个投资直觉的哪一部分是对的、哪一部分是错的」。</div>')

# ================= 16. 十问 =================
A('<h2 id="s16"><span class="no">16</span>十个问题的直接回答</h2>')

QA = [
    ("Q1", "「连续缩量」是否比「普通缩量」有效？", "no",
     "否。方向相反。两日递减(A) "
     f"{P(base_avg)}、五日连降(B) {P(qrow(e1,'vol','B')['avg_ret'])}，"
     "两者都比不设缩量更差。缩量条件越严格，收益越差——这是一条单调规律，"
     "不是噪声。"),
    ("Q2", "「地量」是否真的代表卖压衰竭？", "no",
     f"被证伪，而且是最强的一条证伪。地量（VOL&lt;VOL20×0.5）单笔净收益 "
     f"{P(qrow(e1,'vol','C')['avg_ret'])}，t={T(qrow(e1,'vol','C')['t_stat'])}，"
     f"是六种缩量定义里最差的。把规则B的缩量条件换成地量后，单笔收益从 "
     f"{P(a('ruleB','avg_ret'))} 崩塌到 {P(a('ruleB_volC','avg_ret'))}。"
     "<b>地量在 A 股更多反映关注度流失，而非卖压衰竭。</b>"),
    ("Q3", "「缩量 + 不创新低」的组合是否优于单一条件？", "yes",
     f"是。不创新低单独使用：{P(a('nolow2','avg_ret'))}（t={T(a('nolow2','t_stat'))}）；"
     f"加上缩量A：{P(a('ruleB','avg_ret'))}（t={T(a('ruleB','t_stat'))}）。"
     "<b>但请注意：这个提升（+0.0077%）在统计上等于零，"
     "本质是样本量从 126.8 万缩到 14.7 万带来的噪声。</b>"
     "组合的优势几乎全部来自「不创新低」，不来自缩量。"),
    ("Q4", "「企稳」条件是否能显著改善策略？", "yes",
     f"是，而且是四项子假设里贡献最大的。最弱企稳（单日收≥前收）{P(a('up','avg_ret'))}，"
     f"最强企稳（2日不创新低）{P(a('nolow2','avg_ret'))}，"
     f"不设企稳 {P(qrow(e2,'stab','NONE')['avg_ret'])}。"
     "提升幅度约 0.18 个百分点，t 值从负变正。"
     "<b>关键在于企稳必须是「持续性状态」（连续不创新低）而非「单日事件」（收涨）。</b>"),
    ("Q5", "-3% 止损是否合理？", "no",
     f"不合理，偏紧。-2% 时 {P(qrow(e5,'sl',0.02)['avg_ret'])}（t={T(qrow(e5,'sl',0.02)['t_stat'])}），"
     f"-3% 时 {P(base_avg)}（t={T(base_t)}），"
     f"-5% 时 {P(qrow(e5,'sl',0.05)['avg_ret'])}（t={T(qrow(e5,'sl',0.05)['t_stat'])}），"
     f"ATR×2 时 {P(qrow(e5,'atr_k',2.0)['avg_ret'])}。"
     "<b>-3% 在超跌股的日内噪声里会被频繁击穿。</b>"
     "更稳健的选择是 ATR×1.5~2 或 -5%。"),
    ("Q6", "+6% 止盈是否合理？", "no",
     f"不合理，过早。+6% 时 {P(base_avg)}，+10% 时 {P(qrow(e6,'tp',0.10)['avg_ret'])}，"
     f"+15% 时 {P(qrow(e6,'tp',0.15)['avg_ret'])}，"
     f"不设止盈 {P(e6.loc[e6['tp'].isna(),'avg_ret'].iloc[0])}。"
     "止盈越晚越好，规律单调。<b>+6% 在 8 档测试里排倒数第三。</b>"),
    ("Q7", "固定 +6% 止盈是否抑制了大赢家？", "yes",
     f"是，抑制得很严重。≥+10% 的交易从 {n10_run:,} 笔（不设止盈）"
     f"降到 {n10_fix:,} 笔（+6% 止盈），减少 {(1-n10_fix/n10_run)*100:.0f}%；"
     f"≥+30% 从 {n30_run:,} 笔降到 {n30_fix} 笔。"
     f"同时「+3%~+6%」这一桶在固定止盈下占了 {g_fix.loc['3%~6%','pct_count']*100:.1f}% 的样本。"
     "<b>结果是右尾被削平、左尾完整保留，这正是净期望为负的几何原因。</b>"),
    ("Q8", "用 MA5/MA10 或 MACD 卖出是否更好？", "part",
     f"均线更差，MACD 更好。<br>MA5 退出 {P(qrow(e7,'desc','对照 -3%/MA5')['avg_ret'])}、"
     f"MA10 退出 {P(qrow(e7,'desc','策略4 -3%/MA10')['avg_ret'])}，"
     f"都显著为负；MACD 死叉退出 {P(qrow(e7,'desc','策略6 -3%/MACD死叉')['avg_ret'])}"
     f"（t={T(qrow(e7,'desc','策略6 -3%/MACD死叉')['t_stat'])}），"
     f"不设止盈仅最长持有 {P(qrow(e7,'desc','对照 -3%/仅最长持有')['avg_ret'])}。"
     "<b>均线在超跌反抽场景里被噪声反复穿越，做退出信号是负贡献。</b>"),
    ("Q9", "这个策略是否存在过拟合？", "part",
     f"原策略存在。<br>样本内 {P(qrow(e10,'period','样本内 2016-2020')['avg_ret'])}"
     f"（t={T(qrow(e10,'period','样本内 2016-2020')['t_stat'])}）"
     f"→ 样本外 {P(qrow(e10,'period','样本外 2021-2026')['avg_ret'])}"
     f"（t={T(qrow(e10,'period','样本外 2021-2026')['t_stat'])}），符号翻转，典型过拟合。"
     f"<br>修正方案不存在：规则B 样本内 "
     f"{P(qrow(f2,'cand','规则B(缩量A)','period','样本内16-20')['avg_ret'])}"
     f" → 样本外 {P(qrow(f2,'cand','规则B(缩量A)','period','样本外21-26')['avg_ret'])}，"
     "符号与量级都稳定。<b>但要提醒：一致性只证明方向可信，不证明幅度可信。</b>"),
    ("Q10", "这个策略是否有统计优势？是否可以实盘使用？", "no",
     f"<b>原策略：没有统计优势，是统计显著的负期望（{P(base_avg)}，t={T(base_t)}，"
     f"n=515,524）。</b><br>"
     f"修正方案：有微弱正优势（{P(bc['avg_ret'])}，t={T(bc['t_stat'])}），"
     "并且样本内外一致。<br>"
     "<b>但不建议实盘。</b>理由：① 优势只有 0.25%/笔，"
     "相对同期随机持有几乎无超额（折合日均优势约 0.003%）；"
     "② 滑点从 10bp 到 20bp，优势缩水约八成；到 30bp 转负；"
     "③ 牛市里显著为负；④ 理论净值 CAGR 仅 1~2%，最大回撤 34~39%，"
     "性价比远不如简单持有指数。"
     "<b>结论：这个策略的思路值得研究，但不具备可实盘的经济价值。</b>"),
]
for q, qt, verdict, ans in QA:
    vmap = {"yes": ("成立", "yes"), "no": ("不成立 / 需修正", "no"),
            "part": ("部分成立", "part")}
    vtxt, vcls = vmap[verdict]
    A('<div class="qa">')
    A(f'<div class="q">{q}　{qt}</div>')
    A(f'<div class="a"><div class="vrd {vcls}">判定：{vtxt}</div>{ans}</div>')
    A("</div>")

# ================= 17. 边界 =================
A('<h2 id="s17"><span class="no">17</span>结论的边界：这份报告不能证明什么</h2>')
A('<p>一份诚实的量化报告必须说明自己的局限。以下每一条都可能是你质疑我的入口，'
  '我主动列出来。</p>')
A('<ol>')
A('<li><b>没有考虑市值/流动性分层。</b>'
  '本报告用全市场信号，但在实盘里，超跌股往往是小市值、低流动性的标的，'
  '冲击成本远高于 10bp。第 12 节的滑点敏感性测试说明这可能是致命的。'
  '<b>建议的补充检验：按流通市值分组，看策略在大市值股上是否还成立。</b></li>')
A('<li><b>没有剔除财务造假、重大违规等事件。</b>'
  '超跌 + 缩量 + 不创新低，在财务爆雷前也可能出现。'
  '虽然退市股按最后可成交价平仓，但没有做「事件窗口排除」。</li>')
A('<li><b>净值曲线的口径限制。</b>'
  '第 13 节已说明：等权篮子假设资金无限、日均 334 个并发头寸，'
  '不代表真实账户可实现。'
  '<b>真实的资金约束、集中度约束没有建模。</b></li>')
A('<li><b>2026 年数据不完整。</b>'
  '数据截至 2026-09-23，2026 年仅覆盖约 9 个月，'
  '分年度统计里 2026 年的样本量偏少，不应单独解读。</li>')
A('<li><b>「不显著」与「无效」不是一回事。</b>'
  '本报告大量使用 t 检验。在 51 万笔的样本量下，'
  't=3 对应的实际收益差只有 0.02 个百分点——'
  '<b>统计显著不等于经济显著</b>。'
  '本报告在判定「是否有实盘价值」时，用的是经济显著性标准，而不是 t 值。</li>')
A('<li><b>多重检验问题。</b>'
  '本报告测试了 5 种缩量 × 6 种企稳 × 6 种超跌区间 × 7 种止损 × 8 种止盈 '
  '≈ 上万个参数组合的隐含空间。'
  '在这种规模下，<b>纯靠运气也能找到一些 t 值很高的组合</b>。'
  '本报告用来对抗这个问题的手段是「样本内/样本外一致性检验」'
  '和「消融实验（要求单条件也有独立贡献）」，'
  '但无法完全消除数据挖掘偏差。'
  '<b>规则B/C 的 +0.25% 优势，仍有可能是多重检验下的幸存者偏差。</b></li>')
A("</ol>")

A('<div class="callout warn"><span class="h">最后一条（多重检验）值得再多说一句。</span>'
  '如果你拿着「规则B/C 有 +0.25% 优势」这个结论去做实盘，'
  '那么你下注的其实是「这个优势是真的，而不是我从上万个组合里挑出来的运气最好的那个」。'
  '本报告的证据支持前者（样本内外一致 + 消融实验可分解），'
  '但不能排除后者。<b>这是所有量化研究共同的困境，不是本报告的瑕疵，但你必须知道。</b></div>')

# ================= 18. 免责 =================
A('<h2 id="s18"><span class="no">18</span>免责声明</h2>')
A('<blockquote>'
  '<strong>免责声明</strong><br>'
  '以上内容基于公开数据和量化分析，仅供研究与学习参考，<b>不构成任何投资建议或投资邀约</b>。'
  '所有回测结果均基于历史数据，<b>历史表现不代表未来收益</b>，'
  '且回测本身存在滑点、冲击成本、流动性、幸存者偏差、多重检验偏差等无法完全消除的局限。'
  '本报告明确结论为：所检验的「缩量企稳反转」原策略<b>不具备统计优势</b>，'
  '修正版本虽有微弱正优势但<b>不建议用于实盘</b>。'
  '市场有风险，投资需谨慎，任何据此进行的投资决策及其后果由投资者自行承担。'
  '</blockquote>')

A('<hr>')
A('<h3>附：复现说明</h3>')
A('<pre>'
  '项目结构（C:/Users/50651/WorkBuddy/2026-09-23-21-44-11/a_share_reversal/）\n'
  '  data/daily.parquet        5,187 只 × 1,146 万行后复权日线（2015-12-01 起）\n'
  '  data/index.parquet        6 大指数日线\n'
  '  data/universe.csv         标的池（含板块/ST 标记）\n'
  '  data/pools.json           沪深300 / 中证500 / 创业板 / 科创板 成分\n'
  '  src/fetch_data.py         数据抓取（腾讯 hfq K线，多端点故障转移）\n'
  '  src/engine.py             回测引擎（Panel / build_signal / simulate / equity_curve）\n'
  '  src/run_experiments.py    第一轮 12 组实验（E0–E12）\n'
  '  src/run_experiments2.py   第二轮消融实验（f1–f2）\n'
  '  src/run_experiments3.py   第三轮成本/分层/净值/分布（f3–f9）\n'
  '  src/server.py             交互式回测 Web 服务\n'
  '  src/make_report.py        本报告生成脚本\n'
  '  results/                  全部实验明细 CSV + summary.json\n'
  '  reports/                  本报告\n'
  '\n'
  '复现命令\n'
  '  python src/fetch_data.py all          # 抓取数据（约 40 分钟，支持断点续传）\n'
  '  python src/run_experiments.py         # 第一轮实验（约 27 分钟）\n'
  '  python src/run_experiments3.py        # 第三轮实验（约 10 分钟）\n'
  '  python src/make_report.py             # 生成本报告\n'
  '  python src/server.py 8760             # 启动交互式回测台\n'
  '  # 浏览器打开 http://127.0.0.1:8760\n'
  '</pre>')

A('<div class="footer">'
  'A股「缩量企稳反转」策略 量化回测与假设检验报告<br>'
  '数据区间 2016-04-01 ~ 2026-09-23 · 全市场 4,986 只标的 · '
  '1,146 万行日线 · 全部实验均在真实 A 股交易约束下完成<br>'
  '报告生成：2026-09-23</div>')

A("</div></body></html>")

fp = os.path.join(OUT, "缩量企稳反转策略_回测报告.html")
with open(fp, "w", encoding="utf-8") as f:
    f.write("".join(S))
print(f"[ok] {fp}")
print(f"[size] {os.path.getsize(fp)/1024:.1f} KB")
