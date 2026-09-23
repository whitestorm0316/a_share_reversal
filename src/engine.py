# -*- coding: utf-8 -*-
"""
A股短线“缩量企稳反转”策略回测引擎

设计原则
--------
1. 无未来函数：信号在 T 日收盘产生（仅用 <=T 的数据），T+1 开盘执行。
2. 真实交易约束：
   - T+1（当日买入不可当日卖出，最早 T+2 卖出）
   - 手续费（双边）、印花税（卖出，2023-08-28 前 0.1%，后 0.05%）、过户费、滑点
   - 开盘一字涨停不可买入；跌停不可卖出
   - 停牌（无K线）自动跳过
   - ST / 退市股剔除
3. 横截面分层：ret20 按当日全市场排序取十分位（D1 最低 = 最超跌）。
"""
import os
import datetime as dt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# ---------------------------------------------------------------- 交易成本
COMMISSION = 0.00025        # 佣金 双边
TRANSFER_FEE = 0.00001      # 过户费 双边
STAMP_AFTER = 0.0005        # 印花税（卖出）2023-08-28 起
STAMP_BEFORE = 0.0010       # 印花税（卖出）2023-08-28 前
STAMP_SWITCH = "2023-08-28"
BUY_COST = COMMISSION + TRANSFER_FEE
SELL_COST = COMMISSION + TRANSFER_FEE


# ---------------------------------------------------------------- 数据面板
class Panel:
    """全市场日线宽表面板 + 指标。"""

    def __init__(self, daily_path=None, index_path=None, min_history=60,
                 exclude_st=True, date_from="2015-12-01"):
        daily_path = daily_path or os.environ.get("REV_DAILY") \
            or os.path.join(DATA, "daily.parquet")
        index_path = index_path or os.environ.get("REV_INDEX") \
            or os.path.join(DATA, "index.parquet")
        d = pd.read_parquet(daily_path)
        if date_from:
            d = d[d.date >= date_from]
        uni = pd.read_csv(os.path.join(DATA, "universe.csv"), dtype={"code": str})
        keep = uni[["symbol", "name", "board", "limit", "is_st", "is_delisting"]]
        d = d.merge(keep, left_on="code", right_on="symbol", how="left")
        if exclude_st:
            bad = set(uni.loc[uni.is_st | uni.is_delisting, "symbol"])
            d = d[~d.code.isin(bad)]

        d["date"] = d["date"].astype(str)
        self.dates = np.array(sorted(d.date.unique()))
        self.didx = {x: i for i, x in enumerate(self.dates)}
        codes = sorted(d.code.unique())
        self.codes = np.array(codes)
        self.cidx = {c: j for j, c in enumerate(codes)}
        T, N = len(self.dates), len(codes)

        def wide(col):
            m = np.full((T, N), np.nan, dtype=np.float32)
            i = d.date.map(self.didx).to_numpy()
            j = d.code.map(self.cidx).to_numpy()
            m[i, j] = d[col].to_numpy(dtype=np.float32)
            return m

        self.O = wide("open")
        self.H = wide("high")
        self.L = wide("low")
        self.C = wide("close")
        self.V = wide("volume")
        # 涨跌停幅度矩阵
        lim = uni.set_index("symbol")["limit"].to_dict()
        self.LIM = np.array([lim.get(c, 0.10) for c in codes], dtype=np.float32)[None, :]

        self._build_indicators(min_history)
        self._load_index(index_path)

    # ------------------------------------------------------------ 指标
    def _build_indicators(self, min_history):
        C, V, H, L = self.C, self.V, self.H, self.L
        Cdf = pd.DataFrame(C)
        Vdf = pd.DataFrame(V)
        Hdf = pd.DataFrame(H)
        Ldf = pd.DataFrame(L)

        self.MA5 = Cdf.rolling(5, min_periods=5).mean().to_numpy(dtype=np.float32)
        self.MA10 = Cdf.rolling(10, min_periods=10).mean().to_numpy(dtype=np.float32)
        self.MA20 = Cdf.rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        self.MA60 = Cdf.rolling(60, min_periods=60).mean().to_numpy(dtype=np.float32)
        self.VOL20 = Vdf.rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            self.RET20 = (C / np.roll(C, 20, axis=0) - 1).astype(np.float32)
        self.RET20[:20] = np.nan
        self.RET20[C <= 0] = np.nan

        # ATR14（真实波幅均值）
        prevC = np.roll(C, 1, axis=0)
        prevC[0] = np.nan
        tr = np.maximum.reduce([H - L, np.abs(H - prevC), np.abs(L - prevC)])
        tr = np.nan_to_num(tr, nan=0.0)
        valid = ~np.isnan(H) & ~np.isnan(L)
        self.ATR14 = np.asarray(
            pd.DataFrame(tr).rolling(14, min_periods=14).mean()
            .to_numpy(dtype=np.float32)).copy()
        self.ATR14[~valid] = np.nan

        # MACD
        e12 = Cdf.ewm(span=12, adjust=False).mean()
        e26 = Cdf.ewm(span=26, adjust=False).mean()
        dif = e12 - e26
        dea = dif.ewm(span=9, adjust=False).mean()
        self.DIF = dif.to_numpy(dtype=np.float32)
        self.DEA = dea.to_numpy(dtype=np.float32)

        # 上市/有效历史计数（用于上市不足 min_history 日剔除）
        ok = (~np.isnan(C)).astype(np.int32)
        self.BARS = np.cumsum(ok, axis=0)
        self.MINH = min_history

        # 有效性：有K线、价格>0、量>0
        self.ALIVE = (~np.isnan(C)) & (C > 0) & (~np.isnan(V))
        self.WARM = (self.BARS >= min_history) & self.ALIVE

        # ret20 横截面十分位（D1 = ret20 最小 = 最超跌）
        R = pd.DataFrame(np.where(self.WARM, self.RET20, np.nan))
        pct = R.rank(axis=1, pct=True, na_option="keep")
        dec = np.ceil(pct.to_numpy() * 10.0)
        dec = np.where(np.isnan(self.RET20) | ~self.WARM, np.nan, dec)
        self.DEC = dec.astype(np.float32)               # 1..10

    # ------------------------------------------------------------ 指数
    def _load_index(self, index_path):
        p = index_path or os.path.join(DATA, "index.parquet")
        self.idx_close, self.idx_ma60, self.idx_state = {}, {}, {}
        self.market_on = self._mk_market()
        if not os.path.exists(p):
            return
        ix = pd.read_parquet(p)
        for name, g in ix.groupby("idx"):
            g = g.sort_values("date")
            s = pd.Series(g.close.to_numpy(), index=g.date.to_numpy())
            ma60 = s.rolling(60, min_periods=60).mean()
            self.idx_close[name] = s
            self.idx_ma60[name] = ma60
        # 市场状态（以沪深300 为准）
        if "hs300" in self.idx_close:
            s = self.idx_close["hs300"]
            ma60 = self.idx_ma60["hs300"]
            ret120 = s / s.shift(120) - 1
            ma200 = s.rolling(200, min_periods=200).mean()
            st = pd.Series("震荡", index=s.index)
            st[(s > ma200) & (ret120 > 0.10)] = "牛市"
            st[(s < ma200) & (ret120 < -0.10)] = "熊市"
            st[s.isna() | ma200.isna()] = "未知"
            self.idx_state["hs300"] = st
        self.market_on = self._mk_market()

    def _mk_market(self, idx="hs300"):
        """返回逐日的 {'above_ma60':bool,'state':str} 数组，与 self.dates 对齐。"""
        above = np.zeros(len(self.dates), dtype=bool)
        state = np.array(["未知"] * len(self.dates), dtype=object)
        if idx not in self.idx_close:
            return {"above_ma60": above, "state": state}
        c = self.idx_close[idx]
        m = self.idx_ma60[idx]
        s = self.idx_state.get(idx)
        for i, dt in enumerate(self.dates):
            if dt in c.index:
                v = c.loc[dt]
                mv = m.loc[dt]
                above[i] = bool(not np.isnan(mv) and v > mv)
                if s is not None:
                    state[i] = s.loc[dt]
        return {"above_ma60": above, "state": state}


# ---------------------------------------------------------------- 条件定义
def vol_cond(P, kind):
    V = P.V
    if kind == "A":      # VOL_t < VOL_{t-1} < VOL_{t-2}
        return (V < np.roll(V, 1, axis=0)) & (np.roll(V, 1, axis=0) < np.roll(V, 2, axis=0))
    if kind == "B":      # 过去5日成交量连续下降（4 次比较）
        a = np.ones_like(V, dtype=bool)
        for k in range(1, 5):
            a &= np.roll(V, k - 1, axis=0) < np.roll(V, k, axis=0)
        return a
    if kind == "C":      # VOL_t < VOL20 * 0.5
        return V < P.VOL20 * 0.5
    if kind == "D":      # VOL_t < VOL20 * 0.7
        return V < P.VOL20 * 0.7
    if kind == "E":      # 近3日均量 < VOL20 * 0.6
        m3 = pd.DataFrame(V).rolling(3, min_periods=3).mean().to_numpy(dtype=np.float32)
        return m3 < P.VOL20 * 0.6
    if kind == "NONE":
        return np.ones_like(V, dtype=bool)
    raise ValueError(kind)


def stab_cond(P, kind):
    C, L, O = P.C, P.L, P.O
    if kind == "A":      # Close_t >= Close_{t-1}
        return C >= np.roll(C, 1, axis=0)
    if kind == "B":      # 连续2日不创新低（低点不低于前一日低点）
        return (L >= np.roll(L, 1, axis=0)) & (np.roll(L, 1, axis=0) >= np.roll(L, 2, axis=0))
    if kind == "C":      # 过去5日最低点出现在前3日，之后2日未跌破
        M = pd.DataFrame(L).shift(2).rolling(3, min_periods=3).min().to_numpy(dtype=np.float32)
        return (np.roll(L, 1, axis=0) > M) & (L > M)
    if kind == "D":      # Low_t >= Low_{t-1} 且 Close_t > Open_t
        return (L >= np.roll(L, 1, axis=0)) & (C > O)
    if kind == "E":      # 站上 MA5
        return C > P.MA5
    if kind == "NONE":
        return np.ones_like(C, dtype=bool)
    raise ValueError(kind)


DECILE_RANGES = {
    "D1-D3": (1, 3), "D2-D6": (2, 6), "D3-D7": (3, 7), "D4-D8": (4, 8),
    "D1-D6": (1, 6), "D1-D10": (1, 10), "D5-D8": (5, 8),
}


def oversold_cond(P, rng):
    lo, hi = DECILE_RANGES[rng]
    return (P.DEC >= lo) & (P.DEC <= hi)


# ---------------------------------------------------------------- 退出方式
EXIT_FIXED_TP = "fixed_tp"        # 固定止盈（全仓）
EXIT_TP_HALF_MA5 = "tp_half_ma5"  # +X% 卖 50%，余下跌破 MA5 卖
EXIT_MA10 = "ma10"                # 跌破 MA10 卖出
EXIT_MACD = "macd"                # MACD 死叉卖出
EXIT_MA5 = "ma5"                  # 跌破 MA5 卖出
EXIT_HOLD = "hold"                # 仅止损 + 最长持有


def build_signal(P, vol="A", stab="A", rng="D2-D6", buy_rule="A",
                 start=None, end=None, regime_filter=None):
    """构建 T 日收盘信号矩阵（布尔）。"""
    dates = P.dates
    T = len(dates)
    t0 = 0 if start is None else np.searchsorted(dates, start)
    t1 = T if end is None else (np.searchsorted(dates, end) + 1)
    t0 = max(t0, 70)

    if rng == "ALL":
        overs = np.ones_like(P.C, dtype=bool)
    else:
        overs = oversold_cond(P, rng)

    if buy_rule == "A":
        sig = overs & vol_cond(P, vol) & stab_cond(P, stab)
    elif buy_rule == "B":
        sig = (overs & vol_cond(P, vol)
               & (P.L >= np.roll(P.L, 1, axis=0)) & (P.L >= np.roll(P.L, 2, axis=0))
               & (P.C > np.roll(P.C, 1, axis=0)))
    elif buy_rule == "C":
        sig = (overs & vol_cond(P, vol)
               & (P.L >= np.roll(P.L, 1, axis=0)) & (P.L >= np.roll(P.L, 2, axis=0))
               & (P.C > P.MA5))
    else:
        raise ValueError(buy_rule)

    sig = sig & P.WARM
    sig[:t0] = False
    sig[t1:] = False

    if regime_filter == "above_ma60":
        sig &= P.market_on["above_ma60"][:, None]
    elif regime_filter == "below_ma60":
        sig &= ~P.market_on["above_ma60"][:, None]
    elif regime_filter in ("牛市", "熊市", "震荡"):
        sig &= (P.market_on["state"] == regime_filter)[:, None]
    return sig


def run_backtest(P, vol="A", stab="A", rng="D2-D6", buy_rule="A",
                 sl=0.03, sl_mode="fixed", atr_k=None,
                 tp=None, exit_mode=EXIT_FIXED_TP, tp_half=0.06,
                 max_hold=20, slippage=0.001, start=None, end=None,
                 regime_filter=None):
    """信号 + 交易模拟。"""
    sig = build_signal(P, vol=vol, stab=stab, rng=rng, buy_rule=buy_rule,
                       start=start, end=end, regime_filter=regime_filter)
    return simulate(P, sig, sl=sl, sl_mode=sl_mode, atr_k=atr_k, tp=tp,
                    exit_mode=exit_mode, tp_half=tp_half, max_hold=max_hold,
                    slippage=slippage, start=start, end=end)


def simulate(P, sig, sl=0.03, sl_mode="fixed", atr_k=None,
             tp=None, exit_mode=EXIT_FIXED_TP, tp_half=0.06,
             max_hold=20, slippage=0.001, start=None, end=None, max_gap_days=10):
    """
    交易模拟：T 日收盘信号 → T+1 开盘买入；含 T+1、涨跌停、停牌、成本、滑点约束。
    max_gap_days: 信号日到实际成交日之间允许的最大自然日间隔（超过则放弃该信号，
                  防止长期停牌股用陈旧信号入场）。
    """
    dates, codes = P.dates, P.codes
    T, N = len(dates), len(codes)
    t1 = T if end is None else (np.searchsorted(dates, end) + 1)
    t0 = 0 if start is None else np.searchsorted(dates, start)
    t0 = max(t0, 70)

    ci, cj = np.nonzero(sig)
    if len(ci) == 0:
        return _empty_trades()

    O, H, L, C = P.O, P.H, P.L, P.C
    MA5, MA10, ATR, LIM = P.MA5, P.MA10, P.ATR14, P.LIM
    ALIVE = P.ALIVE
    trades = []
    tmax = min(T, t1 + 1)
    dord = None
    if max_gap_days:
        dord = np.array([dt.date.fromisoformat(x).toordinal() for x in dates])

    for t, j in zip(ci, cj):
        # ---- 入场日：T 之后第一个有K线的交易日
        e = t + 1
        while e < tmax and not ALIVE[e, j]:
            e += 1
        if e >= tmax:
            continue
        # 信号日到成交日间隔过长（长期停牌）→ 作废
        if dord is not None and (dord[e] - dord[t]) > max_gap_days:
            continue
        p0 = O[e, j]
        if not np.isfinite(p0) or p0 <= 0:
            continue
        prev_c = C[e - 1, j] if e - 1 >= 0 else np.nan
        if np.isfinite(prev_c) and prev_c > 0:
            chg = p0 / prev_c - 1
            if chg >= LIM[0, j] - 0.004:
                continue                      # 开盘一字涨停买不到
            if chg <= -LIM[0, j] + 0.004:
                continue                      # 开盘跌停不接

        # ---- 止损价
        if sl_mode == "atr" and atr_k is not None and np.isfinite(ATR[e, j]) and ATR[e, j] > 0:
            stop = max(p0 - atr_k * ATR[e, j], p0 * (1 - 0.20))
        else:
            stop = p0 * (1 - sl)
        tgt = p0 * (1 + tp) if tp else None

        leg1_px, leg1_day, half_done = None, None, False
        exit_px = exit_day = reason = None
        d_end = min(T, e + max_hold + 1)

        for dd in range(e + 1, d_end):
            if not ALIVE[dd, j]:
                continue
            o, h, l, c = O[dd, j], H[dd, j], L[dd, j], C[dd, j]
            pc = C[dd - 1, j]
            # 一字跌停不可卖出
            if (np.isfinite(pc) and pc > 0
                    and abs(o / pc - 1 + LIM[0, j]) < 0.004 and abs(h - l) < 1e-9):
                continue
            # 半仓止盈腿
            if exit_mode == EXIT_TP_HALF_MA5 and not half_done:
                hp = p0 * (1 + tp_half)
                if o >= hp:
                    leg1_px, leg1_day, half_done = o, dd, True
                elif h >= hp:
                    leg1_px, leg1_day, half_done = hp, dd, True
            # 止损优先（同日同时触及时按止损，保守）
            if o <= stop:
                exit_px, exit_day, reason = o, dd, "止损跳空"
                break
            if l <= stop:
                exit_px, exit_day, reason = stop, dd, "止损"
                break
            # 固定止盈
            if exit_mode == EXIT_FIXED_TP and tgt is not None:
                if o >= tgt:
                    exit_px, exit_day, reason = o, dd, "止盈跳空"
                    break
                if h >= tgt:
                    exit_px, exit_day, reason = tgt, dd, "止盈"
                    break
            # 收盘类退出（尾盘执行）
            if exit_mode == EXIT_TP_HALF_MA5 and half_done:
                if np.isfinite(MA5[dd, j]) and c < MA5[dd, j]:
                    exit_px, exit_day, reason = c, dd, "跌破MA5"
                    break
            elif exit_mode == EXIT_MA5:
                if np.isfinite(MA5[dd, j]) and c < MA5[dd, j]:
                    exit_px, exit_day, reason = c, dd, "跌破MA5"
                    break
            elif exit_mode == EXIT_MA10:
                if np.isfinite(MA10[dd, j]) and c < MA10[dd, j]:
                    exit_px, exit_day, reason = c, dd, "跌破MA10"
                    break
            elif exit_mode == EXIT_MACD:
                if (np.isfinite(P.DIF[dd, j]) and np.isfinite(P.DEA[dd, j])
                        and np.isfinite(P.DIF[dd - 1, j]) and np.isfinite(P.DEA[dd - 1, j])
                        and P.DIF[dd, j] < P.DEA[dd, j]
                        and P.DIF[dd - 1, j] >= P.DEA[dd - 1, j]):
                    exit_px, exit_day, reason = c, dd, "MACD死叉"
                    break
        else:
            dd = min(T - 1, e + max_hold)
            if dd <= e:               # 样本末端，无足够交易日完成退出
                continue
            while dd > e + 1 and not ALIVE[dd, j]:
                dd -= 1
            exit_px, exit_day, reason = C[dd, j], dd, "到期"

        if exit_px is None or not np.isfinite(exit_px) or exit_px <= 0:
            continue

        if leg1_px is not None and np.isfinite(leg1_px):
            r_half = _net_ret(p0, leg1_px, dates[leg1_day], slippage)
            r_rest = _net_ret(p0, exit_px, dates[exit_day], slippage)
            net = 0.5 * r_half + 0.5 * r_rest
            gross = 0.5 * (leg1_px / p0 - 1) + 0.5 * (exit_px / p0 - 1)
        else:
            net = _net_ret(p0, exit_px, dates[exit_day], slippage)
            gross = exit_px / p0 - 1

        trades.append((
            codes[j], dates[t], dates[e], p0, dates[exit_day], exit_px,
            gross, net, (exit_day - e), reason,
            float(P.DEC[t, j]), float(P.RET20[t, j]),
            P.market_on["state"][t], bool(P.market_on["above_ma60"][t]),
        ))

    if not trades:
        return _empty_trades()
    return pd.DataFrame(trades, columns=[
        "code", "signal_date", "entry_date", "entry_px", "exit_date", "exit_px",
        "gross_ret", "net_ret", "hold_days", "exit_reason",
        "decile", "ret20", "mkt_state", "above_ma60"])


def sample_signal(P, mask, n, seed=42, start=None, end=None):
    """从 mask 中随机抽样 n 个 (t, j)，构造稀疏信号矩阵（用于构造基准/随机对照）。"""
    dates = P.dates
    T = len(dates)
    t0 = max(0 if start is None else np.searchsorted(dates, start), 70)
    t1 = T if end is None else (np.searchsorted(dates, end) + 1)
    m = mask.copy()
    m[:t0] = False
    m[t1:] = False
    ci, cj = np.nonzero(m)
    if len(ci) == 0:
        return np.zeros_like(m)
    rng = np.random.default_rng(seed)
    if n < len(ci):
        idx = rng.choice(len(ci), size=n, replace=False)
        ci, cj = ci[idx], cj[idx]
    out = np.zeros_like(m)
    out[ci, cj] = True
    return out


def _net_ret(p0, px, exit_date, slippage):
    buy = p0 * (1 + slippage) * (1 + BUY_COST)
    sell = px * (1 - slippage)
    stamp = STAMP_BEFORE if exit_date < STAMP_SWITCH else STAMP_AFTER
    sell = sell * (1 - SELL_COST - stamp)
    return sell / buy - 1


def _empty_trades():
    return pd.DataFrame(columns=[
        "code", "signal_date", "entry_date", "entry_px", "exit_date", "exit_px",
        "gross_ret", "net_ret", "hold_days", "exit_reason",
        "decile", "ret20", "mkt_state", "above_ma60"])


# ---------------------------------------------------------------- 指标
def metrics(tr, label="", rf=0.0):
    if tr is None or len(tr) == 0:
        return {"label": label, "n": 0}
    r = tr.net_ret.to_numpy(dtype=float)
    n = len(r)
    wins = r[r > 0]
    loss = r[r <= 0]
    pf = (wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else np.inf
    tstat = r.mean() / (r.std(ddof=1) / np.sqrt(n)) if n > 1 and r.std(ddof=1) > 0 else np.nan
    m = {
        "label": label, "n": n,
        "win_rate": len(wins) / n,
        "avg_ret": r.mean(),
        "median_ret": float(np.median(r)),
        "avg_gross": tr.gross_ret.mean(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": loss.mean() if len(loss) else 0.0,
        "profit_factor": pf,
        "t_stat": tstat,
        "std": r.std(ddof=1) if n > 1 else np.nan,
        "best": r.max(), "worst": r.min(),
        "avg_hold": tr.hold_days.mean(),
        "sum_ret": r.sum(),               # 等权单笔加总（未复利）
        "avg_ret_gross": tr.gross_ret.mean(),
    }
    return m


def bucket_report(tr, edges=(-1, -0.05, -0.03, 0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 10)):
    r = tr.net_ret.to_numpy(dtype=float)
    total = r.sum()
    rows = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        m = (r > a) & (r <= b)
        cnt = int(m.sum())
        s = float(r[m].sum())
        rows.append({
            "bucket": f"{a*100:.0f}%~{b*100:.0f}%" if b < 10 else f">{a*100:.0f}%",
            "count": cnt, "pct_count": cnt / len(r) if len(r) else 0,
            "sum_ret": s, "pct_sum": (s / total) if total != 0 else np.nan,
            "avg": float(r[m].mean()) if cnt else np.nan,
        })
    return pd.DataFrame(rows)


def equity_curve(tr, P, start=None, end=None, slippage=0.001):
    """等权篮子净值：逐日持有所有未平仓头寸、等权、**计入成本与滑点**。

    注意：该曲线假设资金无限、每日等权再平衡，只用于观察策略方向性，
    不代表可实盘复制的收益；判断策略优劣应以单笔交易统计为准。
    """
    if tr is None or len(tr) == 0:
        return pd.Series(dtype=float)
    dates, codes = P.dates, P.codes
    T, N = len(dates), len(codes)
    didx, cidx = P.didx, P.cidx
    C, O = P.C, P.O
    prevC = np.roll(C, 1, axis=0)
    prevC[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        Rmat = np.nan_to_num(C / prevC - 1, nan=0.0).astype(np.float64)
    hold = np.zeros((T, N), dtype=bool)

    for code, ed, xd, ep, xp in zip(tr.code, tr.entry_date, tr.exit_date,
                                    tr.entry_px, tr.exit_px):
        j = cidx[code]
        a, b = didx[ed], didx[xd]
        if a >= b:
            continue
        hold[a:b + 1, j] = True
        o = O[a, j]
        if np.isfinite(o) and o > 0:
            # 入场当日：从开盘（含滑点+买入费用）到收盘
            Rmat[a, j] = C[a, j] / (o * (1 + slippage)) - 1 - BUY_COST
        pc = C[b - 1, j]
        if np.isfinite(pc) and pc > 0 and np.isfinite(xp):
            stamp = STAMP_BEFORE if dates[b] < STAMP_SWITCH else STAMP_AFTER
            Rmat[b, j] = xp * (1 - slippage) * (1 - SELL_COST - stamp) / pc - 1

    num = (Rmat * hold).sum(axis=1)
    den = hold.sum(axis=1)
    R = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    nav = np.cumprod(1 + R)
    s = pd.Series(nav, index=dates)
    if start:
        s = s[s.index >= start]
    if end:
        s = s[s.index <= end]
    return s


def curve_stats(nav):
    if nav is None or len(nav) < 2:
        return {}
    nav = nav / nav.iloc[0]
    ret = nav.pct_change().dropna()
    yrs = len(nav) / 244.0
    cagr = nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (nav / nav.cummax() - 1)
    sharpe = (ret.mean() / ret.std() * np.sqrt(244)) if ret.std() > 0 else np.nan
    return {"total_ret": nav.iloc[-1] - 1, "cagr": cagr,
            "max_dd": dd.min(), "sharpe": sharpe,
            "vol": ret.std() * np.sqrt(244)}
