# -*- coding: utf-8 -*-
"""
A股日线数据抓取（多源容错）

数据源优先级：
  清单：新浪 Market_Center（全市场 A 股，含名称，用于 ST/退市识别）
  日线：腾讯 web.ifzq.gtimg.cn fqkline（后复权 hfq，单次上限 640 根，向过去翻页）
       失败时回退 新浪 getKLineData（不复权，仅近 1023 根，作兜底）

产出：
  data/universe.csv
  data/daily.parquet
  data/index.parquet
"""
import json
import os
import sys
import time
import threading
import urllib.request
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
UA_SINA = dict(UA, Referer="https://finance.sina.com.cn/")
UA_TX = dict(UA, Referer="https://gu.qq.com/")

START_DATE = "2015-12-01"      # 预热起点（MA60 / ret20 需要）
END_DATE = dt.date.today().isoformat()

TXPAGE = 640
# 腾讯日K多端点（按顺序故障转移；单端点限流时自动切换）
TX_ENDPOINTS = [
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
    "https://ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
]
_lock = threading.Lock()
_cnt = [0, 0]                  # [done, failed]


def _get(url, headers, tries=6, timeout=25, dec="utf-8"):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode(dec, "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.25 * (i + 1))
    raise last


# ----------------------------------------------------------------------------- 清单
def fetch_universe():
    rows = []
    page = 1
    while True:
        u = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             f"Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1"
             "&node=hs_a&symbol=")
        txt = _get(u, UA_SINA, dec="gbk")
        try:
            d = json.loads(txt)
        except Exception:  # noqa: BLE001
            break
        if not d:
            break
        rows.extend(d)
        if len(d) < 100:
            break
        page += 1
        time.sleep(0.12)

    df = pd.DataFrame(rows)
    df = df[["code", "name", "symbol"]].drop_duplicates("code")
    df["mkt"] = df["symbol"].str[:2]

    def board(sym, code):
        if sym.startswith("bj") or str(code).startswith(("8", "4", "920")):
            return "北交所"
        if str(code).startswith("688"):
            return "科创板"
        if str(code).startswith(("300", "301")):
            return "创业板"
        return "主板"

    def limit(sym, code, name):
        if "ST" in str(name).upper():
            return 0.05
        b = board(sym, code)
        return {"科创板": 0.20, "创业板": 0.20, "北交所": 0.30}.get(b, 0.10)

    df["board"] = [board(s, c) for s, c in zip(df.symbol, df.code)]
    df["limit"] = [limit(s, c, n) for s, c, n in zip(df.symbol, df.code, df.name)]
    nm = df["name"].astype(str)
    df["is_st"] = nm.str.upper().str.contains("ST")
    df["is_delisting"] = nm.str.contains("退")
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(DATA, "universe.csv"), index=False, encoding="utf-8-sig")
    print(f"[universe] {len(df)} 只 | ST {int(df.is_st.sum())} | 含退 {int(df.is_delisting.sum())} "
          f"| 板块 {df.board.value_counts().to_dict()}")
    return df


# ----------------------------------------------------------------------------- 日线
_ep_rr = [0]


def _tx_page(symbol, end):
    """腾讯日K（后复权 hfq）。多端点轮询 + 故障转移，降低单端点限流概率。"""
    last = None
    n = len(TX_ENDPOINTS)
    with _lock:
        start = _ep_rr[0]
        _ep_rr[0] = (_ep_rr[0] + 1) % n
    for k in range(n):
        ep = TX_ENDPOINTS[(start + k) % n]
        u = (f"{ep}?param={symbol},day,{START_DATE},{end},{TXPAGE},hfq")
        try:
            txt = _get(u, UA_TX, tries=2, timeout=25)
            d = json.loads(txt)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        node = (d.get("data") or {}).get(symbol) or {}
        if not isinstance(node, dict):
            continue
        for kk in ("hfqday", "qfqday", "day"):
            v = node.get(kk)
            if isinstance(v, list) and v and isinstance(v[0], list):
                return [r[:6] for r in v]
    if last:
        time.sleep(1.0)
        raise last
    return []


def fetch_one_tx(symbol):
    """向过去翻页，直到覆盖 START_DATE。"""
    end = END_DATE
    seen, out = set(), []
    for _ in range(12):
        try:
            arr = _tx_page(symbol, end)
        except Exception:  # noqa: BLE001
            break
        if not arr:
            break
        fresh = [r for r in arr if r[0] not in seen]
        if not fresh:
            break
        for r in fresh:
            seen.add(r[0])
        out.extend(fresh)
        first = min(r[0] for r in arr)
        if first <= START_DATE:
            break
        end = (dt.date.fromisoformat(first) - dt.timedelta(days=1)).isoformat()
        time.sleep(0.05)
    if not out:
        return None
    out.sort(key=lambda r: r[0])
    df = pd.DataFrame(out, columns=["date", "open", "close", "high", "low", "volume"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.close > 0) & df.open.notna()]
    df["code"] = symbol
    return df


def _dump(frames, path):
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["code", "date"]).sort_values(["code", "date"])
    df = df.reset_index(drop=True)
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = df[c].astype("float32")
    df["date"] = df["date"].astype(str)
    df["code"] = df["code"].astype(str)
    df.to_parquet(path, index=False)


def fetch_all(universe, workers=8, limit=None, start=None, end=None,
              offset=0, min_bars=100, resume=True, tag=""):
    """带断点续传的全市场日线抓取；每 250 只落盘一次。"""
    global START_DATE, END_DATE
    if start:
        START_DATE = start
    if end:
        END_DATE = end
    out = os.path.join(DATA, f"daily{tag}.parquet")
    # 北交所（bj*）在腾讯 fqkline 上不支持历史范围，仅返回当日一根，直接剔除
    u = universe[universe["board"] != "北交所"]
    syms = list(dict.fromkeys(u["symbol"].tolist()))
    if offset:
        syms = syms[offset:]
    if limit:
        syms = syms[:limit]

    frames, done = [], set()
    if resume and os.path.exists(out):
        try:
            old = pd.read_parquet(out)
            done = set(old.code.unique())
            frames.append(old)
            print(f"[resume] 已完成 {len(done)} 只", flush=True)
        except Exception:  # noqa: BLE001
            pass
    todo = [s for s in syms if s not in done]
    print(f"[fetch] 待抓 {len(todo)} 只（跳过 {len(syms)-len(todo)}）", flush=True)
    if not todo:
        return frames[0] if frames else None
    _cnt[0] = 0
    _cnt[1] = 0
    t0 = time.time()

    def work(s):
        try:
            r = fetch_one_tx(s)
        except Exception:  # noqa: BLE001
            r = None
        with _lock:
            _cnt[0] += 1
            if r is None or len(r) < min_bars:
                _cnt[1] += 1
            else:
                frames.append(r)
            if _cnt[0] % 250 == 0 or _cnt[0] == len(todo):
                el = time.time() - t0
                print(f"  ...{_cnt[0]}/{len(todo)} 失败{_cnt[1]} {el:.0f}s "
                      f"{_cnt[0]/max(el,1e-9):.1f}只/s", flush=True)
                try:
                    _dump(frames, out)
                except Exception as e:  # noqa: BLE001
                    print("  dump failed:", e, flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))

    _dump(frames, out)
    df = pd.read_parquet(out)
    print(f"[daily] {len(df):,} 行 / {df.code.nunique()} 只 → {out}")
    return df


# ----------------------------------------------------------------------------- 指数
INDEXES = {
    "hs300": "sh000300",
    "sse": "sh000001",
    "csi500": "sh000905",
    "csi1000": "sh000852",
    "chinext": "sz399006",
    "szcomp": "sz399001",
}

# 预设股票池（新浪指数成分节点）
POOLS = {
    "hs300": ("沪深300", "hs300"),
    "zz500": ("中证500", "zhishu_000905"),
    "cyb":   ("创业板全部", "cyb"),
    "kcb":   ("科创板全部", "kcb"),
}


def fetch_pools():
    """抓取预设股票池成分（用于前端选择回测范围）。"""
    out = {}
    for key, (label, node) in POOLS.items():
        codes, page = [], 1
        while True:
            u = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                 f"Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1"
                 f"&node={node}&symbol=")
            try:
                d = json.loads(_get(u, UA_SINA, dec="gbk"))
            except Exception:  # noqa: BLE001
                break
            if not d:
                break
            for x in d:
                pre = "sh" if str(x["code"]).startswith(("6", "9", "5")) else "sz"
                codes.append(pre + str(x["code"]))
            if len(d) < 100:
                break
            page += 1
            time.sleep(0.1)
        out[key] = {"label": label, "codes": codes}
        print(f"  [pool] {label}: {len(codes)} 只")
    p = os.path.join(DATA, "pools.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[pools] → {p}")
    return out


def fetch_index():
    frames = []
    for name, sym in INDEXES.items():
        end = END_DATE
        seen, out = set(), []
        for _ in range(12):
            arr = None
            for ep in TX_ENDPOINTS:
                u = (f"{ep}?param={sym},day,{START_DATE},{end},{TXPAGE},qfq")
                try:
                    d = json.loads(_get(u, UA_TX, tries=2))
                except Exception:  # noqa: BLE001
                    continue
                node = (d.get("data") or {}).get(sym) or {}
                if not isinstance(node, dict):
                    continue
                for k in ("day", "qfqday", "hfqday"):
                    v = node.get(k)
                    if isinstance(v, list) and v and isinstance(v[0], list):
                        arr = [r[:6] for r in v]
                        break
                if arr:
                    break
            if not arr:
                break
            fresh = [r for r in arr if r[0] not in seen]
            if not fresh:
                break
            for r in fresh:
                seen.add(r[0])
            out.extend(fresh)
            first = min(r[0] for r in arr)
            if first <= START_DATE:
                break
            end = (dt.date.fromisoformat(first) - dt.timedelta(days=1)).isoformat()
            time.sleep(0.05)
        if not out:
            print(f"  [index] {name} 无数据")
            continue
        out.sort(key=lambda r: r[0])
        df = pd.DataFrame(out, columns=["date", "open", "close", "high", "low", "volume"])
        df["idx"] = name
        df["date"] = df["date"].astype(str)
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df)
        print(f"  [index] {name}: {len(df)} 根 {df.date.iloc[0]} ~ {df.date.iloc[-1]}")
        time.sleep(0.1)
    df = pd.concat(frames, ignore_index=True)
    out = os.path.join(DATA, "index.parquet")
    df.to_parquet(out, index=False)
    print(f"[index] {len(df):,} 行 → {out}")
    return df


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode in ("all", "universe"):
        uni = fetch_universe()
    else:
        uni = pd.read_csv(os.path.join(DATA, "universe.csv"), dtype={"code": str})
    if mode in ("all", "daily"):
        fetch_all(uni, limit=lim)
    if mode in ("all", "index"):
        fetch_index()
    if mode in ("all", "pools"):
        fetch_pools()
    print("done")
