# -*- coding: utf-8 -*-
"""
回测 Web 服务（仅用 Python 标准库 + numpy/pandas）

启动：
    python src/server.py [port]
然后浏览器打开 http://127.0.0.1:8760

接口
----
GET  /                     前端页面
GET  /api/meta             股票池、板块、数据区间、可选参数
GET  /api/universe?q=&limit=  股票搜索
POST /api/backtest         运行回测
"""
import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

BASE = E.BASE
WEB = os.path.join(BASE, "web")
DATA = os.path.join(BASE, "data")

_P = None
_LOCK = threading.Lock()
_SIG_CACHE = {}
_UNI = None
_POOLS = {}


def get_panel():
    global _P
    with _LOCK:
        if _P is None:
            t = time.time()
            _P = E.Panel()
            print(f"[panel] {len(_P.dates)} 交易日 × {len(_P.codes)} 只 "
                  f"({time.time()-t:.1f}s)", flush=True)
    return _P


def get_universe():
    global _UNI
    if _UNI is None:
        u = pd.read_csv(os.path.join(DATA, "universe.csv"), dtype={"code": str})
        u = u[u.board != "北交所"]
        _UNI = u[["symbol", "code", "name", "board", "is_st"]].copy()
        _UNI["is_st"] = _UNI["is_st"].astype(bool)
    return _UNI


def get_pools():
    global _POOLS
    if not _POOLS:
        p = os.path.join(DATA, "pools.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                _POOLS = json.load(f)
    return _POOLS


def get_signals(P, params):
    key = (params["vol"], params["stab"], params["rng"], params["buy_rule"],
           params.get("start"), params.get("end"), params.get("regime_filter"))
    if key not in _SIG_CACHE:
        if len(_SIG_CACHE) > 40:
            _SIG_CACHE.clear()
        _SIG_CACHE[key] = E.build_signal(
            P, vol=params["vol"], stab=params["stab"], rng=params["rng"],
            buy_rule=params["buy_rule"], start=params.get("start"),
            end=params.get("end"), regime_filter=params.get("regime_filter"))
    return _SIG_CACHE[key]


def _yearly(tr):
    if len(tr) == 0:
        return []
    d = tr.copy()
    d["year"] = d.signal_date.str[:4]
    g = d.groupby("year").net_ret.agg(["count", "mean", "median",
                                       lambda x: float((x > 0).mean())])
    g.columns = ["n", "avg", "median", "win"]
    out = []
    for y, r in g.iterrows():
        out.append({"year": y, "n": int(r["n"]), "avg": float(r["avg"]),
                    "median": float(r["median"]), "win": float(r["win"])})
    return out


def run(payload):
    P = get_panel()
    p = dict(payload.get("params") or {})
    params = dict(
        vol=p.get("vol", "A"), stab=p.get("stab", "A"),
        rng=p.get("rng", "D2-D6"), buy_rule=p.get("buy_rule", "A"),
        sl=abs(float(p.get("sl", 0.03))),
        sl_mode=p.get("sl_mode", "fixed"),
        atr_k=(float(p["atr_k"]) if p.get("atr_k") else None),
        tp=(abs(float(p["tp"])) if p.get("tp") else None),
        exit_mode=p.get("exit_mode", E.EXIT_FIXED_TP),
        tp_half=abs(float(p.get("tp_half", 0.06))),
        max_hold=int(p.get("max_hold", 20)),
        slippage=float(p.get("slippage", 0.001)),
        start=p.get("start"), end=p.get("end"),
        regime_filter=p.get("regime_filter"),
    )
    sig = get_signals(P, params)
    sig = sig.copy()

    # 股票范围（横截面分位仍基于全市场，保证口径正确）
    pool = payload.get("pool") or "all"
    codes = None
    if pool == "custom":
        codes = set(payload.get("codes") or [])
    elif pool in get_pools() and pool != "all":
        codes = set(get_pools()[pool]["codes"])
    if codes is not None:
        mask = np.zeros(len(P.codes), dtype=bool)
        idx = [P.cidx[c] for c in codes if c in P.cidx]
        if not idx:
            return {"error": "所选股票池在本地数据中没有可用标的"}
        mask[idx] = True
        sig &= mask[None, :]

    if payload.get("cost_on") is False:
        params["slippage"] = 0.0

    t0 = time.time()
    tr = E.simulate(P, sig, sl=params["sl"], sl_mode=params["sl_mode"],
                    atr_k=params["atr_k"], tp=params["tp"],
                    exit_mode=params["exit_mode"], tp_half=params["tp_half"],
                    max_hold=params["max_hold"], slippage=params["slippage"],
                    start=params.get("start"), end=params.get("end"))
    el = time.time() - t0

    out = {"n_signals": int(sig.sum()), "elapsed": el, "params": params}
    if len(tr) == 0:
        out["metrics"] = {"n": 0}
        return out

    m = E.metrics(tr)
    out["metrics"] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                      for k, v in m.items() if k != "label"}

    nav = E.equity_curve(tr, P, params.get("start"), params.get("end"))
    if len(nav) > 1:
        step = max(1, len(nav) // 900)
        nv = nav.iloc[::step]
        out["curve"] = [[str(i), round(float(v), 5)] for i, v in nv.items()]
        out["curve_stats"] = E.curve_stats(nav)

    b = E.bucket_report(tr)
    out["buckets"] = b.replace({np.nan: None}).to_dict("records")

    er = tr.groupby("exit_reason").net_ret.agg(["count", "mean"]).reset_index()
    out["exit_reasons"] = [{"reason": str(r["exit_reason"]), "n": int(r["count"]),
                            "avg": float(r["mean"])} for r in er.to_dict("records")]

    out["yearly"] = _yearly(tr)

    rg = tr.groupby("mkt_state").net_ret.agg(
        ["count", "mean", lambda x: float((x > 0).mean())]).reset_index()
    rg.columns = ["state", "n", "avg", "win"]
    out["regime"] = rg.to_dict("records")

    rg2 = tr.groupby("above_ma60").net_ret.agg(
        ["count", "mean", lambda x: float((x > 0).mean())]).reset_index()
    rg2.columns = ["above", "n", "avg", "win"]
    out["above_ma60"] = [{"label": "指数>MA60" if r["above"] else "指数<MA60",
                          "n": int(r["n"]), "avg": float(r["avg"]),
                          "win": float(r["win"])} for r in rg2.to_dict("records")]

    dec = tr.groupby("decile").net_ret.agg(
        ["count", "mean", lambda x: float((x > 0).mean())]).reset_index()
    dec.columns = ["decile", "n", "avg", "win"]
    out["by_decile"] = dec.to_dict("records")

    head = tr.sort_values("signal_date", ascending=False).head(200)
    out["trades"] = head[["code", "signal_date", "entry_date", "entry_px",
                          "exit_date", "exit_px", "net_ret", "hold_days",
                          "exit_reason"]].round(5).to_dict("records")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                fp = os.path.join(WEB, "index.html")
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/meta":
                P = get_panel()
                uni = get_universe()
                pools = get_pools()
                return self._send(200, json.dumps({
                    "date_start": str(P.dates[0]), "date_end": str(P.dates[-1]),
                    "n_codes": int(len(P.codes)),
                    "boards": uni.board.value_counts().to_dict(),
                    "pools": {k: {"label": v["label"], "n": len(v["codes"])}
                              for k, v in pools.items()},
                    "indexes": list(P.idx_close.keys()),
                }, ensure_ascii=False))
            if path == "/api/universe":
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                kw = (q.get("q") or [""])[0].strip()
                lim = int((q.get("limit") or ["400"])[0])
                uni = get_universe()
                if kw:
                    s = uni[uni.name.str.contains(kw, case=False, na=False)
                            | uni.code.str.contains(kw, na=False)]
                else:
                    s = uni
                s = s.head(lim)
                return self._send(200, json.dumps(
                    s[["symbol", "code", "name", "board"]].to_dict("records"),
                    ensure_ascii=False))
            return self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"error": f"bad json: {e}"}))
        if path == "/api/backtest":
            try:
                res = run(payload)
                return self._send(200, json.dumps(res, ensure_ascii=False,
                                                  default=lambda o: None))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                return self._send(500, json.dumps({"error": str(e)},
                                                  ensure_ascii=False))
        return self._send(404, json.dumps({"error": "not found"}))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8760
    print(f"[server] http://127.0.0.1:{port}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
