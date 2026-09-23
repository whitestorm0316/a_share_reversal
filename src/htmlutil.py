# -*- coding: utf-8 -*-
"""报告用轻量 HTML 组件（纯 SVG，无外部依赖）。"""
import html as _html

import numpy as np
import pandas as pd

C_RED = "#c0392b"      # 涨/正（A股习惯）
C_GREEN = "#1e7d5a"    # 跌/负
C_BLUE = "#2b6cb0"
C_TX = "#1c2330"
C_TX2 = "#5b6577"
C_TX3 = "#8b95a7"
C_LINE = "#e3e6ec"


def esc(x):
    return _html.escape(str(x))


def table(df, cols=None, headers=None, fmt=None, align=None, cls=""):
    """df → HTML 表格。fmt: {col: callable}"""
    d = df if cols is None else df[cols]
    headers = headers or list(d.columns)
    fmt = fmt or {}
    out = [f'<table class="{cls}"><thead><tr>']
    for h in headers:
        out.append(f"<th>{esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    for _, r in d.iterrows():
        out.append("<tr>")
        for i, c in enumerate(d.columns):
            v = r[c]
            f = fmt.get(c)
            s = f(v) if f else _default_fmt(v)
            klass = ""
            if isinstance(v, (int, float, np.floating)) and c not in ("n", "decile"):
                if v > 0.0005:
                    klass = ' class="pos"'
                elif v < -0.0005:
                    klass = ' class="neg"'
            out.append(f"<td{klass}>{s}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _default_fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float) or isinstance(v, np.floating):
        if not np.isfinite(v):
            return "—"
        if abs(v) < 3 and abs(v) > 0:
            return f"{v*100:.2f}%"
        return f"{v:.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return esc(v)


def bar_svg(labels, values, width=640, height=220, colors=None,
            value_fmt=None, horizontal=False, title=""):
    n = len(labels)
    if n == 0:
        return "<p>无数据</p>"
    value_fmt = value_fmt or (lambda v: f"{v*100:.2f}%")
    mx = max(abs(v) for v in values) or 1.0
    if horizontal:
        row_h = max(20, (height - 30) / n)
        height = int(row_h * n + 30)
        pl, pr = 190, 90
        barw = width - pl - pr
        out = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
        zero = pl
        for i, (lb, v) in enumerate(zip(labels, values)):
            y = i * row_h + 14
            w = abs(v) / mx * barw
            col = (colors[i] if colors else (C_RED if v >= 0 else C_GREEN))
            out.append(f'<text x="{pl-8}" y="{y+row_h*0.62:.1f}" font-size="12" '
                       f'fill="{C_TX2}" text-anchor="end">{esc(lb)}</text>')
            out.append(f'<rect x="{zero:.1f}" y="{y:.1f}" width="{w:.1f}" '
                       f'height="{row_h*0.62:.1f}" fill="{col}" opacity="0.85"/>')
            out.append(f'<text x="{zero+w+6:.1f}" y="{y+row_h*0.62:.1f}" font-size="11.5" '
                       f'fill="{C_TX2}">{value_fmt(v)}</text>')
        out.append(f'<line x1="{zero}" x2="{zero}" y1="8" y2="{height-16}" '
                   f'stroke="{C_LINE}"/>')
        out.append("</svg>")
        return "".join(out)

    pl, pr, pt, pb = 56, 12, 16, 48
    barw = (width - pl - pr) / n
    span = height - pt - pb
    zero = pt + span / 2
    out = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
    out.append(f'<line x1="{pl}" x2="{width-pr}" y1="{zero:.1f}" y2="{zero:.1f}" '
               f'stroke="{C_LINE}"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        hgt = abs(v) / mx * (span / 2) * 0.92
        y = zero - hgt if v >= 0 else zero
        col = (colors[i] if colors else (C_RED if v >= 0 else C_GREEN))
        x = pl + i * barw + barw * 0.16
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{barw*0.68:.1f}" '
                   f'height="{max(hgt,0.8):.1f}" fill="{col}" opacity="0.85"/>')
        out.append(f'<text x="{x+barw*0.34:.1f}" y="{y-3:.1f}" font-size="10" '
                   f'fill="{C_TX2}" text-anchor="middle">{value_fmt(v)}</text>')
        out.append(f'<text x="{x+barw*0.34:.1f}" y="{height-30:.1f}" font-size="10.5" '
                   f'fill="{C_TX3}" text-anchor="middle">{esc(lb)}</text>')
    out.append(f'<text x="{pl-6}" y="{pt+8}" font-size="10" fill="{C_TX3}" '
               f'text-anchor="end">{value_fmt(mx)}</text>')
    out.append("</svg>")
    return "".join(out)


def line_svg(xs, ys, width=640, height=230, ylab="", ref=None):
    if not xs:
        return "<p>无数据</p>"
    pl, pr, pt, pb = 58, 14, 16, 34
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.08 or 0.05
    a, b = lo - pad, hi + pad
    xp = lambda i: pl + (width - pl - pr) * i / (len(xs) - 1 or 1)
    yp = lambda v: pt + (height - pt - pb) * (1 - (v - a) / (b - a or 1))
    path = "".join((("M" if i == 0 else "L") + f"{xp(i):.1f} {yp(v):.1f}")
                   for i, v in enumerate(ys))
    out = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
    for k in range(5):
        v = a + (b - a) * k / 4
        out.append(f'<line x1="{pl}" x2="{width-pr}" y1="{yp(v):.1f}" y2="{yp(v):.1f}" '
                   f'stroke="{C_LINE}"/>')
        out.append(f'<text x="{pl-6}" y="{yp(v)+4:.1f}" font-size="10" fill="{C_TX3}" '
                   f'text-anchor="end">{v:.2f}</text>')
    if ref is not None and a < ref < b:
        out.append(f'<line x1="{pl}" x2="{width-pr}" y1="{yp(ref):.1f}" y2="{yp(ref):.1f}" '
                   f'stroke="#c8d0dc" stroke-dasharray="3 3"/>')
    out.append(f'<path d="{path}" fill="none" stroke="{C_BLUE}" stroke-width="1.7"/>')
    for i in [0, len(xs) // 2, len(xs) - 1]:
        out.append(f'<text x="{xp(i):.1f}" y="{height-10}" font-size="10" '
                   f'fill="{C_TX3}" text-anchor="middle">{esc(xs[i])}</text>')
    out.append("</svg>")
    return "".join(out)


def pct(v, d=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v*100:.{d}f}%"


def num(v, d=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{d}f}"
