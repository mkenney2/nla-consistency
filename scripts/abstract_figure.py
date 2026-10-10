"""Graphical abstract for the NLA confabulation results (results/figures/nla_abstract.png).

    uv run --no-project --with matplotlib python scripts/abstract_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (light): blue = supported by the activation / probe, orange = NLA error
BLUE, BLUE_LT, ORANGE = "#2a78d6", "#8db8ec", "#eb6834"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, CARD, SURF = "#e4e3df", "#f3f2ee", "#fcfcfb"
MONO = "monospace"

for name in ("Helvetica Neue", "Helvetica", "Arial"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

fig = plt.figure(figsize=(16, 6.4), dpi=200, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 64)
ax.axis("off")


def card(x, y, w, h, fc=CARD, ec="none", lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=1))


def text(x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=6, **kw)


def rect(x, y, w, h, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, color=color, zorder=2))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK3, lw=1.4, zorder=3))


# ---- header
text(4, 59.5, "What an NLA's own consistency can catch, and what it drops", 19, weight="bold")
text(4, 55.6, "ceselder's open NLA on Qwen3.6-27B, layer 42  ·  500 WorkspaceBench items with known targets  ·  "
     "600 arithmetic prompts  ·  extends consensus sampling (Bauer 2026) and the readout gap (Dingeto 2026)", 11, INK2)

for x, n, title in [(4, "1", "Agreement across samples filters confabulation"),
                    (56, "2", "...except errors the NLA repeats"),
                    (108, "3", "A probe reads the digits; the NLA, their size")]:
    card(x, 3, 48, 48.5, fc="#ffffff", ec=GRID)
    ax.add_patch(plt.Circle((x + 3.2, 47.3), 1.55, color=INK, zorder=4))
    text(x + 3.2, 47.2, n, 10, "#ffffff", "bold", ha="center")
    text(x + 5.6, 47.3, title, 12.5, weight="bold")

# ---- panel 1: the filter
x0 = 4
card(x0 + 2.5, 30, 9, 11, fc="#ffffff", ec=INK3)
for i in range(9):
    rect(x0 + 3.6 + i * 0.75, 33, 0.55, 1.2 + (i * 37 % 7), BLUE)
text(x0 + 7, 31.6, "activation", 9, INK2, ha="center")
arrow(x0 + 12.3, 35.5, x0 + 17, 35.5)
text(x0 + 14.65, 37.7, "NLA × 8", 9, INK2, ha="center")
for r, (hits, kept) in enumerate([([1, 1, 1, 1, 1, 1, 1, 0], True), ([1, 0, 0, 0, 0, 0, 0, 0], False),
                                   ([1, 1, 1, 0, 1, 1, 1, 1], True)]):
    yy = 40.5 - r * 4.2
    for k, h in enumerate(hits):
        rect(x0 + 18 + k * 2.05, yy - 1.1, 1.55, 2.2, (BLUE if kept else ORANGE) if h else GRID)
    text(x0 + 35, yy, f"{sum(hits)}/8", 10, INK, "bold")
    text(x0 + 39.3, yy, "keep" if kept else "drop", 10, BLUE if kept else ORANGE, "bold")
text(x0 + 18, 29.3, "one row per claim; a square = a sample stating it", 8.5, INK3)

text(x0 + 2.5, 25.6, "False claims about the input", 10.5, INK2)
for i, (lab, v, c) in enumerate([("raw NLA", 38, INK3), ("keep claims in ≥ 2 of 8 samples", 13, BLUE)]):
    yy = 20.2 - i * 5.6
    rect(x0 + 2.5, yy - 1.3, v * 0.62, 2.6, c)
    text(x0 + 2.5 + v * 0.62 + 1, yy, f"{v}%", 12, INK, "bold")
    text(x0 + 2.5, yy + 2.5, lab, 9, INK2)
text(x0 + 2.5, 9.6, "keeps 87% of workspace claims  ·  no source text needed", 9, INK2)
text(x0 + 2.5, 6.4, "AUROC  0.90 input claims  ·  0.75 internal claims", 9.5, INK, "bold")

# ---- panel 2: consistent wrong copy
x0 = 56
card(x0 + 2.5, 33.5, 43, 9.5)
text(x0 + 4.2, 40.7, "model reads", 9, INK2)
text(x0 + 4.2, 37.2, "10 − 1 =", 17, INK, "bold", family=MONO)
arrow(x0 + 19.5, 37.8, x0 + 24, 37.8)
text(x0 + 25.2, 40.7, "NLA says, in 8 of 8 samples", 9, INK2)
text(x0 + 25.2, 37.2, "“2 − 1 = 1”", 15, ORANGE, "bold", family=MONO)
text(x0 + 2.5, 29.8, "It rebuilds a consistent wrong copy of the task, then answers that.", 10, INK)
text(x0 + 2.5, 26.9, "Resampling agrees with itself, so it cannot flag this.", 10, INK2)

text(x0 + 2.5, 23.3, "121 wrong claims about internal computation", 10.5, INK2)
segs = [("vary across samples", 82, BLUE), ("repeat in ≥ 7 of 8 samples, absent at the preceding tokens", 27, BLUE_LT),
        ("repeat in samples and at the preceding tokens", 12, ORANGE)]
xx, W = x0 + 2.5, 43
for i, (lab, n, c) in enumerate(segs):
    w = n / 121 * W
    rect(xx + 0.15, 16.4, w - 0.3, 4.4, c)
    text(xx + w / 2, 18.6, f"{n}", 11, INK if c == BLUE_LT else "#ffffff", "bold", ha="center")
    ly = 13.4 - i * 2.4
    rect(x0 + 2.5, ly - 0.75, 1.5, 1.5, c)
    text(x0 + 4.8, ly, lab, 9, INK2)
    xx += w
text(x0 + 2.5, 6.4, "AUROC 0.75 with samples, 0.80 adding the preceding tokens", 9.5, INK, "bold")

# ---- panel 3: probe vs NLA
x0 = 108
card(x0 + 2.5, 35.5, 43, 7.5)
text(x0 + 4.2, 39.2, "51 − 49 =", 15, INK, "bold", family=MONO)
arrow(x0 + 18.5, 39.2, x0 + 22.5, 39.2)
text(x0 + 23.6, 40.9, "NLA reads", 9, INK2)
text(x0 + 23.6, 38.0, "“64 − 49”", 13.5, ORANGE, "bold", family=MONO)
text(x0 + 36.2, 40.9, "probe reads", 9, INK2)
text(x0 + 36.2, 38.0, "51, 49", 13.5, BLUE, "bold", family=MONO)
text(x0 + 2.5, 31.8, "Read exactly right from the same layer-42 activation", 10.5, INK2)
for g, (lab, p, n) in enumerate([("first operand a", 83, 5), ("second operand b", 93, 38), ("operator", 100, 100)]):
    yb = 27.8 - g * 6.6
    text(x0 + 2.5, yb, lab, 9.5, INK)
    for j, (v, c, who) in enumerate([(p, BLUE, "linear probe"), (n, ORANGE, "NLA")]):
        yy = yb - 1.9 - j * 2.3
        rect(x0 + 15, yy - 0.9, 0.27 * v, 1.8, c)
        text(x0 + 15 + 0.27 * v + 0.8, yy, f"{v}%", 9.5, INK, "bold")
        if g == 0:
            text(x0 + 14.2, yy, who, 8.5, INK2, ha="right")
text(x0 + 2.5, 6.9, "Swapping the model's operand subspace moves the NLA;", 9.5, INK, "bold")
text(x0 + 2.5, 4.5, "swapping the probe's own directions does not", 9.5, INK, "bold")

fig.savefig(OUT / "nla_abstract.png", facecolor=SURF)
print("wrote", OUT / "nla_abstract.png")
