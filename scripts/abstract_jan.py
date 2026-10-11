"""Graphical abstract of the follow-up results (results/figures/nla_abstract_followup.png): why NLA
consensus is biased, what helps, and what perturbations show.

    uv run --no-project --with matplotlib python scripts/abstract_jan.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("results/figures")
BLUE, BLUE_LT, ORANGE, ORANGE_LT = "#2a78d6", "#8db8ec", "#eb6834", "#f6b89c"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, CARD, SURF = "#e4e3df", "#f3f2ee", "#fcfcfb"
MONO = "monospace"
for name in ("Helvetica Neue", "Helvetica", "Arial"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

fig = plt.figure(figsize=(16, 6.6), dpi=200, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 66)
ax.axis("off")


def card(x, y, w, h, fc=CARD, ec="none", r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.0, zorder=1))


def text(x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=6, **kw)


def rect(x, y, w, h, color, z=2):
    ax.add_patch(plt.Rectangle((x, y), w, h, color=color, zorder=z))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK3, lw=1.4, zorder=3))


# ---- header
text(4, 61.5, "Why NLA consensus is biased, and what perturbations can tell you", 19, weight="bold")
text(4, 57.6, "ceselder's open NLA on Qwen3.6-27B, layer 42  ·  WorkspaceBench claims about internal computation  ·  "
     "75 two-hop swaps  ·  follow-up to consensus sampling (Bauer 2026)", 11, INK2)
for x, n, title in [(4, "1", "Repeated errors misread something real"),
                    (56, "2", "A different reader helps; resampling can't"),
                    (108, "3", "Adding steers the NLA; removing doesn't")]:
    card(x, 3, 48, 50.5, fc="#ffffff", ec=GRID)
    ax.add_patch(plt.Circle((x + 3.2, 49.3), 1.55, color=INK, zorder=4))
    text(x + 3.2, 49.2, n, 10, "#ffffff", "bold", ha="center")
    text(x + 5.6, 49.3, title, 12.5, weight="bold")

# ---- panel 1: the "10 - 1 =" example with the lens profile of "1"
x0 = 4
text(x0 + 2.5, 45.2, "Prompt “10 − 1 =”, then the model's answer “9”", 9.5, INK2)
toks = ["1", "0", "−", "1", "=", "9"]
strength = [0.0, 0.98, 0.0, 1.0, 0.0, 0.84]  # J++ score of the word "1" at each position, layer 42
for k, (t, s_) in enumerate(zip(toks, strength)):
    xx = x0 + 3 + k * 6.6
    card(xx, 37.2, 5.6, 5.6, fc="#ffffff", ec=INK if k == 4 else GRID, r=0.6)
    text(xx + 2.8, 40.0, t, 14, INK, "bold", ha="center", family=MONO)
    rect(xx + 0.8, 31.6, 4.0, 0.12, GRID)
    if s_ > 0:
        rect(xx + 0.8, 31.6, 4.0, 4.6 * s_, ORANGE)
    else:
        text(xx + 2.8, 32.6, "0", 8.5, INK3, ha="center")
text(x0 + 3 + 4 * 6.6 + 2.8, 29.9, "NLA reads here", 8.5, INK, "bold", ha="center")
text(x0 + 2.5, 29.9, "lens shows “1”", 8.5, ORANGE, "bold")
text(x0 + 2.5, 26.0, "NLA, 8 of 8 samples:", 9.5, INK2)
text(x0 + 18.0, 26.0, "“…likely ‘1’”", 13, ORANGE, "bold", family=MONO)
text(x0 + 2.5, 22.5, "The operand is salient; the NLA reads it as the answer.", 9.5, INK)
text(x0 + 2.5, 19.2, "Claimed word peaks earlier in the prompt", 9.5, INK2)
for i, (lab, v, c) in enumerate([("wrong claims", 65, ORANGE), ("correct claims", 33, BLUE)]):
    yy = 15.8 - i * 3.3
    rect(x0 + 15, yy - 1.05, v * 0.28, 2.1, c)
    text(x0 + 15 + v * 0.28 + 0.8, yy, f"{v}%", 10, INK, "bold")
    text(x0 + 14.2, yy, lab, 9, INK2, ha="right")
text(x0 + 2.5, 8.7, "39 of 121 wrong claims repeat in ≥ 7 of 8 samples,", 9.5, INK, "bold")
text(x0 + 2.5, 6.0, "and at neighbouring positions: the feature is real, its role is not", 9.5, INK, "bold")

# ---- panel 2: AUROC on the repeated errors
x0 = 56
text(x0 + 2.5, 45.2, "AUROC, correct vs wrong, on claims the NLA repeats", 9.5, INK2)
text(x0 + 2.5, 42.6, "(≥ 7 of 8 samples agree; 316 correct, 28 wrong)", 9.5, INK2)
rows = [("8 samples of the same vector", 0.58, BLUE_LT), ("NLA at following positions*", 0.54, BLUE_LT),
        ("3 preceding positions", 0.68, BLUE_LT), ("samples + preceding", 0.72, BLUE),
        ("J++ Lens shows the claimed word", 0.83, ORANGE), ("removal test", 0.51, GRID)]
x_c, scale = x0 + 25, 40  # x of AUROC 0.5, units per AUROC
for i, (lab, v, c) in enumerate(rows):
    yy = 37.8 - i * 4.4
    rect(x_c, yy - 1.3, (v - 0.5) * scale, 2.6, c)
    text(x_c + (v - 0.5) * scale + 0.8, yy, f"{v:.2f}", 10.5, INK, "bold")
    text(x_c - 0.8, yy, lab, 9, INK2, ha="right")
ax.plot([x_c, x_c], [10.5, 40], color=INK3, lw=0.8, zorder=1)
text(x_c, 9.3, "0.5 = chance", 8.5, INK3, ha="center")
text(x0 + 2.5, 6.0, "A reader with independent errors catches what agreement can't", 9.5, INK, "bold")
text(x0 + 47.5, 4.3, "* 100-item pilot", 7.5, INK3, ha="right")

# ---- panel 3: add vs remove
x0 = 108
card(x0 + 2.5, 27.4, 43, 16.4)
text(x0 + 4.2, 41.3, "ADD: swap the bridge's lens direction", 10, INK, "bold")
text(x0 + 4.2, 37.9, "“the capital of the country where the Eiffel Tower is…”", 8.8, INK2, family=MONO)
text(x0 + 4.2, 34.4, "France", 14, INK, "bold", family=MONO)
arrow(x0 + 14.5, 34.4, x0 + 18.5, 34.4)
text(x0 + 19.6, 34.4, "Italy", 14, ORANGE, "bold", family=MONO)
text(x0 + 30.5, 35.6, "model's answer", 8.5, INK2)
text(x0 + 30.5, 33.0, "changes in 81%", 10.5, INK, "bold")
text(x0 + 4.2, 30.9, "NLA names Italy in 100% of samples, including the 14 pairs", 8.8, INK)
text(x0 + 4.2, 29.0, "where the model still answers Paris", 8.8, INK)
card(x0 + 2.5, 9.8, 43, 15.8)
text(x0 + 4.2, 23.3, "REMOVE: ablate the claimed word's lens directions", 10, INK, "bold")
text(x0 + 4.2, 20.0, "the word leaves the lens's top 2,000; the claim stays (AUROC 0.49)", 8.8, INK2)
for i, (lab, a_, b_) in enumerate([("correct claims", 6.5, 6.0), ("wrong claims", 4.3, 3.5)]):
    yy = 16.3 - i * 3.4
    text(x0 + 4.2, yy, lab, 9, INK2)
    text(x0 + 19, yy, f"{a_} to {b_} of 8 samples", 10, INK, "bold")
text(x0 + 2.5, 6.0, "Grounded ≠ correct ≠ used by the model", 9.5, INK, "bold")
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "nla_abstract_followup.png", facecolor=SURF)
print("wrote", OUT / "nla_abstract_followup.png")
