"""Figure: how well each check separates correct from wrong NLA claims about internal computation
(results/figures/checks.png), from results/figures/checks_auroc.json (one common set of claims: those that
name a checkable word and have removal-test results).

    uv run --no-project --with matplotlib python scripts/checks_figure.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path("results/figures")
BLUE, BLUE_LT, ORANGE, DARK, GREY = "#2a78d6", "#8db8ec", "#eb6834", "#0b0b0b", "#b9b8b3"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
for name in ("Helvetica Neue", "Helvetica", "Arial"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

COLORS = {"8 samples": BLUE_LT, "3 preceding tokens": BLUE_LT, "samples + preceding": BLUE,
          "J++ lens": ORANGE, "samples + preceding + lens": DARK, "removal test": GREY}
LABELS = {"8 samples": "agreement across 8 samples", "3 preceding tokens": "agreement with 3 preceding tokens",
          "samples + preceding": "both agreement checks", "J++ lens": "J++ lens shows the claimed word",
          "samples + preceding + lens": "both agreement checks + lens", "removal test": "removal test (claim drops when its word is ablated)"}


def main() -> None:
    d = json.loads((OUT / "checks_auroc.json").read_text())
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=200, facecolor=SURF, sharey=True)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.76, bottom=0.12, wspace=0.08)
    fig.text(0.012, 0.95, "Which checks tell right from wrong NLA claims about internal computation?", fontsize=17, weight="bold", color=INK)
    fig.text(0.012, 0.885, "AUROC, correct vs wrong claims; all from the activation alone, no source text. One common set of claims that "
             "name a specific word (the lens and removal tests need one).", fontsize=10.5, color=INK2)
    fig.text(0.012, 0.845, "0.5 = chance. The removal test mean-ablates the J++ directions of the claimed word in the vector the NLA "
             "reads and counts how often the claim survives in 8 resamples.", fontsize=10.5, color=INK2)
    ys = list(range(len(rows)))[::-1]
    for ax, col, title in [(axes[0], 1, f"All claims ({d['n'][0]} correct, {d['n'][1]} wrong)"),
                           (axes[1], 2, f"Claims the NLA repeats in ≥ 7 of 8 samples ({d['n_rep'][0]} correct, {d['n_rep'][1]} wrong)")]:
        ax.set_facecolor(SURF)
        for y, r in zip(ys, rows):
            v = r[col]
            ax.barh(y, v - 0.5, left=0.5, height=0.62, color=COLORS[r[0]], zorder=2)
            ax.text(max(v, 0.5) + 0.006, y, f"{v:.2f}", va="center", fontsize=11, weight="bold", color=INK, zorder=3)
        ax.axvline(0.5, color=INK2, lw=1, zorder=1)
        ax.set_xlim(0.45, 0.9)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9], ["0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=9.5, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color=INK, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(ys, [LABELS[r[0]] for r in rows], fontsize=11, color=INK)
    fig.text(0.012, 0.03, "Self-agreement weakens on errors the NLA repeats (0.58); the lens holds up (0.83, but only 28 wrong claims). "
             "Removing the claimed word's directions does not work (0.49–0.51).", fontsize=10, color=INK2)
    fig.savefig(OUT / "checks.png", facecolor=SURF)
    print("wrote", OUT / "checks.png")


if __name__ == "__main__":
    main()
