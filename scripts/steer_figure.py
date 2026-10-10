"""Figure for the steering test (results/figures/steer.png), from results/steer/.

    uv run --no-project --with matplotlib python scripts/steer_figure.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).parent))
from steer_eval import R, parse  # noqa: E402

OUT = Path("results/figures")
BLUE, ORANGE, GREY, DARK, LIGHT = "#2a78d6", "#eb6834", "#b9b8b3", "#0b0b0b", "#dcdbd6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
for name in ("Helvetica Neue", "Helvetica", "Arial"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

CONDS = [("base", "nothing (original vector)", LIGHT), ("src", "whole source vector", DARK),
         ("swap_mean", "18 class-mean directions", BLUE), ("swap_lr", "18 probe-weight directions", ORANGE),
         ("swap_rand", "18 random directions", GREY)]
SETS = [("b1", "Second operand, one digit", 1), ("a", "First operand, two digits", 0), ("b", "Second operand, two digits", 1)]


def stats():
    rows = {r["id"]: r for r in json.loads((R / "pairs.json").read_text())}
    reads = defaultdict(list)
    for l in (R / "nla.jsonl").read_text().splitlines():
        d = json.loads(l)
        pid, c, _ = d["key"].split("|")
        reads[(pid, c)].append(parse(d["text"]))
    out = {}
    for var, _, i in SETS:
        rs = [r for r in rows.values() if r["var"] == var]
        for c, _, _ in CONDS:
            near = far = 0
            for r in rs:
                for p in reads[(r["id"], c)]:
                    if p:
                        v = p[0] if i == 0 else p[2]
                        ds, db = abs(v - r["src"][i]), abs(v - r["base"][i])
                        near, far = near + (ds < db), far + (db < ds)
            out[(var, c)] = {"nla": near / max(near + far, 1),
                             "probe": None if c == "base" else sum(r["probe"][c][i] == r["src"][i] for r in rs) / len(rs)}
    return out


def main() -> None:
    s = stats()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=200, facecolor=SURF, sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.74, bottom=0.14, wspace=0.08)
    fig.text(0.012, 0.95, "The NLA follows the directions the model uses, not the ones a probe reads", fontsize=17, weight="bold", color=INK)
    fig.text(0.012, 0.89, "Qwen3.6-27B, layer 42, at the “=” of few-shot “a ± b =” prompts: one operand's subspace is swapped in from a prompt "
             "with a different value (100 pairs per panel, 8 NLA samples each).", fontsize=10.5, color=INK2)
    fig.text(0.012, 0.85, "Bars: share of NLA readouts whose number is closer to the swapped-in value than to the original. "
             "Right column: share of prompts where a linear probe reads the swapped-in value.", fontsize=10.5, color=INK2)
    ys = list(range(len(CONDS)))[::-1]
    for ax, (var, title, _) in zip(axes, SETS):
        ax.set_facecolor(SURF)
        for y, (c, lab, col) in zip(ys, CONDS):
            v = s[(var, c)]
            ax.barh(y, v["nla"] * 100, height=0.62, color=col, zorder=2)
            light = col in (LIGHT, GREY)
            ax.text(v["nla"] * 100 + (1.5 if v["nla"] < 0.2 else -1.5), y, f"{v['nla']:.0%}", va="center",
                    ha="left" if v["nla"] < 0.2 else "right", fontsize=11, weight="bold",
                    color=INK if light or v["nla"] < 0.2 else "#ffffff", zorder=3)
            ax.text(122, y, "–" if v["probe"] is None else f"{v['probe']:.0%}", va="center", ha="right", fontsize=11,
                    color=INK2, zorder=3)
        ax.set_title(title, loc="left", fontsize=12.5, weight="bold", color=INK, pad=8)
        ax.set_xlim(0, 123)
        ax.text(122, len(CONDS) - 0.35, "probe", ha="right", va="bottom", fontsize=9.5, color=INK2)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=9.5, color=INK2)
        ax.set_xbound(0, 123)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(ys, [lab for _, lab, _ in CONDS], fontsize=11, color=INK)
    fig.text(0.012, 0.035, "Probe-weight directions move the probe (92–97%) but not the NLA, which stays at the random level; they also never change the "
             "model's answer, while class-mean swaps do (28% on one-digit operands).", fontsize=10, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "steer.png", facecolor=SURF)
    print("wrote", OUT / "steer.png")


if __name__ == "__main__":
    main()
