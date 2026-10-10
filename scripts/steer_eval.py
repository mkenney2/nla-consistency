"""Laptop side of the steering test (inputs from scripts/steer_pod.py in results/steer/).

For each pair set (a, b, b1) and condition, from the NLA's quoted "x op y" expression:
  exact->src : share of readouts naming the source operand
  exact->base: share naming the base operand
  closer->src: share of readouts whose operand is nearer the source value than the base value
               (ties dropped); robust to the odd wild parse, unlike a regression slope.
Plus the probe and model-answer rates from pairs.json.

    uv run --no-project python scripts/steer_eval.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

R = Path("results/steer")
FEW = {(12, "+", 7), (45, "-", 13), (8, "+", 26), (71, "-", 9)}
OPEN = re.compile(r"(?<![\d.])(\d+)[ \t]*([-−+])[ \t]*(\d+)[ \t]*=(?![ \t]*\d)")
CONDS = ["base", "src", "swap_mean", "swap_lr", "swap_rand"]


def parse(t: str):
    """The NLA's guess at the target: the incomplete equations it quotes ("96 - 34 =", no answer),
    most frequent first, then first mentioned. Completed lines like "54 + 43 = 97" are invented
    worksheet context, not the target."""
    hits = [(int(x), "-" if o == "−" else o, int(y)) for x, o, y in OPEN.findall(t)]
    hits = [h for h in hits if h not in FEW]
    return max(hits, key=lambda h: (hits.count(h), -hits.index(h))) if hits else None


def ans(s: str):
    t = s.strip().split()[0] if s.strip() else ""
    return int(t) if t.lstrip("-").isdigit() else None


def truth(p, which):
    x, y = p[which]
    return x - y if p["op"] == "-" else x + y


def main() -> None:
    rows = {r["id"]: r for r in json.loads((R / "pairs.json").read_text())}
    reads = defaultdict(list)
    for l in (R / "nla.jsonl").read_text().splitlines():
        d = json.loads(l)
        pid, c, _ = d["key"].split("|")
        reads[(pid, c)].append(parse(d["text"]))

    print("model right on base prompts:", sum(ans(r["model"]["base"]) == truth(r, "base") for r in rows.values()), "/", len(rows),
          "| self-patch unchanged:", sum(r["model"]["self"] == r["model"]["base"] for r in rows.values()))
    for var in ("a", "b", "b1"):
        i = 0 if var == "a" else 1
        rs = [r for r in rows.values() if r["var"] == var]
        print(f"\n## operand {'a' if var == 'a' else 'b'} ({'one digit' if var == 'b1' else 'two digits'}), {len(rs)} pairs\n")
        print("| condition | probe -> src | model answer -> src | NLA parsed | NLA -> src | NLA -> base | NLA closer to src |")
        print("|---|---|---|---|---|---|---|")
        for c in CONDS:
            pc = c
            mc = "full" if c == "src" else c
            probe = sum(r["probe"][pc][i] == r["src"][i] for r in rs) / len(rs)
            model = sum(ans(r["model"][mc]) == truth(r, "src") for r in rs) / len(rs) if mc in rs[0]["model"] else float("nan")
            parsed = tot = to_src = to_base = 0
            near = far = 0
            for r in rs:
                cur = [p[0] if i == 0 else p[2] for p in reads[(r["id"], c)] if p]
                base = [p[0] if i == 0 else p[2] for p in reads[(r["id"], "base")] if p]
                tot += len(reads[(r["id"], c)])
                parsed += len(cur)
                to_src += sum(v == r["src"][i] for v in cur)
                to_base += sum(v == r["base"][i] for v in cur)
                for v in cur:
                    ds, db = abs(v - r["src"][i]), abs(v - r["base"][i])
                    near += ds < db
                    far += db < ds
            print(f"| {c if c != 'src' else 'src (whole vector)'} | {probe:.0%} | {model:.0%} | {parsed / max(tot, 1):.0%} | "
                  f"{to_src / max(parsed, 1):.0%} | {to_base / max(parsed, 1):.0%} | {near / max(near + far, 1):.0%} |")


if __name__ == "__main__":
    main()
