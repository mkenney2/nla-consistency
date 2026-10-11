"""Laptop side of the grounding tests (inputs from scripts/ground_pod.py in results/ground/).

Test 1: for each internal claim that names a word, a Gemini judge counts how many of 8 NLA samples state
it after the claim's J++ read directions were mean-ablated ("ablate") and after a random subspace of the
same rank was ("control"). sensitivity = control - ablate. A claim read from the activation should lose
support under its own ablation and not under the control.
Test 2: share of NLA samples naming the swapped-in bridge (or its answer) vs the original, per condition,
split by whether the swap changed the model's answer.

    workspace-bench/.venv/bin/python scripts/ground_eval.py judge|report
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nla_verify_eval import auroc, jl  # noqa: E402
from t12_eval import SUPPORT, T, V, ask, run_cached  # noqa: E402

R = Path("results/ground")


def readouts(suffix: str = "") -> dict:
    out = defaultdict(list)
    for d in jl(R / f"nla{suffix}.jsonl"):
        k, _ = d["key"].rsplit("|", 1)
        out[k].append(d["text"])
    return out


def judge() -> None:
    rd = readouts()
    claims = json.loads((R / "test1_claims.json").read_text())
    sch = {"type": "object", "properties": {"support": {"type": "integer"}}, "required": ["support"], "additionalProperties": False}
    sys_s = SUPPORT.format(n=8, what="8 samples from a slightly edited version of the same activation")
    jobs = {}
    for c in claims:
        for cond in ("ablate", "control"):
            texts = rd.get(f"t1|{c['group']}|{cond}", [])
            if len(texts) == 8:
                others = "\n\n".join(f"<description {k + 1}>\n{t}\n</description {k + 1}>" for k, t in enumerate(texts))
                jobs[f"{c['key']}|{cond}"] = f"CLAIM: {c['claim']}\n\nOTHER DESCRIPTIONS:\n{others}"
    run_cached(R / "support_t1.jsonl", jobs, lambda cl, u: ask(cl, sys_s, u, sch))


def test1() -> None:
    claims = json.loads((R / "test1_claims.json").read_text())
    sup = {d["key"]: d["support"] for d in jl(V / "support.jsonl")} | {d["key"]: d["r"]["support"] for d in jl(T / "support_new.jsonl")}
    adj = {d["key"]: d["r"]["support"] for d in jl(T / "support_adj.jsonl")}
    res = {d["key"]: d["r"]["support"] for d in jl(R / "support_t1.jsonl")}
    meta = {m["group"]: m for m in json.loads((R / "test1_meta.json").read_text())}
    recs = []
    for c in claims:
        a, r = res.get(f"{c['key']}|ablate"), res.get(f"{c['key']}|control")
        if a is None or r is None or c["key"] not in sup or c["key"] not in adj:
            continue
        recs.append({**c, "s": sup[c["key"]], "adj": adj[c["key"]], "ablate": a, "control": r, "sens": r - a})
    ms = list(meta.values())
    med = lambda v: sorted(v)[len(v) // 2]
    print("## Test 1: does a claim depend on the directions of the word it names?\n")
    print(f"Lens check (J++ rank of the named word, median over {len(ms)} groups): before {med([m['jpp_rank_before'] for m in ms])}, "
          f"after ablation {med([m['jpp_rank_ablate'] for m in ms])}, after the random control {med([m['jpp_rank_control'] for m in ms])}; "
          f"ablated rank {med([m['rank'] for m in ms])} dims, vector change {med([m['norm_change'] for m in ms]):.1%} of its norm\n")
    print("| claims | n | support, original | after control ablation | after own ablation | sensitivity (control - own) |\n|---|---|---|---|---|---|")
    for name, g in [("correct", lambda r: r["ws"] == "correct"), ("wrong", lambda r: r["ws"] == "wrong"),
                    ("correct, repeated >= 7/8", lambda r: r["ws"] == "correct" and r["s"] >= 7),
                    ("wrong, repeated >= 7/8", lambda r: r["ws"] == "wrong" and r["s"] >= 7)]:
        rs = [r for r in recs if g(r)]
        if rs:
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"| {name} | {len(rs)} | {m('s'):.1f} | {m('control'):.1f} | {m('ablate'):.1f} | {m('sens'):+.1f} |")
    print("\n| signal | AUROC correct vs wrong | among repeated >= 7/8 |\n|---|---|---|")
    cons = [r for r in recs if r["s"] >= 7]
    for name, f in {"sensitivity alone": lambda r: r["sens"],
                    "samples + preceding (Result 1)": lambda r: r["s"] / 8 + r["adj"] / 3,
                    "samples + preceding + sensitivity": lambda r: r["s"] / 8 + r["adj"] / 3 + r["sens"] / 8}.items():
        print(f"| {name} | {auroc([f(r) for r in recs if r['ws'] == 'correct'], [f(r) for r in recs if r['ws'] == 'wrong']):.2f} | "
              f"{auroc([f(r) for r in cons if r['ws'] == 'correct'], [f(r) for r in cons if r['ws'] == 'wrong']):.2f} |")
    print("\nRepeated wrong claims (support original / control / own ablation):")
    for r in sorted([r for r in cons if r["ws"] == "wrong"], key=lambda r: -r["sens"]):
        print(f"  {r['s']}/{r['control']}/{r['ablate']}  {r['claim'][:110]}")


def test2(suffix: str = "") -> None:
    d = json.loads((R / f"test2_rows{suffix}.json").read_text())
    rd = readouts(suffix)
    rows = d["rows"]
    lay = lambda L: f"layers {L[0]}-{L[-1]}" if isinstance(L, list) else f"layer {L}"
    print(f"\n## Test 2{' (wide swap)' if suffix else ''}: swapping a computed intermediate ({lay(d['layer'])}, last {d.get('npos', 1)} "
          f"position(s), beta {d['beta']}; {len(rows)} pairs)\n")
    print("Grid on the first 15 pairs (share where the model's answer becomes the new bridge's):")
    print("  " + ", ".join(f"{lay(g['layer'])} x{g.get('npos', 1)} b{g['beta']}: {g['success']:.0%}" for g in d["grid"]))
    print(f"\nmodel answer becomes the new bridge's: swap {sum(r['swap_success'] for r in rows) / len(rows):.0%}, "
          f"random control {sum(r['control_success'] for r in rows) / len(rows):.0%} (control keeps the original answer "
          f"{sum(r['control_unchanged'] for r in rows) / len(rows):.0%})\n")

    def share(texts, words):
        pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
        return sum(bool(pat.search(t)) for t in texts) / max(len(texts), 1)

    print("| pairs | condition | NLA samples naming the new bridge or its answer | naming the original |\n|---|---|---|---|")
    for name, sel in [("all", lambda r: True), ("model answer swapped", lambda r: r["swap_success"]),
                      ("model answer not swapped", lambda r: not r["swap_success"])]:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        for cond in ("base", "swap", "control"):
            new = [share(rd[f"t2|{r['id']}|{cond}"], [r["y"], r["y_ans"]]) for r in rs]
            old = [share(rd[f"t2|{r['id']}|{cond}"], [r["x"], r["x_ans"]]) for r in rs]
            print(f"| {name} ({len(rs)}) | {cond} | {sum(new) / len(new):.0%} | {sum(old) / len(old):.0%} |")
    print("\nNote: legs answers are digits, so 'naming the answer' there can match unrelated numbers.")


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        judge()
    elif sys.argv[1] == "report2":
        test2("2")
    else:
        test2()
        test1()
