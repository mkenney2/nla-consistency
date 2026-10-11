"""Laptop side of the J++ Lens test (inputs from scripts/jpp_pod.py in results/jpp/).

  1. Replication: recall@10 of the J-Lens, R-Lens and J++ Lens on the 500 WorkspaceBench items
     (hit = a WorkspaceBench target among the top 10 tokens), at layer 42 and at any of the
     paper's seven layers (8, 16, ..., 56).
  2. Lens as a check on NLA claims: for each graded claim about internal computation, the rank of
     the word(s) the claim itself says the model is computing or about to produce (extracted by the
     Gemini judge; claims naming none are left out) in each lens's top 50 at the exact layer-42
     vector the NLA read.
     AUROC correct vs wrong, overall and among claims the NLA repeats in >= 7 of 8 samples, alone
     and added to the sample + preceding-token signal. Multi-digit numbers are not single tokens
     in Qwen, so claims naming them are left out.
  3. Steering: on the one-digit operand pairs, whether each lens ranks the swapped-in digit above
     the original one, per condition.

    workspace-bench/.venv/bin/python scripts/jpp_eval.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import claim_judge as cj  # noqa: E402
from nla_verify_eval import FAMS, auroc, jl  # noqa: E402
from t12_eval import T, V, labels  # noqa: E402

R = Path("results/jpp")
LENSES = ["jlens", "rlens", "jpp"]
PAPER_LAYERS = ["8", "16", "24", "32", "40", "48", "56"]


def norm(t: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", t.strip().lower())


def match(tok: str, target: str) -> bool:
    t, g = norm(tok), norm(target)
    if not t or not g:
        return False
    if t == g:
        return True
    if g.isdigit() or t.isdigit():
        return False
    first = g.split()[0]
    return (len(t) >= 3 and (first.startswith(t) or g.startswith(t))) or (len(first) >= 3 and t.startswith(first))


def rank(tokens: list[str], target: str, k: int = 50):
    for i, tok in enumerate(tokens[:k]):
        if match(tok, target):
            return i
    return None


def replication(ws: list[dict]) -> None:
    src = {f: cj.sources(f) for f in FAMS}
    print("## 1. Recall@10 on the 500 WorkspaceBench items\n")
    print("| lens | " + " | ".join(FAMS) + " | mean | layer 42 only (mean) |\n|---|" + "---|" * (len(FAMS) + 2))
    for lens in LENSES:
        cells, at42 = [], []
        for f in FAMS:
            rs = [r for r in ws if r["fam"] == f]
            hit = lambda r, Ls: any(rank(r["layers"][L][lens], t, 10) is not None for L in Ls for t in src[f][r["id"]]["targets"])
            cells.append(sum(hit(r, PAPER_LAYERS) for r in rs) / len(rs))
            at42.append(sum(hit(r, ["42"]) for r in rs) / len(rs))
        print(f"| {lens} | " + " | ".join(f"{c:.0%}" for c in cells) + f" | {sum(cells) / len(cells):.0%} | {sum(at42) / len(at42):.0%} |")
    print("\n(any of the paper's seven layers; tokens with no letter or digit filtered for every lens)")


ENT = """You are given one CLAIM an interpretability tool made about what a language model is computing or
about to produce. List the specific words, names or numbers the claim says the model will output or is
internally computing (e.g. "likely '1'" -> ["1"]; "an element such as Oxygen or Argon" -> ["Oxygen",
"Argon"]). Leave out words merely quoted from the input text and generic categories ("a number", "a
noun", "a new line"). Return JSON {"entities": [...]}, empty if the claim names none."""


def entities(lab: dict) -> dict:
    from nla_verify_eval import FAMS as _F  # noqa: F401
    from t12_eval import ask, claims_all, run_cached

    cl = claims_all()
    sch = {"type": "object", "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
           "required": ["entities"], "additionalProperties": False}
    jobs = {}
    for k, v in lab.items():
        if v in ("correct", "wrong"):
            f, i, n = k.split("|")
            if (f, i) in cl:
                jobs[k] = f"CLAIM: {cl[(f, i)][int(n)]['claim']}"
    res = run_cached(R / "claim_entities.jsonl", jobs, lambda c, u: ask(c, ENT, u, sch, 1500))
    return {k: v["entities"] for k, v in res.items()}


def claims(ws: list[dict]) -> None:
    vec = {(r["fam"], r["id"]): r["nla_vec_42"] for r in ws if "nla_vec_42" in r}
    lab = labels()
    ents = entities(lab)
    sup = {d["key"]: d["support"] for d in jl(V / "support.jsonl")} | {d["key"]: d["r"]["support"] for d in jl(T / "support_new.jsonl")}
    adj = {d["key"]: d["r"]["support"] for d in jl(T / "support_adj.jsonl")}
    recs, skipped = [], 0
    for k, v in lab.items():
        if v not in ("correct", "wrong") or k not in sup or k not in adj:
            continue
        f, i, _ = k.split("|")
        es = [e for e in ents.get(k, []) if norm(e) and not (norm(e).isdigit() and len(norm(e)) > 1)]
        if (f, i) not in vec or not es:
            skipped += 1
            continue
        rec = {"key": k, "ws": v, "s": sup[k], "a": adj[k], "answer": " / ".join(es)}
        for lens in LENSES:
            rks = [rk for e in es if (rk := rank(vec[(f, i)][lens], e)) is not None]
            rec[lens] = 0.0 if not rks else 1 - min(rks) / 50
        recs.append(rec)
    nc, nw = sum(r["ws"] == "correct" for r in recs), sum(r["ws"] == "wrong" for r in recs)
    cons = [r for r in recs if r["s"] >= 7]
    print(f"\n## 2. Lens readouts as a check on the NLA's internal claims ({nc} correct, {nw} wrong; "
          f"{skipped} claims left out: they name no specific word, or only multi-digit numbers)\n")
    print("| signal | AUROC all | AUROC among >=7/8-consistent "
          f"({sum(r['ws'] == 'correct' for r in cons)}+{sum(r['ws'] == 'wrong' for r in cons)}) | answer in top 50: correct / wrong |")
    print("|---|---|---|---|")
    sigs = {"samples/8 + adjacent/3": lambda r: r["s"] / 8 + r["a"] / 3}
    for lens in LENSES:
        sigs[f"{lens} rank alone"] = (lambda L: lambda r: r[L])(lens)
        sigs[f"samples + adjacent + {lens}"] = (lambda L: lambda r: r["s"] / 8 + r["a"] / 3 + r[L])(lens)
    for name, g in sigs.items():
        c = [g(r) for r in recs if r["ws"] == "correct"]
        w = [g(r) for r in recs if r["ws"] == "wrong"]
        cc = [g(r) for r in cons if r["ws"] == "correct"]
        cw = [g(r) for r in cons if r["ws"] == "wrong"]
        lens = name.split()[0] if name.split()[0] in LENSES else None
        found = f"{sum(r[lens] > 0 for r in recs if r['ws'] == 'correct') / nc:.0%} / {sum(r[lens] > 0 for r in recs if r['ws'] == 'wrong') / nw:.0%}" if lens else ""
        print(f"| {name} | {auroc(c, w):.2f} | {auroc(cc, cw):.2f} | {found} |")
    print("\nConsistent wrong claims and whether each lens shows what the claim names (rank score, 0 = absent):")
    for r in sorted(cons, key=lambda r: r["ws"]):
        if r["ws"] == "wrong":
            print(f"  jpp {r['jpp']:.2f} jlens {r['jlens']:.2f} rlens {r['rlens']:.2f}  '{r['answer']}'  {r['key']}")


def steering() -> None:
    rows = {r["id"]: r for r in json.loads((Path("results/steer") / "pairs.json").read_text())}
    reads = {d["key"]: d for d in jl(R / "steer.jsonl")}
    print("\n## 3. Steering, one-digit second operand (100 pairs): share where the lens ranks the swapped-in digit above the original\n")
    print("| condition | " + " | ".join(LENSES) + " |\n|---|" + "---|" * len(LENSES))
    for c in ["base", "src", "swap_mean", "swap_lr", "swap_rand"]:
        cells = []
        for lens in LENSES:
            n = win = 0
            for pid, r in rows.items():
                if r["var"] != "b1" or f"{pid}|{c}" not in reads:
                    continue
                toks = reads[f"{pid}|{c}"][lens]
                rs, rb = rank(toks, str(r["src"][1])), rank(toks, str(r["base"][1]))
                n += 1
                win += (rs if rs is not None else 99) < (rb if rb is not None else 99)
            cells.append(f"{win / max(n, 1):.0%}")
        print(f"| {c} | " + " | ".join(cells) + " |")


def main() -> None:
    ws = jl(R / "ws.jsonl")
    print(f"items: {len(ws)}; mean cosine(recaptured, stored NLA vector) at layer 42: "
          f"{sum(r.get('cos_42', 0) for r in ws) / max(len(ws), 1):.4f}\n")
    replication(ws)
    claims(ws)
    steering()


if __name__ == "__main__":
    main()
