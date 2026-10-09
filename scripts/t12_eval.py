"""Laptop side of threads 1 and 2 (inputs from scripts/thread12_pod.py in results/t12/).

  judge   : (a) 8-sample support for every claim of the 300 newly sampled items, (b) support from
            the 3 adjacent-token readouts for every claim of all 500 items, (c) prior consistency:
            for each control vector, how far its 8 samples agree, (d) correctness of the new
            workspace claims against WorkspaceBench targets.
  report  : thread 1 tables (all 500 items; samples vs adjacent tokens vs both) and thread 2
            (arithmetic: model, probes, NLA).

    workspace-bench/.venv/bin/python scripts/t12_eval.py judge|report
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import claim_judge as cj  # noqa: E402
from nla_verify_eval import FAMS, auroc, jl, sentences  # noqa: E402
from ws_correct import SCHEMA as WS_SCHEMA, SYS as WS_SYS  # noqa: E402

GEMINI = "google/gemini-3.8-flash"
T, V = Path("results/t12"), Path("results/verify")
FALSE, TRUE = ("misquote", "contradicted"), ("workspace", "background_true")

SUPPORT = """You compare one CLAIM, taken from an interpretability tool's description of a language
model's internal state, against {n} OTHER descriptions the same tool produced ({what}).
support: how many of the {n} other descriptions state or clearly imply the same thing as the claim
(the same specific content, not merely the same topic)? An integer 0-{n}.
Return JSON {{"support": int}}."""
PRIOR = """Below are 8 descriptions an interpretability tool produced from the SAME activation vector.
Split description 1 into its specific atomic claims (at most 10; skip generic filler), and for each
count how many of descriptions 2-8 state or clearly imply the same specific content (0-7).
Return JSON {"claims": [{"claim": "...", "support": int}]}."""


def client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120, max_retries=0)


def ask(c, system, user, schema, max_tokens=1500):
    for attempt in range(6):
        try:
            r = c.chat.completions.create(
                model=GEMINI, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": schema}},
                extra_body={"reasoning": {"effort": "minimal"}})
            return json.loads(r.choices[0].message.content)
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
    return None


def run_cached(path: Path, jobs: dict, fn) -> dict:
    cache = {d["key"]: d["r"] for d in jl(path)}
    todo = {k: v for k, v in jobs.items() if k not in cache}
    print(f"{path.name}: {len(todo)} to run", flush=True)
    c = client()
    with ThreadPoolExecutor(32) as ex, path.open("a") as fh:
        futs = {ex.submit(fn, c, v): k for k, v in todo.items()}
        for f in as_completed(futs):
            if (r := f.result()) is not None:
                cache[futs[f]] = r
                fh.write(json.dumps({"key": futs[f], "r": r}, ensure_ascii=False) + "\n")
                fh.flush()
    return cache


def nla_rows() -> dict:
    return {d["key"]: d["text"] for d in jl(T / "nla.jsonl")}


def all_samples() -> dict:
    s = {(d["fam"], d["id"]): d["samples"] for d in jl(V / "samples.jsonl")}
    tmp = defaultdict(dict)
    for k, t in nla_rows().items():
        if k.startswith("sc|"):
            _, f, i, n = k.split("|")
            tmp[(f, i)][int(n)] = t
    for it, d in tmp.items():
        s[it] = [d[n] for n in sorted(d)]
    return s


def claims_all() -> dict:
    return {(f, r["id"]): r["claims"] for f in FAMS for r in jl(Path("results/claims/nla") / f"{f}.jsonl")}


def judge() -> None:
    nla = nla_rows()
    samples, claims = all_samples(), claims_all()
    sch = {"type": "object", "properties": {"support": {"type": "integer"}}, "required": ["support"], "additionalProperties": False}
    # (a) sample support for claims of the newly sampled items (the first 200 are in support.jsonl)
    old = {d["key"] for d in jl(V / "support.jsonl")}
    jobs = {}
    for it, cl in claims.items():
        if it not in samples:
            continue
        others = "\n\n".join(f"<description {k + 1}>\n{s}\n</description {k + 1}>" for k, s in enumerate(samples[it]))
        for n, c in enumerate(cl):
            key = f"{it[0]}|{it[1]}|{n}"
            if key not in old:
                jobs[key] = f"CLAIM: {c['claim']}\n\nOTHER DESCRIPTIONS:\n{others}"
    sys_s = SUPPORT.format(n=8, what="8 more samples from the same activation")
    run_cached(T / "support_new.jsonl", jobs, lambda c, u: ask(c, sys_s, u, sch))
    # (b) adjacent-token support for every claim
    adj = defaultdict(dict)
    for k, t in nla.items():
        if k.startswith("adj|"):
            _, f, i, d = k.split("|")
            adj[(f, i)][int(d)] = t
    jobs = {}
    for it, cl in claims.items():
        if len(adj.get(it, {})) < 3:
            continue
        others = "\n\n".join(f"<description {d}>\n{adj[it][d]}\n</description {d}>" for d in (1, 2, 3))
        for n, c in enumerate(cl):
            jobs[f"{it[0]}|{it[1]}|{n}"] = f"CLAIM: {c['claim']}\n\nOTHER DESCRIPTIONS:\n{others}"
    sys_a = SUPPORT.format(n=3, what="from the activations at the 3 tokens just before")
    run_cached(T / "support_adj.jsonl", jobs, lambda c, u: ask(c, sys_a, u, sch))
    # (c) prior consistency on the controls, and the same measure on real items for comparison
    ctrl = defaultdict(dict)
    for k, t in nla.items():
        if k.startswith("ctrl|"):
            _, kind, j, n = k.split("|")
            ctrl[f"{kind}|{j}"][int(n)] = t
    real = random.Random(0).sample(sorted(samples), 40)
    for it in real:
        ctrl[f"real|{it[0]}|{it[1]}"] = dict(enumerate(samples[it]))
    psch = {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "support": {"type": "integer"}}, "required": ["claim", "support"], "additionalProperties": False}}},
        "required": ["claims"], "additionalProperties": False}
    jobs = {k: "\n\n".join(f"<description {n + 1}>\n{d[n]}\n</description {n + 1}>" for n in sorted(d)[:8])
            for k, d in ctrl.items() if len(d) >= 8}
    run_cached(T / "prior.jsonl", jobs, lambda c, u: ask(c, PRIOR, u, psch, 3000))
    # (d) correctness of workspace claims of all items not graded before
    src = {f: cj.sources(f) for f in FAMS}
    old = {d["key"] for d in jl(V / "ws_correct_v2.jsonl")}
    jobs = {}
    for (f, i), cl in claims.items():
        for n, c in enumerate(cl):
            k = f"{f}|{i}|{n}"
            if c["label"] == "workspace" and k not in old:
                s = src[f][i]
                jobs[k] = (f"SOURCE TEXT:\n<<<{s['text']}>>>\n\nWORKSPACE TARGET(S): {json.dumps(s['targets'], ensure_ascii=False)}"
                           f"\n\nCLAIM: {c['claim']}")
    run_cached(T / "ws_correct_new.jsonl", jobs, lambda c, u: ask(c, WS_SYS, u, WS_SCHEMA, 1000))


def labels() -> dict:
    """workspace-claim key -> correct / wrong / vague (old hand-checked labels + new grades)."""
    lab = json.load(open(V / "ws_correct_final.json"))
    hand = json.load(open(T / "hand_not_wrong.json")) if (T / "hand_not_wrong.json").exists() else []
    for d in jl(T / "ws_correct_new.jsonl"):
        v = d["r"]["verdict"]
        lab.setdefault(d["key"], "excluded" if d["key"] in hand else ("correct" if v.startswith("correct") else v))
    return lab


def report() -> None:
    claims = claims_all()
    sup = {d["key"]: d["support"] for d in jl(V / "support.jsonl")} | {d["key"]: d["r"]["support"] for d in jl(T / "support_new.jsonl")}
    adj = {d["key"]: d["r"]["support"] for d in jl(T / "support_adj.jsonl")}
    lab = labels()
    recs = []
    for (f, i), cl in claims.items():
        for n, c in enumerate(cl):
            k = f"{f}|{i}|{n}"
            recs.append({"key": k, "fam": f, "label": c["label"], "ws": lab.get(k), "s": sup.get(k), "a": adj.get(k)})
    print(f"## Thread 1: all {len({(r['fam'], r['key'].split('|')[1]) for r in recs})} items, {len(recs)} claims\n")
    print("| signal | input claims: AUROC true vs false | internal claims: AUROC correct vs wrong |\n|---|---|---|")
    sig = {"8 samples": lambda r: r["s"], "3 adjacent tokens": lambda r: r["a"],
           "both (samples/8 + adjacent/3)": lambda r: None if r["s"] is None or r["a"] is None else r["s"] / 8 + r["a"] / 3}
    for name, g in sig.items():
        t = [g(r) for r in recs if r["label"] in TRUE and g(r) is not None]
        fl = [g(r) for r in recs if r["label"] in FALSE and g(r) is not None]
        c = [g(r) for r in recs if r["ws"] == "correct" and g(r) is not None]
        w = [g(r) for r in recs if r["ws"] == "wrong" and g(r) is not None]
        print(f"| {name} | {auroc(t, fl):.2f} (n={len(t) + len(fl)}) | {auroc(c, w):.2f} (n={len(c)}+{len(w)}) |")
    wrong = [r for r in recs if r["ws"] == "wrong" and r["s"] is not None]
    cons = [r for r in wrong if r["s"] >= 7]
    print(f"\nwrong internal claims: {len(wrong)}; consistent across samples (>=7/8): {len(cons)}")
    if cons:
        print(f"  of the consistent ones, adjacent support 0: {sum(r['a'] == 0 for r in cons if r['a'] is not None)}, "
              f">=2 of 3: {sum((r['a'] or 0) >= 2 for r in cons)}")
    print("\n| filter on internal claims | wrong among kept | correct kept | wrong removed |\n|---|---|---|---|")
    ws = [r for r in recs if r["ws"] in ("correct", "wrong") and r["s"] is not None and r["a"] is not None]
    nc, nw = sum(r["ws"] == "correct" for r in ws), sum(r["ws"] == "wrong" for r in ws)
    for name, keep in {"none": lambda r: True, "samples >= 4/8": lambda r: r["s"] >= 4,
                       "adjacent >= 1/3": lambda r: r["a"] >= 1, "samples >= 4/8 AND adjacent >= 1/3": lambda r: r["s"] >= 4 and r["a"] >= 1,
                       "samples >= 6/8 AND adjacent >= 2/3": lambda r: r["s"] >= 6 and r["a"] >= 2}.items():
        kept = [r for r in ws if keep(r)]
        kw = sum(r["ws"] == "wrong" for r in kept)
        print(f"| {name} | {100 * kw / max(len(kept), 1):.0f}% | {100 * (len(kept) - kw) / nc:.0f}% | {100 * (1 - kw / nw):.0f}% |")
    # prior consistency
    pr = {d["key"]: d["r"]["claims"] for d in jl(T / "prior.jsonl")}
    print("\n## How consistent is the NLA with nothing to read? (mean support of sample 1's claims among 7 others)\n")
    for kind in ("mean", "noise", "real"):
        vals = [c["support"] for k, cl in pr.items() if k.startswith(kind) for c in cl]
        if vals:
            print(f"{kind}: {sum(vals) / len(vals):.2f} of 7 (n={len(vals)} claims); share at >=6/7: {100 * sum(v >= 6 for v in vals) / len(vals):.0f}%")

    # thread 2
    ar = {r["id"]: r for r in json.load(open(T / "arith_meta.json"))}
    pp = {r["id"]: r for r in json.load(open(T / "probe_preds.json"))}
    texts = defaultdict(list)
    for k, t in nla_rows().items():
        if k.startswith("ar|"):
            texts[k.split("|")[1]].append(t)
    expr = re.compile(r"(\d+)\s*([-−+])\s*(\d+)")
    res = Counter()
    for i, r in ar.items():
        a_ok = (pp[i]["probe_a_tens"] == r["a"] // 10) and (pp[i]["probe_a_units"] == r["a"] % 10)
        b_ok = (pp[i]["probe_b_tens"] == r["b"] // 10) and (pp[i]["probe_b_units"] == r["b"] % 10)
        for t in texts[i]:
            m = expr.search(t)
            res["readouts"] += 1
            res["probe_a"] += a_ok
            res["probe_b"] += b_ok
            if m:
                res["nla_expr"] += 1
                res["nla_a"] += int(m.group(1)) == r["a"]
                res["nla_b"] += int(m.group(3)) == r["b"]
                res["both_wrong_probe_right"] += (int(m.group(1)) != r["a"]) and a_ok
        res["model_right"] += str(r["answer"]).startswith(r["model_next"].strip()) and r["model_next"].strip() != ""
    n = len(ar)
    print(f"\n## Thread 2: arithmetic ({n} prompts, {res['readouts']} NLA readouts)\n")
    print(f"model's greedy next token starts the right answer: {100 * res['model_right'] / n:.0f}%")
    print(f"probe decodes a exactly (both digits): {100 * res['probe_a'] / res['readouts']:.0f}%; b exactly: {100 * res['probe_b'] / res['readouts']:.0f}%")
    print(f"NLA quotes an 'x op y' expression in {100 * res['nla_expr'] / res['readouts']:.0f}% of readouts; of those, a right: "
          f"{100 * res['nla_a'] / max(res['nla_expr'], 1):.0f}%, b right: {100 * res['nla_b'] / max(res['nla_expr'], 1):.0f}%")
    print(f"NLA gets a wrong although the probe decodes it: {100 * res['both_wrong_probe_right'] / max(res['nla_expr'], 1):.0f}% of quoted expressions")


if __name__ == "__main__":
    {"judge": judge, "report": report}[sys.argv[1]]()
