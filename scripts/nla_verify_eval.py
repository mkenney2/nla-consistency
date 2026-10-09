"""Laptop side of the NLA hallucination-filter tests (inputs from scripts/nla_verify_pod.py).

  support : for each labelled claim of the original readout, Gemini counts how many of the 8
            fresh samples state or clearly imply it, and which sentence of the original it
            came from (for the drop-one-sentence reconstructor test).
  best    : writes the best-of-8 sample by reconstructor cosine, and a random one, as readouts
            (results/readouts/nla_best8, nla_rand8) for scripts/claim_judge.py.
  report  : AUROC of each signal for false vs true claims, filter tables in the ensemble
            format, best-of-8 vs random, and the joke diagnostic.

    workspace-bench/.venv/bin/python scripts/nla_verify_eval.py support|best|report
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

GEMINI = "google/gemini-3.8-flash"
V = Path("results/verify")
FAMS = ["multihop", "poetry", "typo", "association", "basic_readout"]
FALSE, TRUE = ("misquote", "contradicted"), ("workspace", "background_true")

SUPPORT = """You compare one CLAIM, taken from an interpretability tool's description of a language
model's internal state, against 8 OTHER descriptions the same tool produced from the same state.
Also given: the ORIGINAL description the claim came from, split into numbered sentences.
1. support: how many of the 8 other descriptions state or clearly imply the same thing as the
   claim (the same specific content, not merely the same topic)? An integer 0-8.
2. sentence: the number of the ORIGINAL sentence the claim was taken from (best match).
Return JSON {"support": int, "sentence": int}."""
SCHEMA = {"type": "object", "properties": {"support": {"type": "integer"}, "sentence": {"type": "integer"}},
          "required": ["support", "sentence"], "additionalProperties": False}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def sentences(text: str) -> list[str]:  # must match nla_verify_pod.sentences
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def load_inputs():
    its = [tuple(x) for x in json.loads((V / "inputs/items.json").read_text())]
    orig = {(f, r["id"]): r["samples"][0] for f in FAMS for r in jl(V / "inputs/readouts" / f"{f}.jsonl")}
    claims = {(f, r["id"]): r["claims"] for f in FAMS for r in jl(V / "inputs/claims" / f"{f}.jsonl")}
    samples = {(d["fam"], d["id"]): d["samples"] for d in jl(V / "samples.jsonl")}
    return its, orig, claims, samples


def support() -> None:
    its, orig, claims, samples = load_inputs()
    path = V / "support.jsonl"
    done = {d["key"] for d in jl(path)}
    c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=90, max_retries=0)
    jobs = {}
    for it in its:
        if it not in samples:
            continue
        sents = "\n".join(f"[{j}] {s}" for j, s in enumerate(sentences(orig[it])))
        others = "\n\n".join(f"<description {k + 1}>\n{s}\n</description {k + 1}>" for k, s in enumerate(samples[it]))
        for n, cl in enumerate(claims.get(it, [])):
            key = f"{it[0]}|{it[1]}|{n}"
            if key not in done:
                jobs[key] = f"CLAIM: {cl['claim']}\n\nORIGINAL (numbered sentences):\n{sents}\n\nOTHER DESCRIPTIONS:\n{others}"

    def ask(user):
        for attempt in range(6):
            try:
                r = c.chat.completions.create(
                    model=GEMINI, temperature=0, max_tokens=1500,
                    messages=[{"role": "system", "content": SUPPORT}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": SCHEMA}},
                    extra_body={"reasoning": {"effort": "minimal"}})
                return json.loads(r.choices[0].message.content)
            except Exception:  # noqa: BLE001
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
        return None

    print(f"{len(jobs)} claims to check", flush=True)
    with ThreadPoolExecutor(32) as ex, path.open("a") as fh:
        futs = {ex.submit(ask, u): k for k, u in jobs.items()}
        for f in as_completed(futs):
            if (r := f.result()) is not None:
                fh.write(json.dumps({"key": futs[f], **r}) + "\n")
                fh.flush()


def ar_scores() -> dict:
    return {(d["set"], d["key"]): d for d in jl(V / "ar_scores.jsonl")}


def best() -> None:
    its, _orig, _claims, samples = load_inputs()
    ar = ar_scores()
    rows = {f: [] for f in FAMS}
    for f in FAMS:
        for r in jl(V / "inputs/readouts" / f"{f}.jsonl"):
            it = (f, r["id"])
            if it not in samples or it not in its:
                continue
            sc = [ar[("ws", f"{f}|{r['id']}|sc{k}")]["cos"] for k in range(len(samples[it]))]
            b = max(range(len(sc)), key=sc.__getitem__)
            rows[f].append(({**r, "samples": [samples[it][b]]}, {**r, "samples": [samples[it][1]]}))
    for name, j in (("nla_best8", 0), ("nla_rand8", 1)):
        d = Path("results/readouts") / name
        d.mkdir(parents=True, exist_ok=True)
        for f in FAMS:
            (d / f"{f}.jsonl").write_text("".join(json.dumps(p[j], ensure_ascii=False) + "\n" for p in rows[f]))
    print("wrote results/readouts/nla_best8 and nla_rand8; now run claim_judge.py on them")


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report() -> None:
    its, orig, claims, samples = load_inputs()
    ar = ar_scores()
    sup = {d["key"]: d for d in jl(V / "support.jsonl")}
    recs = []
    for it in its:
        f, i = it
        base = ar.get(("ws", f"{f}|{i}|orig"))
        sents = sentences(orig[it])
        for n, cl in enumerate(claims.get(it, [])):
            k = f"{f}|{i}|{n}"
            s = sup.get(k)
            alone = ar.get(("ws", f"{f}|{i}|claim{n}"))
            j = s["sentence"] if s and 0 <= s["sentence"] < len(sents) else None
            drop = ar.get(("ws", f"{f}|{i}|drop{j}")) if j is not None else None
            recs.append({"label": cl["label"], "item": it,
                         "support": s["support"] if s else None,
                         "drop_cos": (base["cos"] - drop["cos"]) if base and drop else None,
                         "alone_cos": alone["cos"] if alone else None,
                         "alone_z": alone["z"] if alone else None,
                         "alone_rank": -alone["rank"] if alone else None})
    print(f"{len(recs)} claims; labels {dict(Counter(r['label'] for r in recs))}\n")
    print("## Does the signal separate false claims (misquote/contradicted) from true ones (workspace/background_true)?\n")
    print("| signal (higher = keep) | AUROC true vs false | n |\n|---|---|---|")
    for sig in ("support", "drop_cos", "alone_cos", "alone_z", "alone_rank"):
        t = [r[sig] for r in recs if r["label"] in TRUE and r[sig] is not None]
        fl = [r[sig] for r in recs if r["label"] in FALSE and r[sig] is not None]
        print(f"| {sig} | {auroc(t, fl):.2f} | {len(t) + len(fl)} |")
    for lab in ("workspace", "background_true", "misquote", "contradicted", "invented_content", "unverifiable"):
        v = [r["support"] for r in recs if r["label"] == lab and r["support"] is not None]
        if v:
            print(f"mean support (of 8), {lab}: {sum(v) / len(v):.2f}")

    tot = Counter(r["label"] for r in recs)
    nfalse = sum(tot[l] for l in FALSE)
    print("\n## Filters (same columns as the ensemble table)\n")
    print("| filter | claims kept | false among kept | workspace kept | true background kept | false kept | readouts with >=1 false |")
    print("|---|---|---|---|---|---|---|")
    filters = {"none": lambda r: True}
    for m in (1, 2, 3, 4):
        filters[f"self-consistency: support >= {m}/8"] = lambda r, m=m: r["support"] is not None and r["support"] >= m
    for q in (0.25, 0.5):
        vals = sorted(r["alone_z"] for r in recs if r["alone_z"] is not None)
        t = vals[int(q * len(vals))]
        filters[f"reconstructor: claim-alone z above {int(q * 100)}th pct"] = lambda r, t=t: r["alone_z"] is not None and r["alone_z"] >= t
    for name, keep in filters.items():
        kept = [r for r in recs if keep(r)]
        c = Counter(r["label"] for r in kept)
        fk = sum(c[l] for l in FALSE)
        bad = {r["item"] for r in kept if r["label"] in FALSE}
        print(f"| {name} | {len(kept)} | {100 * fk / max(len(kept), 1):.0f}% | {100 * c['workspace'] / tot['workspace']:.0f}% | "
              f"{100 * c['background_true'] / tot['background_true']:.0f}% | {100 * fk / nfalse:.0f}% | {100 * len(bad) / len(its):.0f}% |")

    # best-of-8 vs random, if the claim judge has run on them
    print("\n## Best-of-8 by reconstruction vs a random sample (claim judge labels)\n")
    for name in ("nla_rand8", "nla_best8"):
        cl = [c for f in FAMS for r in jl(Path("results/claims") / name / f"{f}.jsonl") for c in r["claims"]]
        if cl:
            c = Counter(x["label"] for x in cl)
            items_bad = sum(any(x["label"] in FALSE for x in r["claims"]) for f in FAMS for r in jl(Path("results/claims") / name / f"{f}.jsonl"))
            n_items = sum(1 for f in FAMS for _ in jl(Path("results/claims") / name / f"{f}.jsonl"))
            print(f"{name}: {len(cl)} claims, false {100 * sum(c[l] for l in FALSE) / len(cl):.0f}%, "
                  f"workspace {100 * c['workspace'] / len(cl):.0f}%, readouts with >=1 false {100 * items_bad / n_items:.0f}%")

    # jokes
    j = {k: d for (s, k), d in ar.items() if s == "jokes"}
    pairs = sorted({k.split("|")[0] for k in j})
    print(f"\n## Jokes: does the reconstructor prefer the right hidden point over a wrong one? ({len(pairs)} jokes)\n")
    for kind in ("nla", "tmpl"):
        w = [j[f"{p}|{kind}+right"]["cos"] > j[f"{p}|{kind}+wrong"]["cos"] for p in pairs]
        d = [j[f"{p}|{kind}+right"]["cos"] - j[f"{p}|{kind}+wrong"]["cos"] for p in pairs]
        print(f"{kind}+X: right > wrong in {100 * sum(w) / len(w):.0f}% of jokes; mean cos gap {sum(d) / len(d):+.4f}")
    r = [j[f"{p}|nla+right"]["cos"] - j[f"{p}|nla"]["cos"] for p in pairs]
    print(f"appending the right point to the NLA readout changes cos by {sum(r) / len(r):+.4f} on average")


if __name__ == "__main__":
    {"support": support, "best": best, "report": report}[sys.argv[1]]()
