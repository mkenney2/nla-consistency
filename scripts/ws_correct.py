"""Is each NLA claim about the workspace RIGHT? Re-labels the step-400 claims judged "workspace"
(claims about what the model is computing or about to write) against the item's known target,
then asks whether agreement across the 8 fresh samples predicts correctness.

    workspace-bench/.venv/bin/python scripts/ws_correct.py label    # -> results/verify/ws_correct.jsonl
    workspace-bench/.venv/bin/python scripts/ws_correct.py report
"""
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import claim_judge as cj  # noqa: E402
from nla_verify_eval import FAMS, V, auroc, jl, load_inputs  # noqa: E402

GEMINI = "google/gemini-3.8-flash"
OUT = V / "ws_correct_v2.jsonl"
SYS = """A language model read the SOURCE TEXT. At its last token it holds the WORKSPACE TARGET(S): an
intermediate entity, planned rhyme, corrected word or concept (for multi-step questions the target is
often the INTERMEDIATE step, not the final answer). An interpretability tool made the CLAIM below about
what the model is computing or about to produce. First work out yourself what the correct final answer
or natural continuation of the source is. Then grade the claim:
- "correct_target": it names or predicts the workspace target (or an accepted variant).
- "correct_other": it is right about something else the model computes: the correct final answer, a
  correct intermediate other than the target, or a continuation that genuinely fits the text.
- "wrong": it commits to something false: a wrong answer, wrong entity, wrong rhyme, wrong arithmetic,
  or a continuation that does not fit the text.
- "vague": it commits to nothing specific enough to be right or wrong.
Return JSON {"verdict": "correct_target"|"correct_other"|"wrong"|"vague", "answer": "<the correct final answer or continuation you worked out>"}."""
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["correct_target", "correct_other", "wrong", "vague"]},
                                          "answer": {"type": "string"}},
          "required": ["verdict", "answer"], "additionalProperties": False}


def label() -> None:
    its, _orig, claims, _s = load_inputs()
    src = {f: cj.sources(f) for f in FAMS}
    done = {d["key"] for d in jl(OUT)}
    c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=90, max_retries=0)
    jobs = {}
    for f, i in its:
        for n, cl in enumerate(claims.get((f, i), [])):
            k = f"{f}|{i}|{n}"
            if cl["label"] == "workspace" and k not in done:
                s = src[f][i]
                jobs[k] = (f"SOURCE TEXT:\n<<<{s['text']}>>>\n\nWORKSPACE TARGET(S): {json.dumps(s['targets'], ensure_ascii=False)}"
                           f"\n\nCLAIM: {cl['claim']}")

    def ask(user):
        for attempt in range(6):
            try:
                r = c.chat.completions.create(
                    model=GEMINI, temperature=0, max_tokens=1000,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": SCHEMA}},
                    extra_body={"reasoning": {"effort": "minimal"}})
                return json.loads(r.choices[0].message.content)
            except Exception:  # noqa: BLE001
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
        return None

    print(f"{len(jobs)} workspace claims to grade", flush=True)
    with ThreadPoolExecutor(32) as ex, OUT.open("a") as fh:
        futs = {ex.submit(ask, u): k for k, u in jobs.items()}
        for fu in as_completed(futs):
            if (v := fu.result()) is not None:
                fh.write(json.dumps({"key": futs[fu], **v}) + "\n")


def report() -> None:
    verdict = {d["key"]: ("correct" if d["verdict"].startswith("correct") else d["verdict"]) for d in jl(OUT)}
    sub = Counter(d["verdict"] for d in jl(OUT))
    print(f"fine labels: {dict(sub)}")
    sup = {d["key"]: d["support"] for d in jl(V / "support.jsonl")}
    its, _o, claims, _s = load_inputs()
    rows = [(k, verdict[k], sup.get(k), k.split("|")[0]) for k in verdict]
    print(f"{len(rows)} workspace claims: {dict(Counter(v for _, v, _, _ in rows))}\n")
    print("| family | correct | wrong | vague | support: correct | support: wrong | AUROC correct vs wrong |")
    print("|---|---|---|---|---|---|---|")
    for fam in FAMS + ["ALL"]:
        rs = [r for r in rows if fam in ("ALL", r[3]) and r[2] is not None]
        cnt = Counter(v for _, v, _, _ in rs)
        sc = [s for _, v, s, _ in rs if v == "correct"]
        sw = [s for _, v, s, _ in rs if v == "wrong"]
        m = lambda xs: f"{sum(xs) / len(xs):.1f}" if xs else "–"
        print(f"| {fam} | {cnt['correct']} | {cnt['wrong']} | {cnt['vague']} | {m(sc)} | {m(sw)} | {auroc(sc, sw):.2f} |")
    # precision of a workspace claim at each support threshold
    rs = [r for r in rows if r[2] is not None and r[1] != "vague"]
    print("\n| keep workspace claims with support >= | kept | correct among kept | correct claims kept |")
    print("|---|---|---|---|")
    ncorrect = sum(v == "correct" for _, v, _, _ in rs)
    for t in range(0, 9):
        kept = [r for r in rs if r[2] >= t]
        if kept:
            nc = sum(v == "correct" for _, v, _, _ in kept)
            print(f"| {t}/8 | {len(kept)} | {100 * nc / len(kept):.0f}% | {100 * nc / ncorrect:.0f}% |")
    # wrong claims with high support: what are they?
    print("\nHigh-support WRONG workspace claims (support >= 5):")
    src = {f: cj.sources(f) for f in FAMS}
    for k, v, s, f in rows:
        if v == "wrong" and s is not None and s >= 5:
            _f, i, n = k.split("|")
            print(f"  [{s}/8] {f}/{i}: target {src[f][i]['targets'][:3]} | claim: {claims[(f, i)][int(n)]['claim'][:140]}")


if __name__ == "__main__":
    {"label": label, "report": report}[sys.argv[1]]()
