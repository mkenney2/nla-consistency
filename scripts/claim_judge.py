"""Split prose readouts into atomic claims and label each against the source text.

    workspace-bench/.venv/bin/python scripts/claim_judge.py results/readouts results/claims \
        --methods nla,olens_cs --families multihop,poetry,typo,association,basic_readout

Labels: workspace (the item's latent / what comes next), background_true (a correct statement
about the context: language, genre, format, register, syntax, speaker, topic), misquote
(invented wording attributed to the source), contradicted (any other statement about the source
that it shows is false), invented_content (scene or continuation details the readout makes up,
not asserted about the source), unverifiable (speculation the text cannot settle).
Writes <out>/<method>/<family>.jsonl (cached by item id) and prints summary tables.
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from wsbench import readplan
from wsbench.banks import exact_targets, item_targets, load_bank
from wsbench.registry import REPO_ROOT

MODEL = "google/gemini-3.8-flash"
LABELS = ["workspace", "background_true", "misquote", "contradicted", "invented_content", "unverifiable"]

SYSTEM = """You audit readouts from an interpretability tool. A language model read the SOURCE TEXT
and the tool, looking only at the model's internal state at the READ TOKEN (the source text is cut
right after it), produced the READOUT. The WORKSPACE TARGETS are what the model is known to be
computing there (an answer, a bridge entity, a planned rhyme, a corrected word, a concept).

1. Split the readout into atomic claims (one checkable statement each; drop pure filler and
   formatting). Merge near-duplicates. At most 12 claims.
2. Label each claim:
   - "workspace": it names or predicts the workspace target, or states what the model is about
     to write next / is computing (correct or not about the target, it is about the workspace).
   - "background_true": a correct statement about the source text or its context other than the
     target: its language, genre, document type, format, register, tone, syntax/position, topic,
     speaker. Must be supported by the source text.
   - "misquote": the claim asserts the source contains specific wording (a quote or near-quote)
     that it does not contain, even if the gist is right ("the text says 'I've called three
     times'" when it says "FOUR times").
   - "contradicted": any other statement the readout makes ABOUT THE SOURCE TEXT AS WRITTEN that
     the source shows is false ("the text is in German", "the prompt starts with 'Q:'", "the
     speaker is a child", "the text mentions atomic number 29").
   - "invented_content": details of a scene, story, example or continuation the readout itself
     writes or imagines, not asserted as properties of the source (a continuation that has rain
     falling, a character named Emma in a generated sentence, an invented example quiz). Content
     the source does not contain is NOT "contradicted" unless the readout says the source has it.
   - "unverifiable": speculation about intent, audience or origin that the text cannot settle.
Return JSON {"claims": [{"claim": "...", "label": "...", "evidence": "<10 words>"}]}."""

SCHEMA = {
    "name": "claims",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "label": {"type": "string", "enum": LABELS},
                        "evidence": {"type": "string"},
                    },
                    "required": ["claim", "label", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


def sources(family: str) -> dict[str, dict]:
    """Item id -> source text and workspace targets."""
    _h, items = load_bank(REPO_ROOT / "evals" / family / "items.json")
    targets = {it["id"]: exact_targets(item_targets(it)) for it in items}
    out = {}
    for spec in readplan.plan(family):
        text = spec.text or json.dumps(spec.messages or [], ensure_ascii=False)
        out[spec.id] = {"text": text, "targets": targets.get(spec.id, [])}
    return out


def ask(client: OpenAI, src: dict, token: str, readout: str) -> list[dict] | None:
    user = (f"SOURCE TEXT:\n<<<{src['text']}>>>\n\nREAD TOKEN: {token!r} (the last token above)\n"
            f"WORKSPACE TARGETS: {json.dumps(src['targets'], ensure_ascii=False)}\n\nREADOUT:\n{readout}")
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=1500,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": SCHEMA},
                extra_body={"reasoning": {"effort": "minimal"}},
            )
            return json.loads(r.choices[0].message.content)["claims"]
        except Exception as e:  # noqa: BLE001 - retry transient failures, then give up
            err = e
    print(f"  failed: {err!r}", file=sys.stderr)
    return None


def run(readouts: Path, out: Path, methods: list[str], families: list[str]) -> dict:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                    timeout=90, max_retries=0)
    results: dict = {}
    for f in families:
        src = sources(f)
        for m in methods:
            path = readouts / m / f"{f}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            cpath = out / m / f"{f}.jsonl"
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cache = {}
            if cpath.exists():
                cache = {d["id"]: d for d in map(json.loads, cpath.read_text().splitlines())}
            todo = [r for r in rows if r["id"] not in cache and r.get("samples")]
            with ThreadPoolExecutor(24) as ex, cpath.open("a") as fh:
                for r, claims in zip(todo, ex.map(
                        lambda r: ask(client, src[r["id"]], r["token"], r["samples"][0]), todo)):
                    if claims is not None:
                        d = {"id": r["id"], "claims": claims}
                        cache[r["id"]] = d
                        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            results.setdefault(m, {})[f] = [cache[r["id"]] for r in rows if r["id"] in cache]
    return results


def report(results: dict) -> None:
    print("## Claim labels (share of all claims)\n")
    print("| method | family | readouts | claims/readout | " + " | ".join(LABELS)
          + " | readouts with >=1 misquote or contradicted |")
    print("|---|---|---|---|" + "---|" * len(LABELS) + "---|")
    for m, fams in results.items():
        allc, alln, allbad = Counter(), 0, 0
        for f, docs in [*fams.items(), ("ALL", None)]:
            if docs is None:
                c, n, bad = allc, alln, allbad
            else:
                c = Counter(cl["label"] for d in docs for cl in d["claims"])
                n = len(docs)
                bad = sum(any(cl["label"] in ("misquote", "contradicted") for cl in d["claims"]) for d in docs)
                allc += c; alln += n; allbad += bad
            tot = sum(c.values()) or 1
            cells = " | ".join(f"{100 * c[l] / tot:.0f}%" for l in LABELS)
            print(f"| {m} | {f} | {n} | {tot / max(n, 1):.1f} | {cells} | {100 * bad / max(n, 1):.0f}% |")
    print("\n## Of the claims NOT about the workspace: true background vs false vs unverifiable\n")
    rest_labels = [l for l in LABELS if l != "workspace"]
    print("| method | " + " | ".join(rest_labels) + " |\n|---|" + "---|" * len(rest_labels))
    for m, fams in results.items():
        c = Counter(cl["label"] for docs in fams.values() for d in docs for cl in d["claims"])
        rest = sum(c[l] for l in LABELS if l != "workspace") or 1
        print(f"| {m} | " + " | ".join(f"{100 * c[l] / rest:.0f}%" for l in LABELS if l != "workspace") + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("readouts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--methods", default="nla,olens_cs")
    ap.add_argument("--families", default="multihop,poetry,typo,association,basic_readout")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    res = run(a.readouts, a.out, a.methods.split(","), a.families.split(","))
    report(res)
