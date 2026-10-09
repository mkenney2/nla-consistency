"""Pod side of the NLA hallucination-filter tests (self-consistency + reconstructor checks).

    python scripts/nla_verify_pod.py sample   # 8 fresh NLA samples per WorkspaceBench item
    python scripts/nla_verify_pod.py ar       # reconstructor scores for every text variant

Inputs (copied from the laptop) under OUT/verify/inputs/: readouts/<fam>.jsonl (the original
NLA readouts the claims were extracted from), claims/<fam>.jsonl (labelled claims), items.json
(the 200 (family, id) pairs), jokes.json (pair id -> target, kind). Vectors: OUT/vecs/<fam>.pt.

The reconstructor (EasyNLA's critic): the raw template "Summary of the following text:
<text>{explanation}</text> <summary>" (no chat template), a 43-block truncated Qwen3.6 with
its final norm removed, the hidden state at the last token normalised to sqrt(d), then the fp32
value head. Reward in training was -MSE between the normalised prediction and the normalised
activation, i.e. a monotone function of their cosine, so we report cosine. Each text is also
scored against every other item's vector in the same set, giving a rank ("does this text point
at THIS activation, or at activations in general?").
"""
import json
import math
import re
import sys
from pathlib import Path

import torch

OUT = Path("/workspace/model-internals-mini")
V = OUT / "verify"
FAMS = ["multihop", "poetry", "typo", "association", "basic_readout"]
K = 8
CRITIC = "Summary of the following text: <text>{explanation}</text> <summary>"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def items() -> list[tuple[str, str]]:
    return [tuple(x) for x in json.loads((V / "inputs" / "items.json").read_text())]


def vec_of(fam: str) -> dict[str, torch.Tensor]:
    """id -> the layer-42 vector at the cell the original NLA readout was taken from."""
    pos = {r["id"]: r["pos"] for r in jl(V / "inputs" / "readouts" / f"{fam}.jsonl")}
    return {r["id"]: r["h"].float() for r in torch.load(OUT / "vecs" / f"{fam}.pt") if pos.get(r["id"]) == r["pos"]}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


# ---------------------------------------------------------------- 1. fresh samples

def sample() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from deception_pod import load_nla, nla_batch  # the batched NLA reader

    m, state = load_nla()
    path = V / "samples.jsonl"
    done = {(d["fam"], d["id"]) for d in jl(path)} if path.exists() else set()
    todo = [(f, i) for f, i in items() if (f, i) not in done]
    vecs = {f: vec_of(f) for f in FAMS}
    with path.open("a") as fh:
        for s in range(0, len(todo), 4):  # 4 items x 8 samples = batches of 32
            chunk = todo[s:s + 4]
            Vb = torch.stack([vecs[f][i] for f, i in chunk for _ in range(K)])
            texts = nla_batch(m, state, Vb)
            for j, (f, i) in enumerate(chunk):
                fh.write(json.dumps({"fam": f, "id": i, "samples": texts[j * K:(j + 1) * K]}, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"[sample] {len(todo)} items x {K}", flush=True)


# ---------------------------------------------------------------- 2. reconstructor

def load_ar():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = snapshot_download("ceselder/qwen3.6-27b-nla-rl", allow_patterns=["ar_reconstructor/*", "av_base/tokenizer*",
                                                                             "av_base/*.json", "av_base/*.jinja"])
    ar = f"{root}/ar_reconstructor"
    tok = AutoTokenizer.from_pretrained(f"{root}/av_base")
    model = AutoModelForCausalLM.from_pretrained(ar, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    inner = model.model if hasattr(model, "model") else model
    inner = getattr(inner, "language_model", inner)
    assert hasattr(inner, "norm"), type(inner)
    inner.norm = torch.nn.Identity()  # the value head reads the raw layer-42 residual
    n_layers = len(inner.layers)
    head = torch.nn.Linear(5120, 5120, bias=False).cuda().float()
    head.load_state_dict(load_file(f"{ar}/value_head.safetensors"))
    print(f"[ar] {n_layers} blocks, head loaded", flush=True)
    return tok, inner, head


@torch.no_grad()
def predict(tok, inner, head, texts: list[str], bs: int = 32) -> torch.Tensor:
    """[N, d] predicted activations (unit-normalised)."""
    out = []
    pad = tok.eos_token_id
    for s in range(0, len(texts), bs):
        ids = [tok.encode(CRITIC.format(explanation=t), add_special_tokens=False)[:1024] for t in texts[s:s + bs]]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), pad, dtype=torch.long, device="cuda")
        am = torch.zeros_like(x)
        for r, row in enumerate(ids):
            x[r, :len(row)] = torch.tensor(row, device="cuda")
            am[r, :len(row)] = 1
        h = inner(input_ids=x, attention_mask=am).last_hidden_state
        last = h[torch.arange(len(ids), device="cuda"), am.sum(1) - 1].float()
        last = last / last.norm(dim=-1, keepdim=True) * math.sqrt(5120)
        p = head(last)
        out.append((p / p.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


def ar() -> None:
    tok, inner, head = load_ar()
    jobs = []  # (set, key, item index, text)

    # WorkspaceBench items: one matrix of gold vectors over all 200
    its = items()
    vecs = {f: vec_of(f) for f in FAMS}
    H = torch.stack([vecs[f][i] for f, i in its])
    idx = {it: n for n, it in enumerate(its)}
    orig = {(f, r["id"]): r["samples"][0] for f in FAMS for r in jl(V / "inputs" / "readouts" / f"{f}.jsonl")}
    claims = {(f, r["id"]): r["claims"] for f in FAMS for r in jl(V / "inputs" / "claims" / f"{f}.jsonl")}
    samples = {(d["fam"], d["id"]): d["samples"] for d in jl(V / "samples.jsonl")}
    splits = {}
    for it in its:
        f, i = it
        t = orig[it]
        sents = sentences(t)
        splits[f"{f}|{i}"] = sents
        jobs.append(("ws", f"{f}|{i}|orig", idx[it], t))
        for j in range(len(sents)):
            jobs.append(("ws", f"{f}|{i}|drop{j}", idx[it], " ".join(sents[:j] + sents[j + 1:])))
        for n, c in enumerate(claims.get(it, [])):
            jobs.append(("ws", f"{f}|{i}|claim{n}", idx[it], c["claim"]))
        for k, s in enumerate(samples.get(it, [])):
            jobs.append(("ws", f"{f}|{i}|sc{k}", idx[it], s))
    (V / "splits.json").write_text(json.dumps(splits, ensure_ascii=False))

    # jokes: the NLA readout with the right vs a wrong pun appended, and a fixed template
    jk = json.loads((V / "inputs" / "jokes.json").read_text())  # pair -> {target, kind}
    jrows = {r["id"].split("-")[-1]: r for r in jl(V / "inputs" / "readouts" / "jokes.jsonl")}
    jvec = {r["id"].split("-")[-1]: r["h"].float() for r in torch.load(OUT / "vecs" / "jokes.pt")}
    pairs = [p for p in jk if p in jrows and p in jvec]
    HJ = torch.stack([jvec[p] for p in pairs])
    by_kind = {}
    for p in pairs:
        by_kind.setdefault(jk[p]["kind"], []).append(p)
    for n, p in enumerate(pairs):
        same = by_kind[jk[p]["kind"]]
        wrong = jk[same[(same.index(p) + 1) % len(same)]]["target"]
        right = jk[p]["target"]
        base = jrows[p]["samples"][0]
        for tag, tgt in (("right", right), ("wrong", wrong)):
            jobs.append(("jokes", f"{p}|nla+{tag}", n, f"{base} The joke's hidden point is: {tgt}."))
            jobs.append(("jokes", f"{p}|tmpl+{tag}", n, f"A short joke. Its hidden point, never stated in the text, is: {tgt}."))
        jobs.append(("jokes", f"{p}|nla", n, base))

    print(f"[ar] {len(jobs)} texts", flush=True)
    P = predict(tok, inner, head, [j[3] for j in jobs])
    gold = {"ws": H / H.norm(dim=-1, keepdim=True), "jokes": HJ / HJ.norm(dim=-1, keepdim=True)}
    with (V / "ar_scores.jsonl").open("w") as fh:
        for (st, k, n, _t), p in zip(jobs, P, strict=True):
            sims = gold[st] @ p  # cosine to every vector in the set
            own = sims[n].item()
            rank = (sims > sims[n]).sum().item()  # 0 = own vector is the best match
            fh.write(json.dumps({"set": st, "key": k, "cos": own, "rank": rank, "n": len(sims),
                                 "z": ((sims[n] - sims.mean()) / sims.std()).item()}) + "\n")
    print("[ar] scores written", flush=True)


def ckpts() -> None:
    """One batched NLA readout per WorkspaceBench item for each RL checkpoint (and the warm
    start), in the readouts format -> OUT/readouts/nla_<tag>_ws/<fam>.jsonl."""
    import gc
    sys.path.insert(0, str(Path(__file__).parent))
    from deception_pod import load_nla, nla_batch

    its = items()
    vecs = {f: vec_of(f) for f in FAMS}
    meta = {(f, r["id"]): r for f in FAMS for r in jl(V / "inputs" / "readouts" / f"{f}.jsonl")}
    tags = [("sft", "none")] + [(f"{s}", f"av_rl_adapters/iter_{s:06d}") for s in (100, 200, 300, 500, 600, 700, 800)]
    for tag, adapter in tags:
        d = OUT / "readouts" / f"nla_{tag}_ws"
        if (d / "done").exists():
            continue
        m, state = load_nla(adapter)
        texts = []
        for s in range(0, len(its), 32):
            chunk = its[s:s + 32]
            texts += nla_batch(m, state, torch.stack([vecs[f][i] for f, i in chunk]))
        d.mkdir(parents=True, exist_ok=True)
        for f in FAMS:
            rows = [{**{k: meta[it][k] for k in ("id", "layer", "pos", "token")}, "samples": [t]}
                    for it, t in zip(its, texts) if it[0] == f]
            (d / f"{f}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (d / "done").touch()
        print(f"[ckpt] {tag}: {len(texts)}", flush=True)
        del m, state
        gc.collect()
        torch.cuda.empty_cache()


def score() -> None:
    """Reconstructor scores for arbitrary texts (inputs/cf.jsonl: key, fam, id, text) against the
    200 WorkspaceBench vectors -> cf_scores.jsonl."""
    tok, inner, head = load_ar()
    its = items()
    vecs = {f: vec_of(f) for f in FAMS}
    H = torch.stack([vecs[f][i] for f, i in its])
    H = H / H.norm(dim=-1, keepdim=True)
    idx = {it: n for n, it in enumerate(its)}
    rows = jl(V / "inputs" / "cf.jsonl")
    P = predict(tok, inner, head, [r["text"] for r in rows])
    with (V / "cf_scores.jsonl").open("w") as fh:
        for r, p in zip(rows, P, strict=True):
            sims = H @ p
            n = idx[(r["fam"], r["id"])]
            fh.write(json.dumps({"key": r["key"], "cos": sims[n].item(), "rank": (sims > sims[n]).sum().item(),
                                 "z": ((sims[n] - sims.mean()) / sims.std()).item()}) + "\n")
    print(f"[score] {len(rows)} texts", flush=True)


def smoke() -> None:
    """Load the reconstructor; do original readouts point at their own activation?"""
    tok, inner, head = load_ar()
    its = items()[:40]
    vecs = {f: vec_of(f) for f in FAMS}
    H = torch.stack([vecs[f][i] for f, i in its])
    H = H / H.norm(dim=-1, keepdim=True)
    orig = {(f, r["id"]): r["samples"][0] for f in FAMS for r in jl(V / "inputs" / "readouts" / f"{f}.jsonl")}
    P = predict(tok, inner, head, [orig[it] for it in its])
    S = P @ H.T
    own = S.diag()
    ranks = (S > own[:, None]).sum(1).float()
    print(f"[smoke] own cos {own.mean():.3f}, others {(S.sum(1) - own).mean() / (len(its) - 1):.3f}, "
          f"mean rank {ranks.mean():.1f}/{len(its)}, top-1 {(ranks == 0).float().mean():.2f}", flush=True)


if __name__ == "__main__":
    {"sample": sample, "ar": ar, "smoke": smoke, "score": score, "ckpts": ckpts}[sys.argv[1]]()
