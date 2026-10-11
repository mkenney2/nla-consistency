"""Pod side of the J++ Lens test (Ayonrinde & Lindsey 2026, released lens koayon/jpp-lenses).

  download : fetch qwen3.6-27b/lens.pt (6.6 GB) into the volume's HF cache.
  read     : top-50 readouts of the J-Lens, R-Lens and J++ Lens
             (a) at layer 42 on the exact vectors the NLA read for the 500 WorkspaceBench items,
             (b) at the paper's seven layers (8..56) on the same items and read positions,
             (c) at layer 42 on the steering vectors (one-digit operand pairs, every condition).
J++ readout (README of the release): logits = W_U · final_norm(J_l h), tokens without a letter
or digit dropped. J-Lens and R-Lens use workspace-bench's readouts of record.

    python scripts/jpp_pod.py download
    python scripts/jpp_pod.py read
"""
import gc
import json
import re
import sys
from pathlib import Path

import torch

OUT = Path("/workspace/model-internals-mini")
J = OUT / "jpp"
FAMS = ["multihop", "poetry", "typo", "association", "basic_readout"]
LAYERS = [8, 16, 24, 32, 40, 42, 48, 56]
K = 50


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def download() -> None:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("koayon/jpp-lenses", "qwen3.6-27b/lens.pt")
    print("[download]", p, flush=True)


def read() -> None:
    from huggingface_hub import hf_hub_download
    from wsbench import readplan
    from wsbench.produce.backend import Backend
    from wsbench.produce.methods import JLens, RLens
    from wsbench.produce.render import render

    J.mkdir(parents=True, exist_ok=True)
    b = Backend.load("Qwen/Qwen3.6-27B")
    tok = b.tokenizer

    # phase 1 (model on the GPU): capture every activation the lenses will read
    pos = {(f, r["id"]): r["pos"] for f in FAMS for r in jl(OUT / "verify" / "inputs" / "readouts" / f"{f}.jsonl")}
    items = []
    for f in FAMS:
        nla_vec = {r["id"]: r["h"].float() for r in torch.load(OUT / "vecs" / f"{f}.pt") if pos.get((f, r["id"])) == r["pos"]}
        for spec in readplan.plan(f):
            if (f, spec.id) not in pos:
                continue
            r = render(spec, tok)
            read_at = pos[(f, spec.id)] % len(r)
            acts = b.capture(r.ids, LAYERS, [read_at])
            items.append({"fam": f, "id": spec.id, "token": r.decoded[read_at],
                          "acts": {L: acts[L][0].float() for L in LAYERS}, "nla": nla_vec.get(spec.id)})
    print(f"[read] captured {len(items)} workspace items", flush=True)

    # phase 2: free the model, keep only its final norm and unembedding
    norm = b.final_norm
    w_u = b.unembed.detach().clone()
    b.model = None
    gc.collect()
    torch.cuda.empty_cache()
    vocab = w_u.shape[0]
    text = [tok.decode([i]) for i in range(vocab)]
    keep = torch.tensor([bool(re.search(r"[^\W_]", t)) for t in text], device=w_u.device)
    print(f"[read] readout vocabulary {int(keep.sum())} of {vocab}", flush=True)

    obj = torch.load(hf_hub_download("koayon/jpp-lenses", "qwen3.6-27b/lens.pt"), map_location="cpu", weights_only=True)
    print(f"[read] J++ config: {obj.get('config')}", flush=True)
    jac = obj["parameters"]["jacobians"]
    jpp = {L: (jac[L] if L in jac else jac[str(L)]).float().to(w_u.device) for L in LAYERS}
    del obj, jac

    # J-Lens and R-Lens: the released artifacts workspace-bench uses, with its readouts of record
    # (J: cosine-normalised W_U J h; R: W_U norm(J h)). Each is ranked over the filtered vocabulary,
    # as in the paper's Table 1, and over the full one.
    from wsbench.produce.methods import _load_jacobians

    jl_full = _load_jacobians(JLens.repo, JLens.filename, "cpu")
    rl_full = _load_jacobians(RLens.repo, RLens.filename, "cpu")
    jlj = {L: jl_full[L].to(w_u.device) for L in LAYERS}
    rlj = {L: rl_full[L].to(w_u.device) for L in LAYERS}
    del jl_full, rl_full
    jden = {L: torch.cat([(w_u[i:i + 16384].float() @ jlj[L]).norm(dim=1) for i in range(0, vocab, 16384)]).clamp_min(1e-9)
            for L in LAYERS}

    def top(logits, filtered):
        if filtered:
            logits = logits.masked_fill(~keep, float("-inf"))
        return [text[i] for i in torch.topk(logits, K).indices.tolist()]

    def unemb(x):
        return (x.to(w_u.dtype) @ w_u.T).float()

    def read_all(h, L):
        h = h.to(w_u.device).float()
        jpp_l = unemb(norm(h @ jpp[L].T))
        j_l = unemb(h @ jlj[L].T) / jden[L]
        r_l = unemb(norm(h @ rlj[L].T))
        return {"jpp": top(jpp_l, True), "jlens": top(j_l, True), "rlens": top(r_l, True),
                "jlens_raw": top(j_l, False), "rlens_raw": top(r_l, False)}

    with torch.no_grad():
        with (J / "ws.jsonl").open("w") as fh:
            for it in items:
                rec = {"fam": it["fam"], "id": it["id"], "token": it["token"],
                       "layers": {L: read_all(it["acts"][L], L) for L in LAYERS}}
                if it["nla"] is not None:  # the exact vector the NLA read (stored half precision)
                    rec["nla_vec_42"] = read_all(it["nla"], 42)
                    rec["cos_42"] = float(torch.nn.functional.cosine_similarity(it["acts"][42], it["nla"], dim=0))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[read] workspace items {len(items)}", flush=True)

        sv = torch.load(OUT / "steer" / "vecs.pt")
        with (J / "steer.jsonl").open("w") as fh:
            for key, h in sv.items():
                fh.write(json.dumps({"key": key, **read_all(h.float(), 42)}, ensure_ascii=False) + "\n")
        print(f"[read] steering vectors {len(sv)}", flush=True)

if __name__ == "__main__":
    {"download": download, "read": read}[sys.argv[1]]()
