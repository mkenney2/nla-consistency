"""Pod side of the deception test: layer-42 activations of Qwen3.6-27B on its own answers, then
NLA readouts of them (batched).

    python scripts/deception_pod.py capture   # OUT/deception/vecs/<dataset>.pt
    python scripts/deception_pod.py check     # batched NLA == one-at-a-time NLA (greedy, 3 vectors)
    python scripts/deception_pod.py nla       # OUT/deception/readouts/nla_<pos>/<dataset>.jsonl
    python scripts/deception_pod.py probe     # OUT/deception/probe_scores.jsonl

Inputs: OUT/deception/data/<dataset>.jsonl from scripts/deception_data.py. Each answer is fed as
the generation prompt (chat template, thinking off) + the answer's tokens, i.e. exactly the
sequence the model produced. Saved per row: the mean over answer tokens (for the probe) and the
first and last answer token (for the NLA).
"""
import json
import sys
from pathlib import Path

import torch

OUT = Path("/workspace/model-internals-mini/deception")
LAYER = 42
DATASETS = ["repe", "roleplaying", "ai_liar", "alpaca"]
NLA_DATASETS = ["roleplaying", "ai_liar", "alpaca"]
NLA_POS = ["last", "first"]
BATCH = 32


def rows(ds: str) -> list[dict]:
    return [json.loads(l) for l in (OUT / "data" / f"{ds}.jsonl").read_text().splitlines() if l.strip()]


def key(r: dict) -> str:
    return f"{r['sid']}/{r['sample']}"


def capture() -> None:
    from wsbench.produce.backend import Backend

    b = Backend.load("Qwen/Qwen3.6-27B")
    tok = b.tokenizer
    (OUT / "vecs").mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        path = OUT / "vecs" / f"{ds}.pt"
        if path.exists():
            continue
        out = []
        for r in rows(ds):
            pre = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
            pre = list(pre["input_ids"] if hasattr(pre, "keys") else pre)
            ans = tok(r["answer"], add_special_tokens=False)["input_ids"][:600]
            pos = list(range(len(pre), len(pre) + len(ans)))
            acts = b.capture(pre + ans, [LAYER], pos)[LAYER]  # [n_answer, d]
            out.append({"key": key(r), "label": r["label"], "mean": acts.mean(0).half(),
                        "first": acts[0].half(), "last": acts[-1].half(), "n": len(ans)})
        torch.save(out, path)
        print(f"[vecs] {ds}: {len(out)}", flush=True)


def load_nla(adapter: str = "av_rl_adapters/iter_000400"):
    from wsbench.produce.backend import Backend
    from wsbench.produce.methods import NLA, Sampling

    m = NLA(sampling=Sampling(k=1), adapter=adapter)
    m.bind(Backend(model=None, tokenizer=None, device="cuda", model_id="Qwen/Qwen3.6-27B"))
    (block1,) = [mod for n, mod in m._reader.named_modules() if n.endswith("model.layers.1")]
    block1._forward_hooks.clear()  # replace the one-vector hook with a per-row one
    mp, state = m._marker, {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = state.get("v")
        if v is None or h.shape[1] <= mp:
            return out
        hp = h[:, mp, :]
        vu = v.to(h.device, h.dtype) / (v.norm(dim=-1, keepdim=True).to(h.device, h.dtype) + 1e-9)
        h[:, mp, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
        return out

    block1.register_forward_hook(hook)
    return m, state


def nla_batch(m, state, V: torch.Tensor, greedy: bool = False) -> list[str]:
    s = m.sampling
    state["v"] = V.float()
    try:
        with torch.no_grad():
            out = m._reader.generate(
                torch.tensor([m._ids] * len(V), device="cuda"), do_sample=not greedy,
                temperature=s.temperature, top_p=s.top_p, top_k=s.top_k, max_new_tokens=s.max_new_tokens,
                pad_token_id=m._tok.pad_token_id or m._tok.eos_token_id)
    finally:
        state["v"] = None
    texts = m._tok.batch_decode(out[:, len(m._ids):], skip_special_tokens=True)
    return [t.split("</explanation>")[0].removeprefix("<explanation>").strip() for t in texts]


def check() -> None:
    m, state = load_nla()
    vecs = torch.load(OUT / "vecs" / "roleplaying.pt")[:3]
    V = torch.stack([v["last"].float() for v in vecs])
    batched = nla_batch(m, state, V, greedy=True)
    single = [nla_batch(m, state, V[i:i + 1], greedy=True)[0] for i in range(len(V))]
    for a, b in zip(batched, single, strict=True):
        print("SAME" if a == b else "DIFF", "|", a[:120].replace("\n", " "), "||", b[:120].replace("\n", " "))


def nla() -> None:
    m, state = load_nla()
    for pos in NLA_POS:
        for ds in NLA_DATASETS:
            path = OUT / "readouts" / f"nla_{pos}" / f"{ds}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(l)["key"] for l in path.read_text().splitlines()} if path.exists() else set()
            vecs = [v for v in torch.load(OUT / "vecs" / f"{ds}.pt") if v["key"] not in done]
            with path.open("a") as fh:
                for i in range(0, len(vecs), BATCH):
                    chunk = vecs[i:i + BATCH]
                    texts = nla_batch(m, state, torch.stack([v[pos].float() for v in chunk]))
                    for v, t in zip(chunk, texts, strict=True):
                        fh.write(json.dumps({"key": v["key"], "label": v["label"], "text": t}, ensure_ascii=False) + "\n")
                    fh.flush()
            print(f"[nla_{pos}] {ds}: done", flush=True)


def probe() -> None:
    """Apollo-style probe: trained only on the repe persona facts (honest vs dishonest persona,
    mean over answer tokens), applied unchanged to every dataset. Two variants: L2 logistic
    regression on standardized activations, and the plain difference of class means."""
    tr = torch.load(OUT / "vecs" / "repe.pt")
    X = torch.stack([v["mean"].float() for v in tr])
    y = torch.tensor([1.0 if v["label"] == "deceptive" else 0.0 for v in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    lam = 10.0 / len(Z)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y) + lam * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w, b = w.detach(), b.detach()
    md = (Z[y == 1].mean(0) - Z[y == 0].mean(0))
    acc = (((Z @ w + b) > 0).float() == y).float().mean().item()
    print(f"[probe] repe train acc {acc:.3f}", flush=True)
    with (OUT / "probe_scores.jsonl").open("w") as fh:
        for ds in NLA_DATASETS:
            for v in torch.load(OUT / "vecs" / f"{ds}.pt"):
                rec = {"dataset": ds, "key": v["key"], "label": v["label"]}
                for feat in ("mean", "last"):
                    z = (v[feat].float() - mu) / sd
                    rec[f"lr_{feat}"] = float(z @ w + b)
                    rec[f"md_{feat}"] = float(z @ md)
                fh.write(json.dumps(rec) + "\n")
    print("[probe] scores written", flush=True)


if __name__ == "__main__":
    {"capture": capture, "check": check, "nla": nla, "probe": probe}[sys.argv[1]]()
