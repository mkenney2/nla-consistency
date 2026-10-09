"""Pod side of threads 1 and 2.

Thread 1 (does sample consistency hold up, and what does it miss?):
  - 8 fresh NLA samples for the 300 WorkspaceBench items the first run skipped (all 500 then).
  - one NLA readout at each of the 3 tokens before the read token (Anthropic's cross-position
    heuristic), for all 500 items.
  - 8 samples each on the mean WorkspaceBench vector and 20 random vectors at the same norm: how
    consistent is the NLA when the activation carries nothing item-specific?
Thread 2 (are consistent wrong claims a reader failure or missing information?):
  - 600 arithmetic prompts "a - b =" / "a + b =" in WorkspaceBench's plain format; layer-42
    activation at the final token, the model's own greedy answer, 2 NLA samples each, and linear
    probes (5-fold) for the digits of a, b and the answer from the same activation.

    python scripts/thread12_pod.py capture   # base model: arithmetic + adjacent-token vectors
    python scripts/thread12_pod.py nla       # all NLA readouts
    python scripts/thread12_pod.py probe     # arithmetic probes (CPU)
"""
import json
import random
import sys
from pathlib import Path

import torch

OUT = Path("/workspace/model-internals-mini")
T = OUT / "t12"
FAMS = ["multihop", "poetry", "typo", "association", "basic_readout"]
LAYER, K = 42, 8


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def arith_prompts() -> list[dict]:
    rng = random.Random(0)
    out = []
    for n in range(600):
        op = "-" if n % 2 == 0 else "+"
        a = rng.randint(2, 99)
        b = rng.randint(1, a - 1) if op == "-" else rng.randint(1, 99)
        out.append({"id": f"ar{n:03d}", "a": a, "b": b, "op": op, "answer": a - b if op == "-" else a + b,
                    "text": f"{a} {op} {b} ="})
    return out


# ---------------------------------------------------------------- capture (base model)

def capture() -> None:
    from wsbench import readplan
    from wsbench.produce.backend import Backend
    from wsbench.produce.render import render

    T.mkdir(parents=True, exist_ok=True)
    b = Backend.load("Qwen/Qwen3.6-27B")
    tok = b.tokenizer

    # arithmetic: final-token vector + the model's greedy next token
    rows = []
    for p in arith_prompts():
        ids = tok(p["text"], add_special_tokens=False)["input_ids"]
        h = b.capture(ids, [LAYER], [len(ids) - 1])[LAYER][0]
        with torch.no_grad():
            logits = b.model(torch.tensor([ids], device="cuda")).logits[0, -1]
        top = tok.decode([int(logits.argmax())])
        rows.append({**p, "h": h.half(), "model_next": top})
    torch.save(rows, T / "arith.pt")
    acc = sum(r["model_next"].strip() == str(r["answer"]) or str(r["answer"]).startswith(r["model_next"].strip()) and r["model_next"].strip() != ""
              for r in rows) / len(rows)
    print(f"[capture] arithmetic {len(rows)}; model's next token starts the right answer: {acc:.2f}", flush=True)

    # adjacent tokens: the 3 positions before each WorkspaceBench read cell
    pos = {(f, r["id"]): r["pos"] for f in FAMS for r in jl(OUT / "verify" / "inputs" / "readouts" / f"{f}.jsonl")}
    adj = []
    for f in FAMS:
        for spec in readplan.plan(f):
            if (f, spec.id) not in pos:
                continue
            r = render(spec, tok)
            read = pos[(f, spec.id)] % len(r)
            want = [read - d for d in (1, 2, 3) if read - d >= 0]
            acts = b.capture(r.ids, [LAYER], want)[LAYER]
            for d, h in zip((1, 2, 3), acts):
                adj.append({"fam": f, "id": spec.id, "d": d, "token": r.decoded[read - d], "h": h.half()})
    torch.save(adj, T / "adjacent.pt")
    print(f"[capture] adjacent vectors {len(adj)}", flush=True)


# ---------------------------------------------------------------- NLA

def nla() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from deception_pod import load_nla, nla_batch

    m, state = load_nla()
    path = T / "nla.jsonl"
    done = {d["key"] for d in jl(path)}
    jobs = []  # (key, vector)

    # 8 samples for the 300 items not sampled before
    have = {(d["fam"], d["id"]) for d in jl(OUT / "verify" / "samples.jsonl")}
    pos = {(f, r["id"]): r["pos"] for f in FAMS for r in jl(OUT / "verify" / "inputs" / "readouts" / f"{f}.jsonl")}
    for f in FAMS:
        for r in torch.load(OUT / "vecs" / f"{f}.pt"):
            if pos.get((f, r["id"])) == r["pos"] and (f, r["id"]) not in have:
                jobs += [(f"sc|{f}|{r['id']}|{k}", r["h"].float()) for k in range(K)]
    # adjacent tokens
    for a in torch.load(T / "adjacent.pt"):
        jobs.append((f"adj|{a['fam']}|{a['id']}|{a['d']}", a["h"].float()))
    # controls
    H = torch.cat([torch.stack([r["h"].float() for r in torch.load(OUT / "vecs" / f"{f}.pt")]) for f in FAMS])
    mean, scale = H.mean(0), H.norm(dim=1).mean()
    g = torch.Generator().manual_seed(0)
    jobs += [(f"ctrl|mean|0|{k}", mean) for k in range(K)]
    for j in range(20):
        v = torch.randn(H.shape[1], generator=g)
        v = v / v.norm() * scale
        jobs += [(f"ctrl|noise|{j}|{k}", v) for k in range(K)]
    # arithmetic
    for r in torch.load(T / "arith.pt"):
        jobs += [(f"ar|{r['id']}|{k}", r["h"].float()) for k in range(2)]

    jobs = [j for j in jobs if j[0] not in done]
    print(f"[nla] {len(jobs)} readouts", flush=True)
    with path.open("a") as fh:
        for s in range(0, len(jobs), 32):
            chunk = jobs[s:s + 32]
            texts = nla_batch(m, state, torch.stack([v for _, v in chunk]))
            for (k, _), t in zip(chunk, texts, strict=True):
                fh.write(json.dumps({"key": k, "text": t}, ensure_ascii=False) + "\n")
            fh.flush()
    print("[nla] done", flush=True)


# ---------------------------------------------------------------- probes

def probe() -> None:
    rows = torch.load(T / "arith.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    X = torch.stack([r["h"].float() for r in rows]).to(dev)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    targets = {"a_tens": [r["a"] // 10 for r in rows], "a_units": [r["a"] % 10 for r in rows],
               "b_tens": [r["b"] // 10 for r in rows], "b_units": [r["b"] % 10 for r in rows],
               "ans_tens": [(r["answer"] // 10) % 10 for r in rows], "ans_units": [r["answer"] % 10 for r in rows],
               "ans_hundreds": [r["answer"] // 100 for r in rows], "op": [int(r["op"] == "+") for r in rows]}
    idx = list(range(len(rows)))
    random.Random(1).shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    preds = {k: [None] * len(rows) for k in targets}
    for name, y in targets.items():
        y_list = y
        y = torch.tensor(y, device=dev)
        ncls = int(y.max()) + 1
        for f in folds:
            test = torch.tensor(f, device=dev)
            fs = set(f)
            train = torch.tensor([i for i in idx if i not in fs], device=dev)
            W = torch.zeros(X.shape[1], ncls, device=dev, requires_grad=True)
            bias = torch.zeros(ncls, device=dev, requires_grad=True)
            opt = torch.optim.LBFGS([W, bias], max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(X[train] @ W + bias, y[train]) + 1e-3 * (W ** 2).sum()
                loss.backward()
                return loss

            opt.step(closure)
            p = (X[test] @ W.detach() + bias.detach()).argmax(1)
            for i, v in zip(f, p.tolist()):
                preds[name][i] = v
        _ = y_list
    out = []
    for i, r in enumerate(rows):
        out.append({"id": r["id"], **{f"probe_{k}": preds[k][i] for k in targets}})
    (T / "probe_preds.json").write_text(json.dumps(out))
    for k, y in targets.items():
        acc = sum(preds[k][i] == y[i] for i in range(len(rows))) / len(rows)
        base = max(y.count(v) for v in set(y)) / len(y)
        print(f"[probe] {k}: acc {acc:.2f} (majority {base:.2f})", flush=True)


if __name__ == "__main__":
    {"capture": capture, "nla": nla, "probe": probe}[sys.argv[1]]()
