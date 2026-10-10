"""Pod side of the steering test: does the NLA's claim about an operand follow the operand?

Prompts are few-shot arithmetic ("12 + 7 = 19 ... a op b =") so the model answers. For pairs of
prompts that differ in one operand (base vs source), the layer-42 vector at "=" of the base prompt
gets only that operand's subspace swapped in from the source (interchange intervention):
    h' = h_base + U U^T (h_src - h_base)
Then: does a linear probe now read the source operand, does the model's answer change (patching
only the "=" position), and does the NLA now name the source operand?

Pair sets: "a" (first operand, two digits), "b" (second operand, two digits), "b1" (second operand,
one digit: the NLA reads these 96% of the time, so this is the positive control).
Subspaces per operand: "mean" (class-mean directions of its tens and units digits), "lr" (the
logistic-probe weight directions), "rand" (random, same rank as "mean"; the control).

    python scripts/steer_pod.py capture      # base model: probes, subspaces, patched vectors, model answers
    python scripts/steer_pod.py smoke        # NLA on 4 pairs, printed
    python scripts/steer_pod.py nla          # NLA, K samples per pair x condition
"""
import json
import random
import sys
from pathlib import Path

import torch

OUT = Path("/workspace/model-internals-mini/steer")
LAYER, K, NPAIR = 42, 8, 100
FEW = "12 + 7 = 19\n45 - 13 = 32\n8 + 26 = 34\n71 - 9 = 62\n"
NLA_CONDS = ["base", "src", "swap_mean", "swap_lr", "swap_rand"]


def text(a: int, op: str, b: int) -> str:
    return f"{FEW}{a} {op} {b} ="


def train_set() -> list[tuple]:
    """The 600 operand pairs of the earlier arithmetic run, now in the few-shot format."""
    rng = random.Random(0)
    out = []
    for n in range(600):
        op = "-" if n % 2 == 0 else "+"
        a = rng.randint(2, 99)
        b = rng.randint(1, a - 1) if op == "-" else rng.randint(1, 99)
        out.append((a, op, b))
    return out


def pairs() -> list[dict]:
    rng = random.Random(7)
    out = []
    for var in ("a", "b", "b1"):
        for n in range(NPAIR):
            op = "-" if n % 2 == 0 else "+"
            while True:
                a = rng.randint(10, 99)
                b = rng.randint(1, 9) if var == "b1" else rng.randint(10, 99)
                if var == "a":
                    a2, b2 = rng.randint(10, 99), b
                    differ = a // 10 != a2 // 10
                elif var == "b":
                    a2, b2 = a, rng.randint(10, 99)
                    differ = b // 10 != b2 // 10
                else:
                    a2, b2 = a, rng.randint(1, 9)
                    differ = b != b2
                if differ and (op == "+" or (a > b and a2 > b2)):
                    break
            out.append({"id": f"{var}{n:03d}", "var": var, "op": op, "base": [a, b], "src": [a2, b2]})
    return out


def digits(a: int, b: int) -> dict:
    return {"a_tens": a // 10, "a_units": a % 10, "b_tens": b // 10, "b_units": b % 10}


def fit_lr(Z: torch.Tensor, y: torch.Tensor, ncls: int = 10) -> tuple:
    W = torch.zeros(Z.shape[1], ncls, device=Z.device, requires_grad=True)
    bias = torch.zeros(ncls, device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, bias], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ W + bias, y) + 1e-3 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), bias.detach()


def orth(D: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (d x r) for the span of the columns of D."""
    U, S, _ = torch.linalg.svd(D, full_matrices=False)
    return U[:, S > S[0] * 1e-4]


# ---------------------------------------------------------------- capture (base model)

def capture() -> None:
    from wsbench.produce.backend import Backend

    OUT.mkdir(parents=True, exist_ok=True)
    bk = Backend.load("Qwen/Qwen3.6-27B")
    tok = bk.tokenizer

    def ids(a, op, b):
        return tok(text(a, op, b), add_special_tokens=False)["input_ids"]

    def h_of(a, op, b):
        x = ids(a, op, b)
        return bk.capture(x, [LAYER], [len(x) - 1])[LAYER][0]

    def answer(a, op, b, patch=None) -> str:
        x = ids(a, op, b)
        blk = bk.blocks[LAYER]

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if patch is not None and h.shape[1] > 1:  # the prompt pass only
                h[0, -1, :] = patch.to(h.device, h.dtype)
            return out

        hd = blk.register_forward_hook(hook)
        try:
            with torch.no_grad():
                g = bk.model.generate(torch.tensor([x], device="cuda"), max_new_tokens=4, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
        finally:
            hd.remove()
        return tok.decode(g[0, len(x):])

    # probes and subspaces from the 600 training prompts
    tr = train_set()
    X = torch.stack([h_of(*t) for t in tr]).cuda()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ys = {k: torch.tensor([digits(a, b)[k] for a, _, b in tr], device="cuda") for k in ("a_tens", "a_units", "b_tens", "b_units")}
    lr = {k: fit_lr(Z, y) for k, y in ys.items()}
    sub = {}
    g = torch.Generator().manual_seed(0)
    for v in ("a", "b"):
        mean_dirs, lr_dirs = [], []
        for k in (f"{v}_tens", f"{v}_units"):
            for c in ys[k].unique():
                mean_dirs.append(X[ys[k] == c].mean(0) - mu)
            lr_dirs.append(lr[k][0] / sd[:, None])  # z-space weights -> raw-space read directions
        sub[f"{v}|mean"] = orth(torch.stack(mean_dirs, 1))
        sub[f"{v}|lr"] = orth(torch.cat(lr_dirs, 1))
        r = sub[f"{v}|mean"].shape[1]
        sub[f"{v}|rand"] = orth(torch.randn(X.shape[1], r, generator=g).cuda())
        print(f"[capture] operand {v}: rank mean {r}, lr {sub[f'{v}|lr'].shape[1]}", flush=True)

    def read(h) -> dict:
        z = ((h.cuda() - mu) / sd)[None]
        return {k: int((z @ W + b).argmax()) for k, (W, b) in lr.items()}

    def num(d, v):
        return d[f"{v}_tens"] * 10 + d[f"{v}_units"]

    rows, vecs = [], {}
    for p in pairs():
        a, b = p["base"]
        a2, b2 = p["src"]
        op, v = p["op"], p["var"][0]
        hb, hs = h_of(a, op, b).cuda(), h_of(a2, op, b2).cuda()
        conds = {"base": hb, "src": hs}
        for kind in ("mean", "lr", "rand"):
            U = sub[f"{v}|{kind}"]
            conds[f"swap_{kind}"] = hb + U @ (U.T @ (hs - hb))
        rec = {**p, "probe": {}, "model": {}, "norm": {}}
        for c, h in conds.items():
            d = read(h)
            rec["probe"][c] = [num(d, "a"), num(d, "b")]
            rec["norm"][c] = float(h.norm())
            vecs[f"{p['id']}|{c}"] = h.cpu().half()
        rec["model"]["base"] = answer(a, op, b)
        rec["model"]["src"] = answer(a2, op, b2)
        rec["model"]["self"] = answer(a, op, b, hb)  # patching in its own vector: should equal base
        rec["model"]["full"] = answer(a, op, b, hs)  # the whole source vector at "="
        for kind in ("mean", "lr", "rand"):
            rec["model"][f"swap_{kind}"] = answer(a, op, b, conds[f"swap_{kind}"])
        rows.append(rec)
    torch.save(vecs, OUT / "vecs.pt")
    (OUT / "pairs.json").write_text(json.dumps(rows))

    def ans(s):
        t = s.strip().split()[0] if s.strip() else ""
        return int(t) if t.lstrip("-").isdigit() else None

    def truth(p, which):
        x, y = p[which]
        return x - y if p["op"] == "-" else x + y

    print(f"[capture] model right on base prompts: {sum(ans(r['model']['base']) == truth(r, 'base') for r in rows)}/{len(rows)}; "
          f"self-patch unchanged: {sum(r['model']['self'] == r['model']['base'] for r in rows)}/{len(rows)}", flush=True)
    for var in ("a", "b", "b1"):
        rs = [r for r in rows if r["var"] == var]
        i = 0 if var == "a" else 1
        line = [f"base {sum(r['probe']['base'][i] == r['base'][i] for r in rs)}"]
        for c in ("full", "swap_mean", "swap_lr", "swap_rand"):
            pc = c if c != "full" else "src"
            line.append(f"{c}: probe->src {sum(r['probe'][pc][i] == r['src'][i] for r in rs)}, "
                        f"model->src answer {sum(ans(r['model'][c]) == truth(r, 'src') for r in rs)}")
        print(f"[capture] {var} (n={len(rs)}): " + " | ".join(line), flush=True)


# ---------------------------------------------------------------- NLA

def nla(smoke: bool = False) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from deception_pod import load_nla, nla_batch

    m, state = load_nla()
    vecs = torch.load(OUT / "vecs.pt")
    rows = json.loads((OUT / "pairs.json").read_text())
    if smoke:
        for r in [rows[0], rows[1], rows[NPAIR], rows[2 * NPAIR]]:
            for c in ("base", "src"):
                for t in nla_batch(m, state, torch.stack([vecs[f"{r['id']}|{c}"].float()] * 2)):
                    print(f"--- {r['id']} {c} {r[c]} {r['op']}\n{t[:600]}\n", flush=True)
        return
    path = OUT / "nla.jsonl"
    done = {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
    jobs = [(f"{r['id']}|{c}|{k}", vecs[f"{r['id']}|{c}"]) for r in rows for c in NLA_CONDS for k in range(K)]
    jobs = [j for j in jobs if j[0] not in done]
    print(f"[nla] {len(jobs)} readouts", flush=True)
    with path.open("a") as fh:
        for s in range(0, len(jobs), 32):
            chunk = jobs[s:s + 32]
            texts = nla_batch(m, state, torch.stack([v.float() for _, v in chunk]))
            for (k, _), t in zip(chunk, texts, strict=True):
                fh.write(json.dumps({"key": k, "text": t}, ensure_ascii=False) + "\n")
            fh.flush()
    print("[nla] done", flush=True)


if __name__ == "__main__":
    {"capture": capture, "smoke": lambda: nla(True), "nla": nla}[sys.argv[1]]()
