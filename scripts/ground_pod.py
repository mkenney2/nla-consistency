"""Pod side of the grounding tests (results/ground/ locally).

Test 1, claim-level grounding: for each internal claim that names a word (groups of claims on the same
item naming the same words share one ablation), mean-ablate the J++ Lens read directions of those words
in the layer-42 vector the NLA read, h' = h - U U^T (h - mu), and, as a control, a random subspace of the
same rank. Then 8 NLA samples on each; locally a judge counts how many still state the claim.
Read direction of token t at layer 42: J42^T (g * W_U[t]), g the final RMSNorm weight.

Test 2, computed intermediates: two-hop prompts ("the capital of the country where the Eiffel Tower is
located is") whose bridge (France) is swapped for another (Italy) by moving the final position's
residual at one layer along J++ lens directions: h' = h - (h.r_X) r_X + beta |h| r_Y. A small grid on
the first pairs picks the layer and beta that most often turn the model's answer into the new bridge's;
then every pair runs at that setting, plus a random-direction control. The layer-42 vector of each
patched run goes to the NLA (8 samples).

    python scripts/ground_pod.py prep    # base model: test 1 vectors, test 2 grid + vectors + answers
    python scripts/ground_pod.py nla     # NLA readouts of every vector
"""
import json
import random
import re
import sys
from pathlib import Path

import torch

OUT = Path("/workspace/model-internals-mini")
G = OUT / "ground"
FAMS = ["multihop", "poetry", "typo", "association", "basic_readout"]
K = 8
import os

SUFFIX = os.environ.get("GROUND_SUFFIX", "")  # "" = first pass (tests 1 and 2), "2" = wider test 2 only
if SUFFIX:  # patch a range of layers and the last few positions, at larger strengths
    SETTINGS = [(list(range(20, 41, 2)), 1), (list(range(20, 41, 2)), 4), (list(range(28, 41)), 4)]
    BETAS = [0.1, 0.2, 0.4]
else:
    SETTINGS = [(24, 1), (32, 1), (40, 1)]
    BETAS = [0.05, 0.1, 0.2]
PATCH_LAYERS = sorted({L for Ls, _ in SETTINGS for L in ([Ls] if isinstance(Ls, int) else Ls)})

CAPITAL = {"France": ("the Eiffel Tower", "Paris"), "Italy": ("the Colosseum", "Rome"), "Japan": ("Mount Fuji", "Tokyo"),
           "Egypt": ("the Great Pyramid of Giza", "Cairo"), "Russia": ("the Kremlin", "Moscow"), "Spain": ("the Alhambra", "Madrid"),
           "Germany": ("the Brandenburg Gate", "Berlin"), "Greece": ("the Parthenon", "Athens"), "China": ("the Forbidden City", "Beijing"),
           "Peru": ("Machu Picchu", "Lima"), "Australia": ("the Sydney Opera House", "Canberra"), "Turkey": ("Hagia Sophia", "Ankara"),
           "Canada": ("the CN Tower", "Ottawa"), "Kenya": ("the Maasai Mara", "Nairobi"), "Norway": ("the Geirangerfjord", "Oslo"),
           "Austria": ("Schönbrunn Palace", "Vienna")}
LEGS = {"spider": ("spins webs", "8"), "bee": ("makes honey", "6"), "dog": ("barks at the mailman", "4"),
        "chicken": ("lays the eggs we eat for breakfast", "2"), "snake": ("slithers through the grass", "0"),
        "ant": ("lives in an anthill", "6"), "kangaroo": ("carries its baby in a pouch", "2"), "cat": ("purrs on your lap", "4")}
COLOR = {"banana": ("monkeys famously love to peel", "yellow"), "lime": ("is squeezed into margaritas", "green"),
         "blueberry": ("is baked into muffins", "blue"), "grape": ("is crushed to make wine", "purple"),
         "strawberry": ("is dipped in chocolate on Valentine's Day", "red"), "coconut": ("grows on tropical palm trees", "brown")}
TASKS = {"capital": (CAPITAL, "Fact: The capital of the country where {d} is located is"),
         "legs": (LEGS, "Fact: The number of legs on the animal that {d} is"),
         "color": (COLOR, "Fact: The color of the fruit that {d} is")}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pairs() -> list[dict]:
    rng = random.Random(3)
    out = []
    for task, (table, tmpl) in TASKS.items():
        for x, (d, ans) in table.items():
            alts = [y for y, (_, a) in table.items() if y != x and a != ans]
            for y in rng.sample(alts, min(3, len(alts))):
                out.append({"id": f"{task}-{x}-{y}", "task": task, "x": x, "y": y, "x_ans": ans, "y_ans": table[y][1],
                            "prompt": tmpl.format(d=d)})
    rng.shuffle(out)
    return out


def orth(D: torch.Tensor) -> torch.Tensor:
    U, S, _ = torch.linalg.svd(D, full_matrices=False)
    return U[:, S > S[0] * 1e-4]


def prep() -> None:
    from huggingface_hub import hf_hub_download
    from wsbench.produce.backend import Backend

    G.mkdir(parents=True, exist_ok=True)
    b = Backend.load("Qwen/Qwen3.6-27B")
    tok, dev = b.tokenizer, "cuda"
    w_u = b.unembed
    with torch.no_grad():  # the final norm's effective per-dimension gain (covers 1 + weight variants)
        x = torch.randn(1, 5120, device=dev)
        g = (b.final_norm(x.to(w_u.dtype)).float() / (x / x.pow(2).mean().sqrt()))[0]
    obj = torch.load(hf_hub_download("koayon/jpp-lenses", "qwen3.6-27b/lens.pt"), map_location="cpu", weights_only=True)
    jac = obj["parameters"]["jacobians"]
    need = sorted(set(PATCH_LAYERS + [42]))
    J = {L: (jac[L] if L in jac else jac[str(L)]).float().to(dev) for L in need}
    del obj, jac

    def read_dir(L, ids):  # (d, n) read directions of tokens at layer L
        return J[L].T @ (g[:, None] * w_u[ids].float().T)

    def token_ids(word):
        ids = set()
        for v in {word, " " + word, word.lower(), " " + word.lower(), word.capitalize(), " " + word.capitalize()}:
            t = tok(v, add_special_tokens=False)["input_ids"]
            if t and len(tok.decode([t[0]]).strip()) >= min(3, len(word.strip())):
                ids.add(t[0])
        return sorted(ids)

    def jpp_rank(h, word):
        logits = (b.final_norm((h.to(dev).float() @ J[42].T)).to(w_u.dtype) @ w_u.T).float()
        order = torch.argsort(logits, descending=True)
        pos = {int(t): i for i, t in enumerate(order[:2000].tolist())}
        ids = token_ids(word)
        return min((pos.get(i, 2000) for i in ids), default=2000)

    # ---- test 1 (first pass only)
    if SUFFIX:
        t1 = {}
    pos = {(f, r["id"]): r["pos"] for f in FAMS for r in jl(OUT / "verify" / "inputs" / "readouts" / f"{f}.jsonl")}
    if SUFFIX:
        groups = []
    vec = {}
    for f in FAMS:
        for r in torch.load(OUT / "vecs" / f"{f}.pt"):
            if pos.get((f, r["id"])) == r["pos"]:
                vec[(f, r["id"])] = r["h"].float().to(dev)
    mu = torch.stack(list(vec.values())).mean(0)
    groups = [] if SUFFIX else json.loads((G / "test1_groups.json").read_text())
    gen = torch.Generator(device="cpu").manual_seed(0)
    t1, meta1 = {}, []
    for gk, info in groups:
        h = vec[(info["fam"], info["id"])]
        ids = sorted({i for e in info["entities"] for i in token_ids(e)})
        if not ids:
            continue
        U = orth(read_dir(42, ids))
        R = orth(torch.randn(h.shape[0], U.shape[1], generator=gen).to(dev))
        ha = h - U @ (U.T @ (h - mu))
        hr = h - R @ (R.T @ (h - mu))
        t1[f"t1|{gk}|ablate"], t1[f"t1|{gk}|control"] = ha.half().cpu(), hr.half().cpu()
        e0 = info["entities"][0]
        meta1.append({"group": gk, "rank": len(ids), "jpp_rank_before": jpp_rank(h, e0), "jpp_rank_ablate": jpp_rank(ha, e0),
                      "jpp_rank_control": jpp_rank(hr, e0), "norm_change": float((ha - h).norm() / h.norm())})
    if not SUFFIX:
        (G / "test1_meta.json").write_text(json.dumps(meta1))
    if meta1:
        print(f"[prep] test 1: {len(meta1)} groups; median J++ rank of the named word before / ablated / control: "
          f"{sorted(m['jpp_rank_before'] for m in meta1)[len(meta1) // 2]} / {sorted(m['jpp_rank_ablate'] for m in meta1)[len(meta1) // 2]} / "
          f"{sorted(m['jpp_rank_control'] for m in meta1)[len(meta1) // 2]}", flush=True)

    # ---- test 2
    blk = b.blocks

    def run(prompt, L=None, delta=None, npos=1):
        """Greedy 3 tokens; optionally h += delta(L, h) at each block in L (an int or a list) and each
        of the last npos prompt positions; returns (answer text, layer-42 vector at the final prompt position)."""
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        store = {}
        hooks = []
        for layer in ([] if L is None else ([L] if isinstance(L, int) else L)):
            def patch(_m, _i, out, layer=layer):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] > 1:
                    for q in range(1, npos + 1):
                        h[0, -q, :] = (h[0, -q, :].float() + delta(layer, h[0, -q, :].float())).to(h.dtype)
                return out
            hooks.append(blk[layer].register_forward_hook(patch))

        def grab(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                store["h42"] = h[0, -1, :].detach().float().cpu()
            return out
        hooks.append(blk[42].register_forward_hook(grab))
        try:
            with torch.no_grad():
                out = b.model.generate(torch.tensor([ids], device=dev), max_new_tokens=3, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
        finally:
            for hk in hooks:
                hk.remove()
        return tok.decode(out[0, len(ids):]), store["h42"]

    def says(text, word):
        return re.match(r"\s*" + re.escape(word.lower()), text.lower()) is not None

    unit_cache = {}

    def unit(L, word):
        if (L, word) not in unit_cache:
            d = read_dir(L, token_ids(word)).sum(1)
            unit_cache[(L, word)] = d / d.norm()
        return unit_cache[(L, word)]

    P = pairs()
    base = {}
    for p in P:
        if p["prompt"] not in base:
            base[p["prompt"]] = run(p["prompt"])
    P = [p for p in P if says(base[p["prompt"]][0], p["x_ans"])]
    print(f"[prep] test 2: {len(P)} pairs whose base prompt the model answers correctly", flush=True)

    def swap_delta(L, x, y, beta, rand=None):
        def d(layer, h):
            rx = unit(layer, x) if rand is None else rand[0]
            ry = unit(layer, y) if rand is None else rand[1]
            return -(h @ rx) * rx + beta * h.norm() * ry
        return d

    dev_set = P[:15]
    grid = []
    for L, npos in SETTINGS:
        for beta in BETAS:
            ok = sum(says(run(p["prompt"], L, swap_delta(L, p["x"], p["y"], beta), npos)[0], p["y_ans"]) for p in dev_set)
            grid.append({"layer": L, "npos": npos, "beta": beta, "success": ok / len(dev_set)})
            print(f"[prep] grid layers {L} last {npos} positions beta {beta}: {ok}/{len(dev_set)}", flush=True)
    best = max(grid, key=lambda r: (r["success"], -r["beta"]))
    L, npos, beta = best["layer"], best["npos"], best["beta"]
    rows, t2 = [], {}
    for p in P:
        r1, r2 = torch.randn(5120, generator=gen), torch.randn(5120, generator=gen)
        rand = ((r1 / r1.norm()).to(dev), (r2 / r2.norm()).to(dev))
        a_sw, h_sw = run(p["prompt"], L, swap_delta(L, p["x"], p["y"], beta), npos)
        a_rc, h_rc = run(p["prompt"], L, swap_delta(L, p["x"], p["y"], beta, rand), npos)
        rows.append({**p, "base_answer": base[p["prompt"]][0], "swap_answer": a_sw, "control_answer": a_rc,
                     "swap_success": says(a_sw, p["y_ans"]), "control_success": says(a_rc, p["y_ans"]),
                     "control_unchanged": says(a_rc, p["x_ans"])})
        t2[f"t2|{p['id']}|base"], t2[f"t2|{p['id']}|swap"], t2[f"t2|{p['id']}|control"] = \
            base[p["prompt"]][1].half(), h_sw.half(), h_rc.half()
    (G / f"test2_rows{SUFFIX}.json").write_text(json.dumps({"grid": grid, "layer": L, "npos": npos, "beta": beta, "rows": rows}))
    print(f"[prep] test 2 at layer {L}, beta {beta}: swap success {sum(r['swap_success'] for r in rows)}/{len(rows)}, "
          f"control success {sum(r['control_success'] for r in rows)}/{len(rows)}, control unchanged {sum(r['control_unchanged'] for r in rows)}/{len(rows)}", flush=True)
    torch.save({**t1, **t2} if not SUFFIX else t2, G / f"vecs{SUFFIX}.pt")


def nla() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from deception_pod import load_nla, nla_batch

    m, state = load_nla()
    vecs = torch.load(G / f"vecs{SUFFIX}.pt")
    path = G / f"nla{SUFFIX}.jsonl"
    done = {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
    # test 2 first (smaller), then test 1
    keys = sorted(vecs, key=lambda k: (not k.startswith("t2"), k))
    jobs = [(f"{k}|{s}", vecs[k]) for k in keys for s in range(K) if f"{k}|{s}" not in done]
    print(f"[nla] {len(jobs)} readouts", flush=True)
    with path.open("a") as fh:
        for i in range(0, len(jobs), 32):
            chunk = jobs[i:i + 32]
            texts = nla_batch(m, state, torch.stack([v.float() for _, v in chunk]))
            for (k, _), t in zip(chunk, texts, strict=True):
                fh.write(json.dumps({"key": k, "text": t}, ensure_ascii=False) + "\n")
            fh.flush()
    print("[nla] done", flush=True)


if __name__ == "__main__":
    {"prep": prep, "nla": nla}[sys.argv[1]]()
