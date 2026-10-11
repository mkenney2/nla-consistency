"""Leave-one-family-out check of Result 1: do the filter's numbers hold on a WorkspaceBench family that
played no part in choosing the threshold or the signal weights?

  input claims    : per family, AUROC true vs false of 8-sample support; and the support threshold
                    chosen on the other four families (the strictest that still keeps >= 85% of their
                    workspace claims), applied to the held-out one.
  internal claims : per family, AUROC correct vs wrong of samples/8 + adjacent/3 (fixed weights) and
                    of a logistic regression on (samples, adjacent) fitted on the other four families.

    workspace-bench/.venv/bin/python scripts/lofo.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nla_verify_eval import FAMS, auroc, jl  # noqa: E402
from t12_eval import FALSE, TRUE, T, V, claims_all, labels  # noqa: E402


def records() -> list[dict]:
    claims, lab = claims_all(), labels()
    sup = {d["key"]: d["support"] for d in jl(V / "support.jsonl")} | {d["key"]: d["r"]["support"] for d in jl(T / "support_new.jsonl")}
    adj = {d["key"]: d["r"]["support"] for d in jl(T / "support_adj.jsonl")}
    out = []
    for (f, i), cl in claims.items():
        for n, c in enumerate(cl):
            k = f"{f}|{i}|{n}"
            if k in sup:
                out.append({"fam": f, "label": c["label"], "ws": lab.get(k), "s": sup[k], "a": adj.get(k)})
    return out


def keep_stats(rs: list[dict], t: int) -> tuple[float, float]:
    """(false share among kept claims, share of workspace claims kept) at support >= t."""
    kept = [r for r in rs if r["s"] >= t]
    false = sum(r["label"] in FALSE for r in kept) / max(len(kept), 1)
    ws = [r for r in rs if r["label"] == "workspace"]
    return false, sum(r["s"] >= t for r in ws) / max(len(ws), 1)


def fit_lr(X: list[list[float]], y: list[int], steps: int = 3000, lr: float = 0.5) -> list[float]:
    w = [0.0] * (len(X[0]) + 1)
    for _ in range(steps):
        g = [0.0] * len(w)
        for x, t in zip(X, y):
            p = 1 / (1 + math.exp(-(w[0] + sum(a * b for a, b in zip(w[1:], x)))))
            g[0] += p - t
            for j, v in enumerate(x):
                g[j + 1] += (p - t) * v
        w = [a - lr * b / len(X) for a, b in zip(w, g)]
    return w


def main() -> None:
    rs = records()
    print("## Input claims: 8-sample support, true vs false\n")
    print("| held-out family | AUROC | threshold chosen on the other four | false among kept: raw -> filtered | workspace kept |")
    print("|---|---|---|---|---|")
    pooled = []
    for f in FAMS:
        test = [r for r in rs if r["fam"] == f]
        train = [r for r in rs if r["fam"] != f]
        t = max(t for t in range(9) if keep_stats(train, t)[1] >= 0.85)
        raw = keep_stats(test, 0)[0]
        fs, wk = keep_stats(test, t)
        pooled += [(r, t) for r in test]
        au = auroc([r["s"] for r in test if r["label"] in TRUE], [r["s"] for r in test if r["label"] in FALSE])
        print(f"| {f} | {au:.2f} | >= {t} of 8 | {raw:.0%} -> {fs:.0%} | {wk:.0%} |")
    kept = [r for r, t in pooled if r["s"] >= t]
    ws = [r for r, _ in pooled if r["label"] == "workspace"]
    print(f"| **all, each family held out** | {auroc([r['s'] for r in rs if r['label'] in TRUE], [r['s'] for r in rs if r['label'] in FALSE]):.2f} | – | "
          f"{keep_stats(rs, 0)[0]:.0%} -> {sum(r['label'] in FALSE for r in kept) / len(kept):.0%} | "
          f"{sum(r['s'] >= t for r, t in pooled if r['label'] == 'workspace') / len(ws):.0%} |")

    print("\n## Internal claims: correct vs wrong\n")
    print("| held-out family | correct / wrong | AUROC samples + adjacent (fixed) | AUROC logistic fit on other four | fitted weights (samples/8, adjacent/3) |")
    print("|---|---|---|---|---|")
    ir = [r for r in rs if r["ws"] in ("correct", "wrong") and r["a"] is not None]
    held = []
    for f in FAMS:
        test = [r for r in ir if r["fam"] == f]
        train = [r for r in ir if r["fam"] != f]
        w = fit_lr([[r["s"] / 8, r["a"] / 3] for r in train], [int(r["ws"] == "correct") for r in train])
        score = lambda r: w[1] * r["s"] / 8 + w[2] * r["a"] / 3
        fixed = lambda r: r["s"] / 8 + r["a"] / 3
        c = [r for r in test if r["ws"] == "correct"]
        wr = [r for r in test if r["ws"] == "wrong"]
        held += [(score(r), r["ws"]) for r in test]
        note = " (too few wrong claims)" if len(wr) < 5 else ""
        print(f"| {f} | {len(c)} / {len(wr)}{note} | {auroc([fixed(r) for r in c], [fixed(r) for r in wr]):.2f} | "
              f"{auroc([score(r) for r in c], [score(r) for r in wr]):.2f} | {w[1]:.1f}, {w[2]:.1f} |")
    fx = lambda r: r["s"] / 8 + r["a"] / 3
    print(f"| **all, each family held out** | {sum(r['ws'] == 'correct' for r in ir)} / {sum(r['ws'] == 'wrong' for r in ir)} | "
          f"{auroc([fx(r) for r in ir if r['ws'] == 'correct'], [fx(r) for r in ir if r['ws'] == 'wrong']):.2f} | "
          f"{auroc([s for s, l in held if l == 'correct'], [s for s, l in held if l == 'wrong']):.2f} | – |")


if __name__ == "__main__":
    main()
