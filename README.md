# NLA confabulation: a consistency filter, and the specifics NLAs drop

A mini project on [natural language autoencoders](https://transformer-circuits.pub/2026/nla/) (NLAs): which
parts of an NLA's explanation of an activation can you trust, using only the activation? Run on
[ceselder's open NLA](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl) for Qwen3.6-27B (layer 42, RL step 400),
with [WorkspaceBench](https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the)
items as ground truth.

![Graphical abstract](figures/nla_abstract.png)

## Results

**1. Agreement filters confabulation.** Extending consensus sampling from activation oracles
([Bauer, 2026](https://arxiv.org/abs/2606.02609)) to NLA explanations: keep the claims that recur across 8 samples
of the same activation.

| Agreement signal | Input claims, AUROC true vs false | Internal claims, AUROC correct vs wrong |
| --- | --- | --- |
| 8 samples of the same activation | 0.90 | 0.75 |
| 3 preceding tokens (Anthropic's heuristic) | 0.74 | 0.69 |
| Both | 0.92 | 0.80 |

Keeping claims stated in at least 2 of 8 samples cuts false claims about the input from 38% to 13%, and keeps
87% of workspace claims, with no source text. For claims about internal computation, about a third of the wrong
ones repeat in at least 7 of 8 samples: the NLA rebuilds a consistent wrong copy of the task ("10 − 1 =" read as
"2 − 1 =", answer 1). The preceding-token check still separates right from wrong among those (AUROC 0.67). On
random vectors 29% of the NLA's claims are self-consistent, so consistency is evidence, not proof.

**2. The NLA drops specifics the activation holds.** On 600 "a ± b =" prompts, a linear probe reads both operands
from the layer-42 activation exactly 83–93% of the time. The NLA, reading the same vector (it never sees the
prompt), always gets the operator right but the first operand only 5% of the time and the second 38%; its numbers
keep the rough size (r = 0.57 and 0.88) and cluster on a few default values. [Dingeto (2026)](https://arxiv.org/abs/2607.20379)
showed this readout gap in a sandbox but found the specifics absent from the activation on a released NLA; here
they are present and still dropped. The probe is supervised on this format and the NLA is not.

Why NLAs confabulate (a reconstruction reward that pays for pseudo-quotes and ignores false specifics) is covered by
[Anurin](https://www.lesswrong.com/posts/4nZwNPyfoadAxPH98/natural-language-autoencoders-are-summarizers-but-do-they),
[Zhang & TurnTrout](https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to),
[Hong](https://www.lesswrong.com/posts/DFgt8fi3Wzwwe2Sib/fixing-rewards-for-nla-to-reduce-confabulation) and
[Lek et al.](https://arxiv.org/abs/2609.33702).

## Contents

| Path | What |
| --- | --- |
| `notebook.ipynb` | Every number and figure above, recomputed from `data/` (CPU only, a few seconds) |
| `data/` | Claim labels, agreement counts, NLA samples, control vectors, arithmetic probes and readouts (described in the notebook) |
| `figures/` | The figures the notebook writes, plus the graphical abstract |
| `scripts/` | The pipeline that produced `data/` (GPU and API steps) |

Run the notebook:

```bash
uv run --no-project --with jupyter --with matplotlib jupyter lab notebook.ipynb
```

## Limitations

One open NLA, one layer and checkpoint; claim labels from one Gemini judge (every wrong internal claim checked by
hand); the preceding-token check also drops many correct claims; the arithmetic result covers operands in a format
where the model does not answer, with a probe supervised on that format.

## Reproducing `data/`

The scripts assume the [workspace-bench](https://github.com/camilablank/workspace-bench) package, the NLA weights,
a GPU with about 140 GB for the model and NLA together, and an OpenRouter key for the Gemini judges. The notebook's
last section maps each step to its script.
