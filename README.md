<p align="center">
  <img src="benchmarks/architecture_overview.png" width="100%">
</p>

<h1 align="center">H-JEPA-LM</h1>

<p align="center">
  <strong>Hierarchical Joint-Embedding Predictive Language Model</strong><br>
  A fundamentally different approach to language modeling — predicting latent representations, not tokens.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch 2.0+"></a>
  <a href="/Griffith-7/H-JEPA-LM/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/Griffith-7/H-JEPA-LM/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <a href="https://github.com/Griffith-7/H-JEPA-LM/actions"><img src="https://github.com/Griffith-7/H-JEPA-LM/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

## Two models in one package

This repo contains **two models**. Both predict latent representations instead of tokens, but they differ in complexity:

| | **JEPA-LM** | **H-JEPA-LM** |
|:--|:--|:--|
| **Class** | `JEPELM` | `HJEPELM` |
| **Module** | `jepalm.model` | `jepalm.hjepa` |
| **Architecture** | Single-level encoder + predictor | Multi-level hierarchical encoder |
| **Action conditioning** | No | Yes |
| **World model planning** | No | Yes (latent-space rollouts) |
| **Best for** | Research, baselines, simpler experiments | Production, action-aware tasks |
| **Parameters** | ~5.6M (tiny config) | ~5.8M (tiny config) |

### When to use which

- **Use JEPA-LM** if you want a simple, clean baseline. Single-level prediction with EMA target encoder. Good for understanding JEPA fundamentals.
- **Use H-JEPA-LM** if you need hierarchical prediction, action conditioning, or world model planning. This is the full-featured model.

Both models share the same core idea: predict latent representations (not tokens) via cosine similarity loss, with an EMA target encoder to prevent collapse.

## Install

```bash
pip install git+https://github.com/Griffith-7/H-JEPA-LM.git
```

Or clone and install locally:

```bash
git clone https://github.com/Griffith-7/H-JEPA-LM.git
cd H-JEPA-LM
pip install -e ".[dev]"
```

## Quick start

### JEPA-LM (base model)

```python
from jepalm import JEPELM, JEPAConfig

config = JEPAConfig(
    enc_hidden_dim=256,
    enc_num_layers=4,
    enc_num_heads=4,
    pred_hidden_dim=128,
    pred_num_layers=2,
)

model = JEPELM(config)
print(f"Parameters: {model.count_parameters()['total']:,}")

import torch
x = torch.randint(0, 1000, (2, 32))
out = model(x)
print(f"JEPA loss: {out['jepa_loss']:.4f}")
print(f"Cosine similarity: {out['cosine_similarity']:.4f}")
```

### H-JEPA-LM (hierarchical + action conditioning)

```python
from jepalm import HJEPELM, HConfig

config = HConfig(
    vocab_size=30522,
    dim=256,
    layers=4,
    heads=4,
    ff_dim=1024,
    max_len=128,
    num_levels=2,
    action_dim=64,
)

model = HJEPELM(config)
print(f"Parameters: {model.count_parameters()['total']:,}")

import torch
x = torch.randint(0, 30522, (2, 64))

# Basic forward pass (returns multi-level latent outputs)
outputs = model(x)
print(f"Highest level shape: {outputs[-1].shape}")

# Compute JEPA loss (with optional action conditioning)
loss, val = model.compute_loss(x)
print(f"Loss: {loss.item():.4f}")

# With actions
actions = torch.randn(2, 64)
loss, val = model.compute_loss(x, actions=actions)
print(f"Loss (with action): {loss.item():.4f}")

# World model: predict next state
next_state = model.predict_next_state(x, actions)
print(f"Next state: {next_state.shape}")

# Plan actions to reach a goal
goal = torch.randn(2, 256)
planned = model.plan_actions(x, goal, num_steps=5, num_candidates=10)
print(f"Planned actions: {planned.shape}")
```

## CLI

```bash
# Show both models and parameter counts
hjepa info

# Train base JEPA-LM
hjepa train --preset tiny --epochs 5

# Run 5-way benchmark
hjepa bench

# Run all tests
hjepa test
```

## How JEPA-LM works

Traditional LLMs predict the next token left-to-right. JEPA-LM predicts **meaning** — the latent representation of masked text — in embedding space.

```
Input: "The cat sat on the [MASK] because it was [MASK]"
       |
+-----------------------------------------------------+
|  Bidirectional Encoder                              |
|  -> Latent representations for all tokens           |
+-----------------------------------------------------+
|  Predictor (narrow bottleneck)                      |
|  -> Predicts latents for masked spans               |
+-----------------------------------------------------+
|  EMA Target Encoder (stop-gradient)                 |
|  -> Stable targets, prevents embedding collapse     |
+-----------------------------------------------------+
|  JEPA Loss (cosine similarity in latent space)      |
+-----------------------------------------------------+
```

### What makes H-JEPA-LM different

H-JEPA-LM extends JEPA-LM with three additions:

1. **Hierarchical prediction** — Encodes at multiple levels (token details → semantic meaning). Predictor operates at each level with learned weights.
2. **Action conditioning** — Actions are encoded and fused into the highest-level predictions. The model learns to predict what happens when you take an action.
3. **World model planning** — Plans sequences of actions by rolling out predictions in latent space and selecting the best trajectory.

```
JEPA-LM:                              H-JEPA-LM:
  Encoder                                Hierarchical Encoder
    |                                      Level 0 (tokens) ──┐
    |                                      Level 1 (semantic) ─┤
  Predictor                               Hierarchical Predictor
    |                                      + Action Conditioning
  Target Encoder (EMA)                    Target Encoders (EMA, per level)
    |                                      + World Model Planning
  JEPA Loss                               Multi-level JEPA Loss
```

## Key innovation: JEPA as primary objective

```
┌──────────────────────────────────────────────────┐
│           Why this is different                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  LLM-JEPA (ICLR 2026):                          │
│    Existing LLM + bolted-on JEPA loss            │
│    -> JEPA is secondary (optional)               │
│    -> Causal attention (left-to-right)           │
│    -> Needs paired Text<->Code data              │
│                                                  │
│  JEPA-LM / H-JEPA-LM (This repo):               │
│    New architecture with JEPA as core             │
│    -> JEPA is PRIMARY objective                   │
│    -> Bidirectional attention                     │
│    -> Self-supervised via span masking            │
│    -> EMA target encoder prevents collapse        │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Benchmarks

<p align="center">
  <img src="benchmarks/diversity_benchmarks.png" width="100%">
</p>

| Model | Cosine Sim ↓ | Embed Std ↑ | SV Ratio ↓ | Params |
|:--|:--:|:--:|:--:|:--:|
| GPT (NTP) | 0.998 | 0.040 | 0.898 | 4.7M |
| BERT (MLM) | 0.850 | 0.297 | 0.576 | 8.7M |
| LLM-JEPA | 0.998 | 0.040 | 0.898 | 4.9M |
| **JEPA-LM** | 0.857 | 0.231 | 0.514 | 5.6M |
| **H-JEPA-LM** | **0.774** | **0.308** | **0.457** | 5.8M |

> H-JEPA-LM achieves **23% lower cosine similarity** than LLM-JEPA — embeddings are far more diverse and information-dense.

### Why these metrics matter

- **Cosine Similarity** — How similar embeddings are to each other. Lower = more diverse. GPT embeddings are nearly identical (0.998).
- **Embedding Std Dev** — Variation in embedding magnitudes. Higher = more information encoded.
- **SV Ratio** — Balance of embedding dimensions used. Lower = more balanced representations.

## Project structure

```
H-JEPA-LM/
├── jepalm/                    # Core package (pip installable)
│   ├── __init__.py           # Public API: JEPELM, HJEPELM, configs
│   ├── cli.py                # CLI entry points
│   │
│   ├── model.py              # JEPA-LM (base model)
│   ├── config.py             # JEPAConfig
│   ├── encoder.py            # Bidirectional encoder
│   ├── target_encoder.py     # EMA target encoder
│   ├── predictor.py          # Narrow predictor
│   ├── decoder.py            # Lightweight decoder
│   ├── loss.py               # JEPA + NTP loss
│   ├── masking.py            # Span masking
│   │
│   ├── hjepa.py              # H-JEPA-LM (hierarchical model)
│   │
│   ├── train.py              # Training loop
│   ├── dataset.py            # Dataset loading
│   └── eval.py               # Evaluation
├── tests/                     # 21 unit tests
├── benchmarks/               # Benchmark charts & scripts
├── .github/workflows/        # CI/CD
├── demo.ipynb                # Jupyter notebook demo
├── benchmark_hjepa.py        # 5-way comparison benchmark
├── train.py                  # Training entry point
├── test_model.py             # Quick smoke test
├── pyproject.toml            # Package configuration
├── LICENSE
└── README.md
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## References

- [I-JEPA](https://arxiv.org/abs/2301.08243) — Image-based JEPA (CVPR 2023)
- [V-JEPA](https://arxiv.org/abs/2404.08471) — Video-based JEPA (2024)
- [LLM-JEPA](https://arxiv.org/abs/2502.16982) — Text JEPA bolted onto existing LLMs (ICLR 2026)
- **JEPA-LM / H-JEPA-LM (Ours)** — JEPA as primary objective for text

## License

MIT — contributions welcome.
