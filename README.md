<p align="center">
  <img src="benchmarks/architecture_overview.png" width="100%">
</p>

<h1 align="center">H-JEPA-LM</h1>

<p align="center">
  <strong>Joint-Embedding Predictive Language Model</strong><br>
  A fundamentally different approach to language modeling — predicting latent representations, not tokens.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch 2.0+"></a>
  <a href="/Griffith-7/JEPA-LM/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/Griffith-7/JEPA-LM/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <a href="https://github.com/Griffith-7/JEPA-LM/actions"><img src="https://github.com/Griffith-7/JEPA-LM/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

## What makes JEPA-LM different

Traditional LLMs predict the next token. JEPA-LM predicts **meaning** — the latent representation of masked text — in embedding space. This produces dramatically more diverse and information-rich representations.

| Capability | GPT (NTP) | BERT (MLM) | LLM-JEPA | **JEPA-LM** | **H-JEPA-LM** |
|:--|:--:|:--:|:--:|:--:|:--:|
| Cosine Similarity ↓ | 0.998 | 0.850 | 0.998 | 0.857 | **0.774** |
| Embedding Std Dev ↑ | 0.040 | 0.297 | 0.040 | 0.231 | **0.308** |
| SV Ratio ↓ | 0.898 | 0.576 | 0.898 | 0.514 | **0.457** |

> **H-JEPA-LM achieves 23% lower cosine similarity than LLM-JEPA** — embeddings are far more diverse and information-dense.

## Install

```bash
pip install git+https://github.com/Griffith-7/JEPA-LM.git
```

Or clone and install locally:

```bash
git clone https://github.com/Griffith-7/JEPA-LM.git
cd JEPA-LM
pip install -e ".[dev]"
```

## Quick start

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
params = model.count_parameters()
print(f"Parameters: {params['total']:,}")

# Forward pass
import torch
x = torch.randint(0, 1000, (2, 32))
out = model(x)
print(f"JEPA loss: {out['jepa_loss']:.4f}")
print(f"Cosine similarity: {out['cosine_similarity']:.4f}")
```

## CLI

```bash
# Show model info
hjepa info

# Train with preset
hjepa train --preset tiny --epochs 5

# Run benchmark
hjepa bench

# Smoke test
hjepa test
```

## Key innovation: JEPA as primary objective

```
┌──────────────────────────────────────────────────┐
│              JEPA-LM vs LLM-JEPA                 │
├──────────────────────────────────────────────────┤
│                                                  │
│  LLM-JEPA (ICLR 2026):                          │
│    Existing LLM + bolted-on JEPA loss            │
│    → JEPA is secondary (optional)                │
│    → Causal attention (left-to-right)            │
│    → Needs paired Text<->Code data               │
│                                                  │
│  JEPA-LM (Ours):                                 │
│    New architecture with JEPA as core             │
│    → JEPA is PRIMARY objective                    │
│    → Bidirectional attention                      │
│    → Self-supervised via span masking             │
│    → EMA target encoder prevents collapse         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Architecture

```
Input: "The cat sat on the [MASK] because it was [MASK]"
       |
+-----------------------------------------------------+
|  Hierarchical Encoder (bidirectional)                |
|  -> Multi-level latent representations               |
+-----------------------------------------------------+
|  Hierarchical Predictor (narrow bottleneck)          |
|  -> Predicts latents for masked spans                |
+-----------------------------------------------------+
|  EMA Target Encoder (stop-gradient)                  |
|  -> Stable targets, prevents collapse                |
+-----------------------------------------------------+
|  Multi-level JEPA Loss (cosine similarity)           |
+-----------------------------------------------------+
|  Action Conditioning + World Model Planning          |
|  -> Plan actions in latent space (K=10 rollouts)     |
+-----------------------------------------------------+
```

## Benchmarks

<p align="center">
  <img src="benchmarks/diversity_benchmarks.png" width="100%">
</p>

<p align="center">
  <img src="benchmarks/cosine_similarity.png" width="80%">
</p>

| Model | Cosine Sim ↓ | Embed Std ↑ | SV Ratio ↓ | Params |
|:--|:--:|:--:|:--:|:--:|
| GPT (NTP) | 0.998 | 0.040 | 0.898 | 4.7M |
| BERT (MLM) | 0.850 | 0.297 | 0.576 | 8.7M |
| LLM-JEPA | 0.998 | 0.040 | 0.898 | 4.9M |
| JEPA-LM | 0.857 | 0.231 | 0.514 | 5.6M |
| **H-JEPA-LM** | **0.774** | **0.308** | **0.457** | 5.8M |

<p align="center">
  <img src="benchmarks/parameter_comparison.png" width="70%">
</p>

### Why these metrics matter

- **Cosine Similarity** — How similar embeddings are to each other. Lower = more diverse. GPT embeddings are nearly identical (0.998).
- **Embedding Std Dev** — Variation in embedding magnitudes. Higher = more information encoded.
- **SV Ratio** — Balance of embedding dimensions used. Lower = more balanced representations.

## How it works

1. **Hierarchical prediction** — Predict at two levels: token-level details and semantic-level meaning. The narrow bottleneck forces abstraction.
2. **EMA target encoder** — A slow-moving copy provides stable targets. Stop-gradient prevents collapse to trivial solutions.
3. **Action conditioning** — Actions are encoded and fused into the highest-level predictions.
4. **World model planning** — Plans action sequences by rolling out in latent space (K=10 random rollouts).

## Project structure

```
JEPA-LM/
├── jepalm/                    # Core package
│   ├── __init__.py           # Public API exports
│   ├── cli.py                # CLI entry points
│   ├── model.py              # Main JEPELM model
│   ├── config.py             # Configuration dataclass
│   ├── encoder.py            # Bidirectional encoder
│   ├── target_encoder.py     # EMA target encoder
│   ├── predictor.py          # Narrow predictor
│   ├── decoder.py            # Lightweight decoder
│   ├── loss.py               # JEPA + NTP loss
│   ├── masking.py            # Span masking
│   ├── train.py              # Training loop
│   ├── dataset.py            # Dataset loading
│   └── eval.py               # Evaluation
├── tests/                     # Unit tests (14 tests)
│   └── test_core.py
├── benchmarks/               # Benchmark charts & scripts
│   ├── diversity_benchmarks.png
│   ├── cosine_similarity.png
│   ├── architecture_overview.png
│   ├── parameter_comparison.png
│   └── generate_charts.py
├── .github/workflows/        # CI/CD
│   └── ci.yml
├── demo.ipynb                # Jupyter notebook demo
├── benchmark_hjepa.py        # 5-way comparison benchmark
├── hjepa_model.py            # H-JEPA-LM with action conditioning
├── train.py                  # Training entry point
├── test_model.py             # Quick smoke test
├── pyproject.toml            # Package configuration
├── requirements.txt
└── LICENSE
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
- **JEPA-LM (Ours)** — JEPA as primary objective for text

## License

MIT — contributions welcome.
