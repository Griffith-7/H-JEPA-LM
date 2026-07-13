# JEPA-LM: Joint-Embedding Predictive Language Model

A **fundamentally different language model** that learns by predicting latent representations, not tokens.

## What is JEPA-LM?

Traditional LLMs (GPT, Llama) predict the next token left-to-right. JEPA-LM predicts **meaning** — the latent representation of masked text spans — in embedding space.

### Key Innovation: JEPA as PRIMARY objective
- **JEPA-LM (Ours)**: JEPA is the core design, text generation is secondary
- **LLM-JEPA (ICLR 2026)**: JEPA is bolted on as secondary loss to existing LLM

## Architecture

```
Input: "The cat sat on the [MASK] because it was [MASK]"
       ↓
Hierarchical Encoder (bidirectional) → multi-level latent representations
       ↓
Hierarchical Predictor → predicted latents for masked spans
       ↓
EMA Target Encoder → stable target latents
       ↓
Multi-level JEPA Loss (cosine similarity)
       ↓
Action Conditioning → predict consequences of actions
       ↓
World Model → plan actions in latent space
```

## Features

- **Hierarchical JEPA (H-JEPA)**: Multi-level prediction (token details → semantic meaning)
- **Action Conditioning**: Predict what happens if you take an action
- **World Model**: Plan sequences of actions in latent space
- **EMA Target Encoder**: Prevents embedding collapse
- **Self-supervised**: No paired data needed — masking creates the two views

## Results

| Metric | GPT-NTP | BERT-MLM | LLM-JEPA | **JEPA-LM** | **H-JEPA-LM** |
|--------|---------|----------|----------|-------------|---------------|
| Cosine Sim ↓ | 0.998 | 0.850 | 0.998 | 0.857 | **0.774** |
| Embed Std ↑ | 0.040 | 0.297 | 0.040 | 0.231 | **0.308** |
| SV Ratio ↓ | 0.898 | 0.576 | 0.898 | 0.514 | **0.457** |

**H-JEPA-LM has the most diverse embeddings** — cosine similarity 0.774 vs GPT's 0.998.

## Installation

```bash
pip install torch>=2.0.0
pip install transformers datasets
```

## Quick Start

```python
from jepalm.model import JEPELM
from jepalm.config import JEPAConfig

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
```

## Training

```bash
# Train JEPA-LM on Wikipedia
python train.py --preset small --dataset wikitext --max_samples 10000

# Run 5-way benchmark
python benchmark_hjepa.py
```

## Project Structure

```
JEPA-LM/
├── jepalm/                    # Core package
│   ├── model.py              # Main JEPELM model
│   ├── config.py             # Configuration
│   ├── encoder.py            # Bidirectional encoder
│   ├── target_encoder.py     # EMA target encoder
│   ├── predictor.py          # Narrow predictor
│   ├── decoder.py            # Lightweight decoder
│   ├── loss.py               # JEPA + NTP loss
│   ├── masking.py            # Span masking
│   ├── train.py              # Training loop
│   ├── dataset.py            # Dataset loading
│   └── eval.py               # Evaluation
├── benchmark_hjepa.py        # 5-way comparison benchmark
├── train.py                  # Training entry point
├── test_model.py             # Quick smoke test
└── requirements.txt
```

## How It Differs

| Aspect | LLM-JEPA (ICLR 2026) | JEPA-LM (Ours) |
|--------|----------------------|----------------|
| Architecture | Existing LLM + extra loss | New architecture |
| JEPA Role | Secondary (optional) | Primary (core design) |
| Target Encoder | Same model (symmetric) | EMA copy (asymmetric) |
| Collapse Prevention | None | EMA + stop-gradient |
| Masking | None (needs paired data) | Span masking (self-supervised) |
| Attention | Causal (left-to-right) | Bidirectional |
| Data Required | Paired Text↔Code | Raw text only |

## References

- I-JEPA (CVPR 2023): Image-based JEPA
- V-JEPA (2024): Video-based JEPA
- LLM-JEPA (ICLR 2026): Text JEPA (bolt-on to existing LLM)
- **JEPA-LM (Ours)**: JEPA as primary objective for text

## License

MIT
