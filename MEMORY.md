# MEMORY — Project Tracker

## What We're Building

A **latent-space JEPA LLM** — fundamentally different from standard LLMs. JEPA as primary objective, text generation as secondary decoder.

## Current Status

**Phase:** Evaluation & Comparison (Phase 4) — IN PROGRESS
**Date Started:** 2026-07-13
**Latest Experiment:** EXP-006 (scaled 3-way benchmark, 512-dim)
**Key Result:** JEPA-LM comparable to GPT at 512-dim scale. BERT-MLM wins at this scale. Need BERT-base (125M+).

---

## RESEARCH FINDINGS (Deep Dive)

### 1. I-JEPA (Image-based JEPA) — CVPR 2023

**Architecture:**
- Context Encoder (ViT): processes ONLY visible patches
- Target Encoder (ViT, EMA copy): processes FULL image
- Predictor (narrow ViT): maps context embeddings → target embeddings
- Mask tokens: learnable vectors + positional embeddings for target locations

**Key Mechanics:**
- Predicts in EMBEDDING space, not pixel space
- Multi-block masking: 4 target blocks, each ~15-20% of image
- Context block: single block, ~85-100% of image
- EMA update: θ̄ ← m·θ̄ + (1-m)·θ, momentum 0.996→1.0
- Stop-gradient on target encoder outputs
- Loss: smooth-L1 between predicted and target embeddings

**Why it works:**
- Predicting large semantic blocks → forces learning meaning, not pixels
- EMA + stop-gradient → prevents collapse (target evolves slowly)
- Narrow predictor → forces encoder to learn informative representations
- No data augmentations needed (masking is the only signal)

**Code:** https://github.com/facebookresearch/ijepa
**Minimal implementation:** https://github.com/keon/jepa/blob/main/ijepa.py

### 2. V-JEPA (Video JEPA) — 2024

**Architecture:**
- Same as I-JEPA but with 3D tubelets instead of 2D patches
- Conv3d patchification: (2 frames, 16×16 pixels) tubelets
- 3D positional embeddings (temporal + spatial)

**Key Differences from I-JEPA:**
- Tube masking: spatial block held constant across ALL temporal frames
- Two mask groups:
  - Short-range: 8 masks, scale 0.15 (easy, local)
  - Long-range: 2 masks, scale 0.7 (hard, global)
- ~90% masking ratio (context encoder sees only ~10%)
- L1 loss (not smooth-L1) — more stable
- Multi-masking: 2 masks per video, target computed once

**Why tube masking matters:**
- Kills temporal shortcut (can't just copy from previous frame)
- Forces spatial reasoning across time

**Results:** 81.9% Kinetics-400, 72.2% SSv2 with frozen backbone

**Code:** https://github.com/facebookresearch/jepa
**Minimal:** https://github.com/keon/jepa/blob/main/vjepa.py

### 3. LLM-JEPA (Text) — ICLR 2026

**Architecture:**
- Same transformer as standard LLM (Llama, Gemma, etc.)
- Added JEPA loss: cosine similarity between Text and Code embeddings
- Embedding = last token hidden state from last layer
- [PRED] tokens appended for prediction (weight-tied)
- Two extra forward passes for Text and Code encoding
- Additive mask optimization: combines into 1 forward pass

**Why it's NOT fundamentally different:**
- It's the SAME architecture with an EXTRA loss term
- Next-token prediction is still the primary objective
- JEPA is bolted on, not the core design
- No EMA target encoder (uses same model for both views)
- No masking strategy (relies on paired data views)

**Key Results:**
- Llama3.2-1B: 37.0% → 51.6% on SYNTH (+14.6)
- Llama3.2-1B: 55.2% → 70.9% on Spider (+15.7)
- Llama3.2-1B: 51.5% → 71.8% on GSM8K (+20.3)
- Resists overfitting, no generative degradation
- InfoNCE fails (34.4%), cosine works (71.5%)

**Code:** https://github.com/galilai-group/llm-jepa

### 4. Latent Space Text Models (Related Work)

**Token Assorted (2025):**
- VQ-VAE compresses CoT tokens into discrete latent codes
- Partial replacement: first m tokens → latent, rest → text
- Random mixing of latent/text ratios during training
- 17% shorter reasoning traces, +4-13% accuracy

**LatentLM (2025):**
- VAE encodes continuous data → latent vectors
- Next-token diffusion: autoregressively predicts latent vectors
- Same transformer handles discrete (text) and continuous (latent) tokens
- σ-VAE prevents variance collapse

**LangVAE (2025):**
- Modular VAE on top of pre-trained LLMs
- KV-cache injection for latent → decoder communication
- Better disentanglement of syntactic/semantic features

**GQ-VAE (2025):**
- Variable-length discrete tokenization
- Learned tokenization beats fixed-length VQ-VAE
- More learnable for downstream LLMs than BPE at same compression

---

## ARCHITECTURE INSIGHTS FOR OUR MODEL

### What makes JEPA work (universal principles):
1. **Predict in latent space, not input space** — discard irrelevant details
2. **EMA target encoder + stop-gradient** — prevents collapse
3. **Narrow predictor** — forces encoder to learn informative representations
4. **Masking as training signal** — no hand-crafted augmentations needed
5. **Cosine similarity or L1** — InfoNCE/contrastive fails

### Why current LLM-JEPA is limited:
1. Uses same model for both views (no asymmetry)
2. No EMA target encoder
3. No masking strategy
4. JEPA is secondary to next-token prediction
5. Requires paired data (Text↔Code)

### Our opportunity — what a truly different JEPA LLM would look like:
1. **Encoder**: maps text → latent representation (separate from decoder)
2. **JEPA Predictor**: predicts future/missing latent states
3. **EMA Target Encoder**: slow-moving copy for stable targets
4. **Decoder**: maps latents → text (secondary, lightweight)
5. **Primary objective**: latent-space prediction (JEPA loss)
6. **Secondary objective**: text reconstruction (weak NTP loss)
7. **Masking strategy**: mask spans of text, predict their latent representations
8. **No paired data required** — masking creates the two views

### Potential architecture:

```
Input: "The cat sat on the [MASK] because it was [MASK]"
         ↓
Encoder (bidirectional, like BERT): → latent representations
         ↓
Mask tokens + positional → Predictor → Predicted latents
         ↓
Target Encoder (EMA copy, sees full text) → Target latents
         ↓
JEPA Loss: cosine(predicted_latent, target_latent)
         ↓
Decoder (lightweight, autoregressive): latent → text
         ↓
Weak NTP Loss: reconstruct masked tokens
```

**Key difference from standard LLM:**
- Standard LLM: predict tokens left-to-right
- Ours: predict latent representations of masked spans, decode to text

---

## FILES STATUS

| File | Status |
|------|--------|
| `TARGET.md` | Created |
| `MEMORY.md` | Updated with all findings |
| `PLAN.md` | Created |
| `EXPERIMENTS.md` | Updated with EXP-006 |
| `benchmark.py` | 3-way comparison (40M scale) |
| `benchmark_scaled.py` | 3-way comparison (512-dim scale) |
| `benchmark_results/` | Results JSON (EXP-005) |
| `benchmark_results_scaled/` | Results JSON + checkpoints (EXP-006) |

## NEXT STEPS

1. **Scale to 125M+ params (BERT-base)** — 768-dim, 12 layers
2. **Use FULL Wikipedia** — more pretraining data
3. **Train 20+ epochs** — need convergence
4. Test on GLUE/SuperGLUE — proper NLU benchmarks
5. Explore latent-space reasoning — plan/predict in embedding space

## EXPERIMENT RESULTS (Latest)

### EXP-006: Scaled 3-Way Benchmark (512-dim, 8 layers)
**Setup:** 41-68M params, 10000 Wikipedia articles, 10 epochs
**GPU:** NVIDIA RTX 3050 Laptop (4.3 GB VRAM)

| Metric | GPT-NTP (41M) | BERT-MLM (57M) | JEPA-LM (68M) | Winner |
|--------|--------------|----------------|---------------|--------|
| SST-2 Classification | 0.509 | 0.513 | 0.512 | BERT-MLM |
| Embedding Cosine Sim | 0.997 | **0.550** | 0.827 | BERT-MLM |
| Reasoning Top-1 | 0.021 | **0.043** | 0.021 | BERT-MLM |
| Sample Eff 100% | 0.509 | **0.525** | 0.509 | BERT-MLM |

**Verdict:** JEPA-LM comparable to GPT-NTP (3/7 wins). BERT-MLM wins at this scale. All models get ~50% on SST-2 (random). Need BERT-base scale.

### EXP-005: Small 3-Way Benchmark (256-dim, 4 layers)
**Setup:** ~40M params, 3000 Wikipedia articles, 10 epochs
- JEPA-LM beats GPT on 4/8 tests
- JEPA-LM has 10x more diverse embeddings than GPT (cosine sim 0.72 vs 0.999)
- All models get ~50% on SST-2 (random)

### EXP-004: Small JEPA (113M) — 5 epochs
- Cosine Sim: 0.005 → **0.554** (+11000%)
- JEPA Loss: 1.43 → 1.21 (-16%)

### EXP-003: Tiny JEPA (40M) — 5 epochs
- Cosine Sim: 0.017 → 0.239 (+1300%)
- JEPA Loss: 1.13 → 0.875 (-23%)

### Key Finding
**JEPA-LM prevents collapse better than GPT (0.72-0.83 vs 0.997-0.999) but BERT-MLM has the most diverse embeddings at scale (0.55). All models need 125M+ params to solve downstream tasks.**

## CODE IMPLEMENTATION (Phase 2+3 Complete)

### Project Structure
```
experment 1/
├── jepalm/
│   ├── __init__.py          # Package init
│   ├── config.py            # JEPAConfig dataclass
│   ├── encoder.py           # BidirectionalEncoder (BERT-style)
│   ├── target_encoder.py    # EMATargetEncoder (EMA copy)
│   ├── predictor.py         # Predictor (narrow transformer)
│   ├── decoder.py           # LightweightDecoder (autoregressive)
│   ├── masking.py           # Span masking strategy
│   ├── loss.py              # JEPA + NTP loss functions
│   ├── model.py             # Main JEPELM model
│   ├── train.py             # Training loop
│   ├── dataset.py           # Real text dataset (HuggingFace)
│   ├── baseline.py          # Baseline MLM model for comparison
│   └── eval.py              # Evaluation metrics & visualization
├── train.py                 # Main entry point
├── test_model.py            # Quick test script
├── run_experiment.py        # Full experiment pipeline
├── requirements.txt
├── EXPERIMENTS.md           # Experiment log
├── TARGET.md, MEMORY.md, PLAN.md
```

### Experiment Results (EXP-003, EXP-004)
- Tiny (40M): CosSim 0.24, 5 epochs, loss still decreasing
- Small (113M): CosSim **0.55**, 5 epochs, excellent latent prediction
- JEPA-LM consistently produces more diverse embeddings than baseline
- Both models train stably, no collapse detected
- Output directories: `./experiment_tiny_5ep/`, `./experiment_small_5ep/`

### How to Run
```bash
# Quick test
python test_model.py

# Full experiment (synthetic data)
python run_experiment.py --preset tiny --dataset synthetic --max_samples 500

# Full experiment (Wikipedia)
python run_experiment.py --preset tiny --dataset wikitext --max_samples 10000

# Train only JEPA-LM
python train.py --preset tiny --dataset wikitext
```

## IMPORTANT DECISIONS

| Decision | Choice | Reason |
|----------|--------|--------|
| Use existing LLM-JEPA code? | Yes, as reference/baseline | Good baseline for comparison |
| Start small? | Yes (100M-500M params) | Faster iteration |
| Training data | Wikipedia + masked spans | No paired data needed |
| Architecture base | BERT-style encoder + GPT-style decoder | Bidirectional encoding, autoregressive decoding |
| Collapse prevention | EMA + stop-gradient | Proven in I-JEPA/V-JEPA |
| Primary loss | Cosine similarity in latent space | Works best in LLM-JEPA experiments |
| Secondary loss | Weak next-token prediction | Maintain generative capability |

## REMINDERS

- Don't just bolt JEPA onto a transformer — redesign the core objective
- Keep tracking everything in these files
- Run comparisons against standard LLM baseline at every stage
- EMA + stop-gradient is CRITICAL for collapse prevention
- Narrow predictor forces encoder to learn meaningful representations
- Masking strategy matters — large semantic spans, not small tokens
