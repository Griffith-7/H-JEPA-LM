# PLAN — Building a Latent-Space JEPA LLM

## Phase 1: Research & Understanding ✅ COMPLETE

- [x] Study I-JEPA paper & architecture (CVPR 2023)
- [x] Study V-JEPA paper & architecture (2024)
- [x] Study LLM-JEPA paper & code (ICLR 2026)
- [x] Study latent space text models (Token Assorted, LatentLM, LangVAE, GQ-VAE)
- [x] Identify universal JEPA principles
- [x] Document all insights in MEMORY.md

---

## Phase 2: Architecture Design (NEXT)

### Target Architecture: JEPA-LM

```
┌─────────────────────────────────────────────────────┐
│                  TRAINING PHASE                      │
│                                                      │
│  Input: "The cat sat on the mat because it was"      │
│         ↓                                            │
│  [MASK] spans randomly (e.g., 2 spans of 3-5 tokens)│
│         ↓                                            │
│  ┌──────────────────────────────────────────┐        │
│  │ BIDIRECTIONAL ENCODER (BERT-style)       │        │
│  │ - Processes masked input                  │        │
│  │ - Outputs latent representation           │        │
│  │ - Only sees unmasked tokens               │        │
│  └──────────────────────────────────────────┘        │
│         ↓                                            │
│  ┌──────────────────────────────────────────┐        │
│  │ EMA TARGET ENCODER (same architecture)   │        │
│  │ - Processes FULL unmasked input           │        │
│  │ - Updated via EMA (no gradients)          │        │
│  │ - Produces stable prediction targets      │        │
│  └──────────────────────────────────────────┘        │
│         ↓                                            │
│  ┌──────────────────────────────────────────┐        │
│  │ PREDICTOR (narrow transformer)           │        │
│  │ - Takes: encoder output + mask tokens     │        │
│  │ - Predicts: latent representations of     │        │
│  │   masked spans                            │        │
│  │ - Much smaller than encoder               │        │
│  └──────────────────────────────────────────┘        │
│         ↓                                            │
│  JEPA LOSS: cosine(Predicted latent, Target latent)  │
│                                                      │
│  ┌──────────────────────────────────────────┐        │
│  │ LIGHTWEIGHT DECODER (optional, weak)     │        │
│  │ - Maps predicted latents → text tokens    │        │
│  │ - Trained with weak NTP loss              │        │
│  │ - Used only for text generation           │        │
│  └──────────────────────────────────────────┘        │
│         ↓                                            │
│  WEAK NTP LOSS: reconstruct masked tokens            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                 INFERENCE PHASE                       │
│                                                      │
│  Input: "The cat sat on the"                         │
│         ↓                                            │
│  Encoder → latent state                              │
│         ↓                                            │
│  Predictor (with position tokens for next span)      │
│         ↓                                            │
│  Predicted latent → Decoder → "mat because it was"   │
│                                                      │
│  (Or: latent → latent planning/reasoning → decode)   │
└─────────────────────────────────────────────────────┘
```

### Component Specifications

#### 1. Bidirectional Encoder
- **Type:** Transformer (BERT-style)
- **Size:** 12 layers, 768 dim, 12 heads (~125M params)
- **Input:** Tokenized text with [MASK] spans
- **Output:** Hidden states for unmasked tokens
- **Key:** Bidirectional attention (not causal)

#### 2. EMA Target Encoder
- **Same architecture as encoder**
- **Updated via:** θ̄ ← m·θ̄ + (1-m)·θ
- **Momentum schedule:** 0.996 → 1.0 (linear)
- **Stop-gradient:** No gradients flow through this
- **Processes:** Full unmasked text

#### 3. Predictor (Narrow)
- **Type:** Transformer (much smaller)
- **Size:** 6 layers, 384 dim, 12 heads (~30M params)
- **Input:** Encoder output + mask tokens with positional embeddings
- **Output:** Predicted latent vectors for masked positions
- **Key:** Bottleneck forces encoder to learn good representations

#### 4. Lightweight Decoder (Optional)
- **Type:** Small autoregressive transformer
- **Size:** 6 layers, 768 dim (~50M params)
- **Input:** Predicted latent vectors
- **Output:** Text tokens for masked spans
- **Key:** Only used for text generation, not primary training

### Training Details

#### Masking Strategy (inspired by I-JEPA)
- **Span masking:** Random spans of 3-15 tokens
- **Number of spans:** 2-4 per sequence
- **Total masked:** ~40-60% of tokens
- **Context:** Remaining unmasked tokens

#### Loss Function
```
L_total = λ₁ · L_JEPA + λ₂ · L_NTP

L_JEPA = cosine_similarity(Pred(Enc(masked_input)), Enc(full_input))
         → maximize this (or minimize 1 - cosine)

L_NTP = cross_entropy(Decoder(predicted_latents), masked_tokens)
         → standard next-token prediction on masked spans only

λ₁ = 1.0 (primary)
λ₂ = 0.1 (weak, secondary)
```

#### Collapse Prevention
- EMA target encoder with momentum schedule
- Stop-gradient on target encoder outputs
- Narrow predictor bottleneck
- Layer normalization on target embeddings

### Datasets for Training

| Dataset | Type | Size | Use |
|---------|------|------|-----|
| Wikipedia | Plain text | 16GB | Pretraining |
| OpenWebText | Plain text | 8GB | Pretraining |
| CodeSearchNet | NL↔Code pairs | — | Fine-tuning (baseline comparison) |
| GSM8K | Math QA | — | Evaluation |
| Spider | NL→SQL | — | Evaluation |

---

## Phase 3: Minimal Prototype ✅ COMPLETE

- [x] Implement in PyTorch
- [x] Components:
  - [x] BidirectionalEncoder (BERT-like)
  - [x] EMATargetEncoder (EMA copy)
  - [x] Predictor (narrow transformer)
  - [x] LightweightDecoder (small GPT-like)
- [x] Training loop with:
  - [x] Span masking
  - [x] JEPA loss (cosine similarity)
  - [x] Weak NTP loss on decoder
  - [x] EMA update
- [x] Start with tiny config (~1.4M params)
- [x] Test on synthetic data — forward/backward pass works
- [x] Loss decreasing: 1.6975 → 1.5760 in 24 steps

---

## Phase 4: Evaluation & Comparison

- [ ] Train standard BERT/GPT (same size) as baseline
- [ ] Compare:
  - [ ] Embedding quality (linear probing)
  - [ ] Masked span prediction accuracy
  - [ ] Text generation quality (BLEU, ROUGE)
  - [ ] Reasoning tasks (GSM8K)
  - [ ] Generalization (train on less data)
  - [ ] Overfitting resistance
  - [ ] Representation similarity to LLM-JEPA
- [ ] Visualize latent space (t-SNE, clustering)
- [ ] Write up results

---

## Phase 5: Iterate & Scale

- [ ] Fix issues found in Phase 4
- [ ] Scale to 350M-1B params
- [ ] Try more datasets
- [ ] Explore latent-space reasoning (plan in latent space, decode to text)
- [ ] Compare against LLM-JEPA directly
- [ ] Paper/blog post if results are strong

---

## Timeline

| Phase | Est. Duration | Status |
|-------|--------------|--------|
| Phase 1 | 1 week | ✅ COMPLETE |
| Phase 2 | 1-2 weeks | ✅ COMPLETE |
| Phase 3 | 3-4 weeks | ✅ COMPLETE |
| Phase 4 | 2-3 weeks | NEXT |
| Phase 5 | Ongoing | PENDING |

## Resources Needed

- GPU: 1x A100 or equivalent (for 125M-500M experiments)
- Python, PyTorch, HuggingFace Transformers
- Datasets: Wikipedia, OpenWebText
- Reference code: I-JEPA, V-JEPA, LLM-JEPA repos
