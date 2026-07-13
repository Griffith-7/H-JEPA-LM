# TARGET — Latent-Space JEPA LLM

## Vision

Build a **fundamentally different LLM** called **JEPA-LM** that operates primarily in **latent/embedding space** rather than token space. Text generation becomes a secondary decoder output, not the core training objective.

## Why This Matters

Current LLMs (GPT, Llama, etc.) are fundamentally **input-space reconstructors** — they predict raw tokens. This wastes compute on surface-level details and limits abstract reasoning. A latent-space JEPA LLM would:

- Learn **meaning** not **surfaces**
- Be more sample-efficient
- Generalize better
- Resist overfitting
- Enable true planning/reasoning in latent space

## Core Architecture

```
Input Text → [MASK spans] → Bidirectional Encoder → Latent State
                                    ↓
                              JEPA Predictor → Predicted Latents
                                    ↓
                          EMA Target Encoder → Target Latents
                                    ↓
                            JEPA Loss (cosine)
                                    ↓
                          Lightweight Decoder → Text
                                    ↓
                           Weak NTP Loss
```

## Key Differences from Current LLM-JEPA

| Aspect | LLM-JEPA (ICLR 2026) | Our JEPA-LM |
|--------|----------------------|-------------|
| Architecture | Same transformer | New architecture |
| JEPA role | Bolt-on loss | Primary objective |
| Target encoder | Same model | EMA copy (asymmetric) |
| Collapse prevention | None needed (has NTP) | EMA + stop-gradient |
| Masking | None (paired data views) | Span masking |
| Training signal | Paired data (Text↔Code) | Self-supervised masking |
| Decoder | None (uses LLM head) | Lightweight decoder |
| Bidirectional? | No (causal) | Yes (encoder is bidirectional) |

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Literature review & architecture analysis | ✅ DONE |
| 2 | Design JEPA-LM architecture | IN PROGRESS |
| 3 | Implement minimal prototype (125M) | PENDING |
| 4 | Train on small dataset & measure | PENDING |
| 5 | Compare against BERT/GPT baseline | PENDING |
| 6 | Compare against LLM-JEPA | PENDING |
| 7 | Scale to 350M-1B | PENDING |
| 8 | Explore latent-space reasoning | PENDING |

## Success Criteria

- [ ] Model learns meaningful latent representations via JEPA loss
- [ ] Decoder can reconstruct text from predicted latents
- [ ] Outperforms or matches same-size BERT/GPT on reasoning tasks
- [ ] Shows better generalization with less data
- [ ] Resists overfitting better than standard models
- [ ] Latent space shows meaningful clustering/structure

## Open Questions to Resolve

1. **Masking ratio**: How much to mask? (40%? 60%? 80%?)
2. **Predictor size**: How narrow? (Too narrow = underfit, too wide = no bottleneck)
3. **Decoder role**: How much weight should NTP loss get?
4. **Latent dimension**: 768? 512? 256?
5. **EMA schedule**: How fast should target encoder evolve?
6. **Inference**: How to generate text autoregressively from latent predictions?
