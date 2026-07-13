"""Configuration system for JEPA-LM."""

from dataclasses import dataclass, field


@dataclass
class JEPAConfig:
    # Encoder
    enc_vocab_size: int = 30522
    enc_hidden_dim: int = 768
    enc_num_layers: int = 12
    enc_num_heads: int = 12
    enc_ff_dim: int = 3072
    enc_dropout: float = 0.1
    enc_max_seq_len: int = 512

    # Target Encoder (same as encoder, updated via EMA)
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0

    # Predictor (narrow)
    pred_hidden_dim: int = 384
    pred_num_layers: int = 6
    pred_num_heads: int = 12
    pred_ff_dim: int = 1536
    pred_dropout: float = 0.1

    # Decoder (lightweight)
    dec_hidden_dim: int = 768
    dec_num_layers: int = 6
    dec_num_heads: int = 12
    dec_ff_dim: int = 3072
    dec_dropout: float = 0.1

    # Masking
    mask_prob: float = 0.15
    mean_span_length: int = 3
    num_spans_to_sample: int = 3
    mask_token_id: int = 103  # [MASK] token
    pad_token_id: int = 0

    # Tokenizer
    tokenizer_name: str = "bert-base-uncased"

    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    max_samples: int = 50000

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    lambda_jepa: float = 1.0
    lambda_ntp: float = 0.1

    # Evaluation
    eval_samples: int = 500
    linear_probe_dim: int = 256

    # General
    device: str = "cuda"
    seed: int = 42
    fp16: bool = True
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    output_dir: str = "./checkpoints"
