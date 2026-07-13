"""Loss functions for JEPA-LM training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def jepa_loss(predicted, target, mask_valid=None):
    """Compute JEPA loss: cosine similarity between predicted and target latents.

    Maximizes cosine similarity (minimizes 1 - cosine_similarity).

    Args:
        predicted: (B, T, D) — predicted latent vectors from predictor
        target: (B, T, D) — target latent vectors from EMA encoder
        mask_valid: (B, T) — boolean mask for valid positions

    Returns:
        loss: scalar
        cosine_sim: mean cosine similarity for logging
    """
    # Normalize
    pred_norm = F.normalize(predicted, p=2, dim=-1)
    tgt_norm = F.normalize(target, p=2, dim=-1)

    # Cosine similarity per position
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (B, T)

    if mask_valid is not None:
        # Mask out invalid positions
        cos_sim = cos_sim * mask_valid.float()
        loss = (1.0 - cos_sim).sum() / mask_valid.float().sum().clamp(min=1.0)
        mean_sim = cos_sim.sum() / mask_valid.float().sum().clamp(min=1.0)
    else:
        loss = (1.0 - cos_sim).mean()
        mean_sim = cos_sim.mean()

    return loss, mean_sim


def ntp_loss(logits, labels):
    """Compute next-token prediction loss on masked tokens.

    Args:
        logits: (B, S, vocab_size) — decoder output
        labels: (B, S) — target token ids (-100 for ignored positions)

    Returns:
        loss: scalar
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # Reshape for cross entropy
    B, S, V = logits.shape
    loss = loss_fn(logits.reshape(-1, V), labels.reshape(-1))
    return loss


def total_loss(jepa_loss_val, ntp_loss_val, lambda_jepa=1.0, lambda_ntp=0.1):
    """Combine JEPA and NTP losses.

    Args:
        jepa_loss_val: scalar — JEPA loss
        ntp_loss_val: scalar — NTP loss (or 0 if no decoder)
        lambda_jepa: weight for JEPA loss
        lambda_ntp: weight for NTP loss

    Returns:
        total: scalar — combined loss
    """
    return lambda_jepa * jepa_loss_val + lambda_ntp * ntp_loss_val
