"""Span masking strategy for JEPA-LM training."""

import random
import torch


def create_span_mask(input_ids, mask_prob=0.15, mean_span_length=3,
                     num_spans_to_sample=3, mask_token_id=103, pad_token_id=0):
    """Create span masks for JEPA-LM training.

    Randomly samples spans to mask, returning:
    - masked_input_ids: input with [MASK] tokens replacing masked spans
    - mask_labels: original tokens at masked positions (for NTP loss)
    - mask_positions: indices of masked positions (for predictor)
    - unmasked_ids: original input_ids (for target encoder)

    Args:
        input_ids: (B, N) — token ids
        mask_prob: fraction of tokens to mask
        mean_span_length: average span length (Poisson distribution)
        num_spans_to_sample: number of spans to mask per sequence
        mask_token_id: token id for [MASK]
        pad_token_id: token id for padding

    Returns:
        dict with masked_input_ids, mask_labels, mask_positions, unmasked_ids, attention_mask
    """
    B, N = input_ids.shape
    device = input_ids.device

    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids != pad_token_id).long()

    masked_input_ids = input_ids.clone()
    mask_labels = torch.full_like(input_ids, -100)  # -100 = ignore in loss
    mask_positions = []

    for b in range(B):
        # Find valid positions (non-padding)
        valid_mask = attention_mask[b]
        valid_positions = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_positions) == 0:
            continue

        # Calculate how many tokens to mask
        num_to_mask = max(1, int(len(valid_positions) * mask_prob))

        # Sample spans
        spans = []
        total_masked = 0
        attempts = 0

        while total_masked < num_to_mask and attempts < num_spans_to_sample * 3:
            # Sample span length from Poisson distribution
            span_len = max(1, int(random.expovariate(1.0 / mean_span_length)))
            span_len = min(span_len, num_to_mask - total_masked)
            span_len = min(span_len, 10)  # cap at 10

            # Sample start position from valid positions
            start_idx = random.randint(0, max(0, len(valid_positions) - span_len))
            start_pos = valid_positions[start_idx].item()

            # Check if span overlaps with existing spans
            span_positions = list(range(start_pos, min(start_pos + span_len, N)))
            if attention_mask[b, span_positions[-1]] == 0:
                attempts += 1
                continue

            overlap = any(p in [s for span in spans for s in span] for p in span_positions)
            if overlap:
                attempts += 1
                continue

            spans.append(span_positions)
            total_masked += len(span_positions)
            attempts += 1

        # Apply masks
        all_masked_positions = []
        for span in spans:
            for pos in span:
                mask_labels[b, pos] = input_ids[b, pos]
                masked_input_ids[b, pos] = mask_token_id
                all_masked_positions.append(pos)

        mask_positions.append(sorted(all_masked_positions))

    # Pad mask_positions to same length across batch
    max_masks = max(len(m) for m in mask_positions) if mask_positions else 0
    if max_masks == 0:
        max_masks = 1

    padded_mask_positions = torch.full((B, max_masks), pad_token_id, dtype=torch.long, device=device)
    mask_valid = torch.zeros(B, max_masks, dtype=torch.bool, device=device)

    for b, positions in enumerate(mask_positions):
        if len(positions) > 0:
            padded_mask_positions[b, :len(positions)] = torch.tensor(positions, device=device)
            mask_valid[b, :len(positions)] = True

    return {
        "masked_input_ids": masked_input_ids,
        "mask_labels": mask_labels,
        "mask_positions": padded_mask_positions,
        "mask_valid": mask_valid,
        "unmasked_ids": input_ids,
        "attention_mask": attention_mask,
    }
