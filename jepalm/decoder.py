"""Lightweight Decoder — maps predicted latents back to text tokens."""

import torch
import torch.nn as nn


class LightweightDecoder(nn.Module):
    """Small autoregressive decoder for text generation from latents.

    This is NOT the primary training objective. It provides a weak
    next-token prediction signal to maintain generative capability.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 ff_dim, max_seq_len, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.embed_norm = nn.LayerNorm(hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Project latent vectors to decoder hidden dim
        self.latent_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([
            self._make_block(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _make_block(self, dim, num_heads, ff_dim, dropout):
        return nn.ModuleDict({
            "attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
            "norm1": nn.LayerNorm(dim),
            "ff": nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout),
            ),
            "norm2": nn.LayerNorm(dim),
        })

    def forward(self, input_ids, latent_context=None, attention_mask=None):
        """Decode tokens, optionally conditioned on latent context.

        Args:
            input_ids: (B, S) — token ids to decode
            latent_context: (B, L, D) — latent vectors to condition on (optional)
            attention_mask: (B, S) — padding mask

        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = input_ids.shape
        device = input_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(self.embed_norm(x))

        # Prepend latent context if provided
        if latent_context is not None:
            latent_projected = self.latent_proj(latent_context)
            x = torch.cat([latent_projected, x], dim=1)
            S_total = S + latent_context.size(1)
        else:
            S_total = S

        # Build attention mask for nn.MultiheadAttention with batch_first=True
        # Needs 3D: (B*num_heads, S_total, S_total) — 0.0 = attend, -inf = block
        causal_block = torch.triu(
            torch.ones(S_total, S_total, device=device), diagonal=1
        )  # upper triangle = 1.0

        # Padding: (B, S_total)
        if attention_mask is not None:
            if latent_context is not None:
                latent_mask = torch.ones(B, latent_context.size(1), device=device)
                full_pad = torch.cat([latent_mask, attention_mask.float()], dim=1)
            else:
                full_pad = attention_mask.float()
        else:
            full_pad = torch.ones(B, S_total, device=device)

        # Combine causal + padding into (B, S_total, S_total)
        pad_block = (1.0 - full_pad).unsqueeze(1)  # (B, 1, S_total) — 1.0 where padded
        attn_mask = causal_block.unsqueeze(0) + pad_block  # (B, S_total, S_total)
        attn_mask = attn_mask.masked_fill(attn_mask > 0, float("-inf"))
        # Expand for num_heads: (B, S, S) -> (B*num_heads, S, S)
        num_heads = self.layers[0]["attn"].num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        for layer in self.layers:
            residual = x
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["attn"](x_norm, x_norm, x_norm, attn_mask=attn_mask)
            x = residual + attn_out
            x = residual + layer["ff"](layer["norm2"](x))

        x = self.norm(x)

        # If we prepended latent context, only decode from token positions
        if latent_context is not None:
            x = x[:, latent_context.size(1):, :]

        logits = self.head(x)
        return logits
