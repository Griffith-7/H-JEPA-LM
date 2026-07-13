"""Predictor — narrow transformer that predicts target latents from context."""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """Narrow transformer predictor.

    Takes encoder output + mask tokens and predicts latent representations
    of masked spans. The narrow architecture forces the encoder to learn
    informative representations (bottleneck principle from I-JEPA).
    """

    def __init__(self, enc_hidden_dim, pred_hidden_dim, pred_num_layers,
                 pred_num_heads, pred_ff_dim, dropout=0.1):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim

        # Project from encoder dim to predictor dim
        self.in_proj = nn.Linear(enc_hidden_dim, pred_hidden_dim)
        # Project back to encoder dim for loss computation
        self.out_proj = nn.Linear(pred_hidden_dim, enc_hidden_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_hidden_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer blocks
        self.layers = nn.ModuleList([
            self._make_block(pred_hidden_dim, pred_num_heads, pred_ff_dim, dropout)
            for _ in range(pred_num_layers)
        ])
        self.norm = nn.LayerNorm(pred_hidden_dim)

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

    def forward(self, encoder_output, mask_positions):
        """Predict latent representations for masked positions.

        Args:
            encoder_output: (B, N, enc_hidden_dim) — encoder hidden states
            mask_positions: (B, T) — indices of masked positions to predict

        Returns:
            predicted: (B, T, enc_hidden_dim) — predicted latents
        """
        B, N, _ = encoder_output.shape
        T = mask_positions.shape[1]

        # Project encoder output to predictor dim
        ctx = self.in_proj(encoder_output)  # (B, N, pred_hidden_dim)

        # Create mask tokens with positional information
        mask_tok = self.mask_token.expand(B, T, -1)  # (B, T, pred_hidden_dim)

        # Gather context tokens at mask positions for conditioning
        ctx_at_mask = torch.gather(
            ctx, 1, mask_positions.unsqueeze(-1).expand(-1, -1, ctx.size(-1))
        )  # (B, T, pred_hidden_dim)

        # Concatenate: [context at mask positions, mask tokens]
        # The mask tokens will cross-attend to context tokens
        x = torch.cat([ctx_at_mask, mask_tok], dim=1)  # (B, T*2, pred_hidden_dim)

        # Self-attention + FFN
        for layer in self.layers:
            residual = x
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["attn"](x_norm, x_norm, x_norm)
            x = residual + attn_out

            residual = x
            x = residual + layer["ff"](layer["norm2"](x))

        x = self.norm(x)

        # Take only the mask token outputs (last T tokens)
        mask_outputs = x[:, T:, :]  # (B, T, pred_hidden_dim)

        # Project back to encoder dim
        predicted = self.out_proj(mask_outputs)  # (B, T, enc_hidden_dim)

        return predicted
