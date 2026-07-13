"""Baseline models for comparison with JEPA-LM."""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class BaselineBertEncoder(nn.Module):
    """Standard BERT encoder with masked language modeling (MLM) loss.

    Used as baseline to compare against JEPA-LM's JEPA loss.
    Same architecture as JEPA-LM's encoder, but trained with MLM only.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 ff_dim, max_seq_len, dropout=0.1, pad_token_id=0, mask_token_id=103):
        super().__init__()
        self.config = type("C", (), {
            "enc_vocab_size": vocab_size, "pad_token_id": pad_token_id,
            "mask_token_id": mask_token_id,
        })()

        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.embed_norm = nn.LayerNorm(hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, input_ids, attention_mask=None):
        B, N = input_ids.shape
        device = input_ids.device
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(self.embed_norm(x))

        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            bidir_mask = torch.ones(N, N, device=device, dtype=torch.bool).triu(diagonal=1)
            full_mask = pad_mask & (~bidir_mask).unsqueeze(0).unsqueeze(0)
        else:
            full_mask = None

        for layer in self.layers:
            x = layer(x, full_mask)

        return self.norm(x)

    def get_embedding(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return hidden_states.mean(dim=1)


class BaselineMLM(nn.Module):
    """Complete baseline: BERT encoder + MLM head.

    Trained with standard masked language modeling.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 ff_dim, max_seq_len, dropout=0.1, pad_token_id=0, mask_token_id=103):
        super().__init__()
        self.encoder = BaselineBertEncoder(
            vocab_size, hidden_dim, num_layers, num_heads,
            ff_dim, max_seq_len, dropout, pad_token_id, mask_token_id
        )
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.mlm_head(hidden)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return {"loss": loss, "logits": logits, "hidden_states": hidden}

    def get_embedding(self, hidden_states, attention_mask=None):
        return self.encoder.get_embedding(hidden_states, attention_mask)

    @property
    def config(self):
        return self.encoder.config
