"""
H-JEPA-LM: Hierarchical JEPA with Action Conditioning for Text
==============================================================
Improvements over base JEPA-LM:
1. H-JEPA: Multi-level prediction (low-level = tokens, high-level = semantics)
2. Action Conditioning: Predict consequences of actions in latent space
3. World Model: Plan actions in latent space, then decode to text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class HConfig:
    vocab_size: int = 30522
    dim: int = 256
    layers: int = 4
    heads: int = 4
    ff_dim: int = 1024
    max_len: int = 128
    num_levels: int = 2
    action_dim: int = 64
    dropout: float = 0.1
    mask_prob: float = 0.15
    mask_token_id: int = 103
    momentum: float = 0.996


class Block(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout))

    def forward(self, x, causal=False):
        B, N, _ = x.shape
        r = x
        xn = self.n1(x)
        if causal:
            m = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
            o, _ = self.attn(xn, xn, xn, attn_mask=m)
        else:
            o, _ = self.attn(xn, xn, xn)
        x = r + o
        return x + self.ff(self.n2(x))


class HierarchicalEncoder(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads, ff_dim, max_len, num_levels=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        self.level_layers = nn.ModuleList()
        self.level_norms = nn.ModuleList()
        for lvl in range(num_levels):
            self.level_layers.append(
                nn.ModuleList([Block(dim, heads, ff_dim, dropout) for _ in range(layers // num_levels)])
            )
            self.level_norms.append(nn.LayerNorm(dim))

        self.pool_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_levels - 1)])

    def forward(self, input_ids, level=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(input_ids) + self.pos_embed(pos)

        if level is not None:
            for layer in self.level_layers[level]:
                h = layer(h, causal=False)
            return self.level_norms[level](h)

        outputs = []
        for lvl in range(self.num_levels):
            for layer in self.level_layers[lvl]:
                h = layer(h, causal=False)
            h = self.level_norms[lvl](h)
            outputs.append(h)
            if lvl < self.num_levels - 1:
                B2, N2, D = h.shape
                h_pooled = h.mean(dim=1, keepdim=True).expand(B2, N2, D)
                h = self.pool_proj[lvl](h_pooled) + h

        return outputs

    def encode_level(self, input_ids, level):
        return self.forward(input_ids, level=level)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, action):
        return self.proj(action)


class HierarchicalPredictor(nn.Module):
    def __init__(self, dim, num_levels, num_heads, dropout=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, dim) * 0.02) for _ in range(num_levels)
        ])

        self.level_predictors = nn.ModuleList()
        for lvl in range(num_levels):
            pred_layers = nn.ModuleList([Block(dim, num_heads, dim * 2, dropout) for _ in range(2)])
            self.level_predictors.append(nn.ModuleDict({
                'layers': pred_layers,
                'norm': nn.LayerNorm(dim),
                'out': nn.Linear(dim, dim),
            }))

        self.action_proj = nn.Linear(dim, dim)
        self.combine = nn.Linear(dim * 2, dim)

    def forward(self, encoder_outputs, mask_positions, actions=None):
        B = encoder_outputs[0].shape[0]
        predictions = []

        for lvl, h in enumerate(encoder_outputs):
            T = mask_positions.shape[1]
            ctx = h
            mask_tok = self.mask_tokens[lvl].expand(B, T, -1)
            ctx_at = torch.gather(ctx, 1, mask_positions.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))

            if actions is not None and lvl == self.num_levels - 1:
                act_emb = self.action_proj(actions)
                act_expanded = act_emb.unsqueeze(1).expand(-1, T, -1)
                ctx_at = self.combine(torch.cat([ctx_at, act_expanded], dim=-1))

            x = torch.cat([ctx_at, mask_tok], dim=1)
            for layer in self.level_predictors[lvl]['layers']:
                x = layer(x, causal=False)
            x = self.level_predictors[lvl]['norm'](x)
            pred = self.level_predictors[lvl]['out'](x[:, T:, :])
            predictions.append(pred)

        return predictions


class HJEPELM(nn.Module):
    """Hierarchical JEPA-LM with Action Conditioning.

    Architecture:
        HierarchicalEncoder: multi-level bidirectional encoding
        HierarchicalPredictor: multi-level prediction with action inputs
        EMA Target Encoder: stable targets at each level
        World Model: plan actions in latent space

    Primary objective: Multi-level JEPA loss
    Secondary objective: Weak NTP loss (reconstruct tokens)
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = HConfig(**kwargs)
        self.config = config

        self.dim = config.dim
        self.num_levels = config.num_levels
        self.vocab_size = config.vocab_size

        self.encoder = HierarchicalEncoder(
            config.vocab_size, config.dim, config.layers, config.heads,
            config.ff_dim, config.max_len, config.num_levels, config.dropout)

        layers_per_level = config.layers // config.num_levels
        self.target_layers = nn.ModuleList()
        self.target_norms = nn.ModuleList()
        for lvl in range(config.num_levels):
            tl = nn.ModuleList([Block(config.dim, config.heads, config.ff_dim, config.dropout)
                                for _ in range(layers_per_level)])
            tn = nn.LayerNorm(config.dim)
            for p in tl.parameters():
                p.requires_grad = False
            for p in tn.parameters():
                p.requires_grad = False
            self.target_layers.append(tl)
            self.target_norms.append(tn)

        self.predictor = HierarchicalPredictor(config.dim, config.num_levels, config.heads, config.dropout)

        self.action_encoder = ActionEncoder(config.action_dim, config.dim)

        self.decoder = nn.Sequential(
            nn.Linear(config.dim, config.ff_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.vocab_size))

        self._sync_targets()

    @torch.no_grad()
    def _sync_targets(self):
        for lvl in range(self.num_levels):
            for tp, sp in zip(self.target_layers[lvl].parameters(),
                              self.encoder.level_layers[lvl].parameters()):
                tp.data.copy_(sp.data)
            self.target_norms[lvl].load_state_dict(self.encoder.level_norms[lvl].state_dict())

    @torch.no_grad()
    def update_targets(self, momentum=None):
        if momentum is None:
            momentum = self.config.momentum
        for lvl in range(self.num_levels):
            for p, tp in zip(self.encoder.level_layers[lvl].parameters(),
                             self.target_layers[lvl].parameters()):
                tp.data.mul_(momentum).add_(p.data, alpha=1 - momentum)
            for p, tp in zip(self.encoder.level_norms[lvl].parameters(),
                             self.target_norms[lvl].parameters()):
                tp.data.mul_(momentum).add_(p.data, alpha=1 - momentum)

    def _target_forward(self, input_ids, level):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h = self.encoder.token_embed(input_ids) + self.encoder.pos_embed(pos)
        for layer in self.target_layers[level]:
            h = layer(h, causal=False)
        return self.target_norms[level](h)

    def encode(self, input_ids):
        return self.encoder(input_ids)

    def get_embedding(self, hidden_levels):
        if isinstance(hidden_levels, list):
            return hidden_levels[-1].mean(dim=1)
        return hidden_levels.mean(dim=1)

    def compute_loss(self, input_ids, mask_token_id=None, actions=None, mask_prob=None):
        if mask_token_id is None:
            mask_token_id = self.config.mask_token_id
        if mask_prob is None:
            mask_prob = self.config.mask_prob

        B, N = input_ids.shape
        mask = torch.rand(B, N, device=input_ids.device) < mask_prob

        masked_ids = input_ids.clone()
        masked_ids[mask] = mask_token_id

        encoder_outputs = self.encode(masked_ids)
        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]

        mask_pos = []
        for b in range(B):
            pos = mask[b].nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                mask_pos.append(pos)

        if not mask_pos:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True), 0.0

        max_t = max(len(p) for p in mask_pos)
        padded_pos = torch.zeros(len(mask_pos), max_t, dtype=torch.long, device=input_ids.device)
        valid = torch.zeros(len(mask_pos), max_t, dtype=torch.bool, device=input_ids.device)
        for b, pos in enumerate(mask_pos):
            padded_pos[b, :len(pos)] = pos
            valid[b, :len(pos)] = True

        with torch.no_grad():
            target_outputs = []
            for lvl in range(self.num_levels):
                full_enc = self._target_forward(input_ids, lvl)
                target_outputs.append(full_enc)

        encoded_actions = self.action_encoder(actions) if actions is not None else None
        predicted = self.predictor(encoder_outputs, padded_pos, encoded_actions)

        total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        for lvl in range(self.num_levels):
            target_at_pos = torch.gather(
                target_outputs[lvl], 1,
                padded_pos.unsqueeze(-1).expand(-1, -1, target_outputs[lvl].size(-1)))
            cos_sim = F.cosine_similarity(predicted[lvl], target_at_pos, dim=-1)
            lvl_loss = (1.0 - cos_sim[valid]).mean()
            weight = 1.0 if lvl == self.num_levels - 1 else 0.5
            total_loss = total_loss + weight * lvl_loss

        return total_loss, total_loss.item()

    def predict_next_state(self, input_ids, actions):
        with torch.no_grad():
            current = self.encode(input_ids)
            if isinstance(current, list):
                current = current[-1]

        act_emb = self.action_encoder(actions)
        predicted = current[:, -1:, :] + act_emb.unsqueeze(1)
        return predicted

    def plan_actions(self, input_ids, goal_latent, num_steps=5, num_candidates=10):
        with torch.no_grad():
            current = self.encode(input_ids)
            if isinstance(current, list):
                current = current[-1]

        best_actions = None

        for _ in range(num_steps):
            candidates = torch.randn(
                input_ids.shape[0], num_candidates, self.action_encoder.proj[0].in_features,
                device=input_ids.device)

            candidate_actions = []
            for c in range(num_candidates):
                act_emb = self.action_encoder(candidates[:, c, :])
                next_state = current[:, -1:, :] + act_emb.unsqueeze(1)
                distance = F.mse_loss(next_state, goal_latent.unsqueeze(1), reduction='none').mean(dim=[1, 2])
                candidate_actions.append((candidates[:, c, :], distance))

            best_idx = min(range(len(candidate_actions)), key=lambda i: candidate_actions[i][1].mean().item())
            current_action = candidate_actions[best_idx][0]
            act_emb = self.action_encoder(current_action)
            current = current.clone()
            current[:, -1:, :] = current[:, -1:, :] + act_emb.unsqueeze(1)

            if best_actions is None:
                best_actions = current_action.unsqueeze(1)
            else:
                best_actions = torch.cat([best_actions, current_action.unsqueeze(1)], dim=1)

        return best_actions

    def forward(self, input_ids, actions=None):
        return self.encode(input_ids)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_params,
            "predictor": predictor_params,
        }
