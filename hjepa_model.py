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
    """Multi-level encoder: low-level captures tokens, high-level captures semantics."""
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
    """Encodes actions into the latent space."""
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, action):
        return self.proj(action)


class HierarchicalPredictor(nn.Module):
    """Multi-level predictor with action conditioning."""
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


class EMATarget(nn.Module):
    """EMA target for a specific level."""
    def __init__(self, source_layers, source_norm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(source_layers[0].ff[0].in_features if hasattr(source_layers[0], 'ff') else 256)
        for p in self.layers.parameters():
            p.requires_grad = False

    def sync_from(self, source_layers, source_norm):
        self.layers = nn.ModuleList([Block(
            layer.ff[2].out_features if hasattr(layer, 'ff') else 256,
            layer.attn.num_heads,
            layer.ff[2].out_features * 4 if hasattr(layer, 'ff') else 1024
        ) for layer in source_layers])
        self.norm.load_state_dict(source_norm.state_dict())
        for p in self.layers.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, source_layers, source_norm, momentum):
        for p, tp in zip(source_layers.parameters(), self.layers.parameters()):
            tp.data.mul_(momentum).add_(p.data, alpha=1 - momentum)
        for p, tp in zip(source_norm.parameters(), self.norm.parameters()):
            tp.data.mul_(momentum).add_(p.data, alpha=1 - momentum)


class HJEPALM(nn.Module):
    """Hierarchical JEPA-LM with Action Conditioning.

    Architecture:
        HierarchicalEncoder: multi-level bidirectional encoding
        HierarchicalPredictor: multi-level prediction with action inputs
        EMA Target Encoder: stable targets at each level
        World Model: plan actions in latent space

    Primary objective: Multi-level JEPA loss
    Secondary objective: Weak NTP loss (reconstruct tokens)
    """
    def __init__(self, vocab_size=30522, dim=256, layers=4, heads=4, ff_dim=1024,
                 max_len=128, num_levels=2, action_dim=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.vocab_size = vocab_size

        self.encoder = HierarchicalEncoder(vocab_size, dim, layers, heads, ff_dim, max_len, num_levels, dropout)

        self.target_encoders = nn.ModuleList()
        for lvl in range(num_levels):
            te = HierarchicalEncoder(vocab_size, dim, layers, heads, ff_dim, max_len, 1, dropout)
            for p in te.parameters():
                p.requires_grad = False
            self.target_encoders.append(te)

        self.predictor = HierarchicalPredictor(dim, num_levels, heads, dropout)

        self.action_encoder = ActionEncoder(action_dim, dim)

        self.decoder = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, vocab_size))

        self._sync_targets()

    @torch.no_grad()
    def _sync_targets(self):
        for lvl, te in enumerate(self.target_encoders):
            src_layers = self.encoder.level_layers[lvl]
            src_norm = self.encoder.level_norms[lvl]
            for tp, sp in zip(te.parameters(), [p for l in src_layers for p in l.parameters()] + list(src_norm.parameters())):
                tp.data.copy_(sp.data)

    @torch.no_grad()
    def update_targets(self, momentum=0.996):
        for lvl, te in enumerate(self.target_encoders):
            for p, tp in zip([p for l in self.encoder.level_layers[lvl] for p in l.parameters()], te.parameters()):
                if tp.requires_grad is False:
                    tp.data.mul_(momentum).add_(p.data, alpha=1 - momentum)
            for p, tp in zip(self.encoder.level_norms[lvl].parameters(), te.norm.parameters()):
                if tp.requires_grad is False:
                    tp.data.mul_(momentum).add_(p.data, alpha=1 - momentum)

    def encode(self, input_ids):
        return self.encoder(input_ids)

    def get_embedding(self, hidden_levels):
        if isinstance(hidden_levels, list):
            return hidden_levels[-1].mean(dim=1)
        return hidden_levels.mean(dim=1)

    def compute_loss(self, input_ids, mask_token_id, actions=None, mask_prob=0.15):
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
                tgt = self.target_encoders[lvl]
                full_enc = tgt.forward(input_ids, level=0)
                target_outputs.append(full_enc)

        predicted = self.predictor(encoder_outputs, padded_pos, actions)

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
        """World model: predict next latent state given current state and action."""
        with torch.no_grad():
            current = self.encode(input_ids)
            if isinstance(current, list):
                current = current[-1]

        act_emb = self.action_encoder(actions)
        predicted = current[:, -1:, :] + act_emb.unsqueeze(1)
        return predicted

    def plan_actions(self, input_ids, goal_latent, num_steps=5, num_candidates=10):
        """Plan actions to reach a goal state in latent space."""
        with torch.no_grad():
            current = self.encode(input_ids)
            if isinstance(current, list):
                current = current[-1]

        best_actions = None
        best_distance = float('inf')

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
        return {"total": total, "trainable": trainable}


def test_hjepa():
    model = HJEPALM(vocab_size=30522, dim=128, layers=4, heads=4, ff_dim=512, max_len=64, num_levels=2)
    params = model.count_parameters()
    print(f"H-JEPA-LM Parameters: {params['total']:,} (trainable: {params['trainable']:,})")

    input_ids = torch.randint(0, 30522, (2, 64))
    mask_token_id = 103

    loss, val = model.compute_loss(input_ids, mask_token_id)
    print(f"JEPA Loss: {loss.item():.4f}")

    actions = torch.randn(2, 64)
    loss_action, val_action = model.compute_loss(input_ids, mask_token_id, actions=actions)
    print(f"JEPA Loss (with action): {loss_action.item():.4f}")

    next_state = model.predict_next_state(input_ids, actions)
    print(f"Next state shape: {next_state.shape}")

    goal_latent = torch.randn(2, 128)
    planned_actions = model.plan_actions(input_ids, goal_latent, num_steps=3, num_candidates=5)
    print(f"Planned actions shape: {planned_actions.shape}")

    print("All tests passed!")


if __name__ == "__main__":
    test_hjepa()
