"""EMA Target Encoder — slow-moving copy of encoder for stable JEPA targets."""

import copy
import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    """Target encoder updated via exponential moving average.

    This is an exact copy of the BidirectionalEncoder but:
    - Not trained via gradients
    - Updated via EMA: θ̄ ← m·θ̄ + (1-m)·θ
    - Produces stable prediction targets
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)

    @torch.no_grad()
    def get_embedding(self, hidden_states, attention_mask=None):
        return self.encoder.get_embedding(hidden_states, attention_mask)

    @torch.no_grad()
    def update(self, source_encoder, momentum):
        """Update target encoder weights via EMA.

        θ̄ ← m·θ̄ + (1-m)·θ
        """
        for param, source_param in zip(
            self.encoder.parameters(), source_encoder.parameters()
        ):
            param.data.mul_(momentum).add_(source_param.data, alpha=1 - momentum)
