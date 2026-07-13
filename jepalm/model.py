"""Main JEPA-LM model — combines all components."""

import torch
import torch.nn as nn

from .encoder import BidirectionalEncoder
from .target_encoder import EMATargetEncoder
from .predictor import Predictor
from .decoder import LightweightDecoder
from .masking import create_span_mask
from .loss import jepa_loss, ntp_loss, total_loss
from .config import JEPAConfig


class JEPELM(nn.Module):
    """Joint-Embedding Predictive Language Model.

    Architecture:
        Encoder (bidirectional) → latent representations
        Predictor (narrow) → predicted latents for masked spans
        EMA Target Encoder → stable target latents
        Decoder (lightweight) → text generation from latents

    Primary objective: JEPA loss (cosine similarity in latent space)
    Secondary objective: Weak NTP loss (reconstruct masked tokens)
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # Components
        self.encoder = BidirectionalEncoder(
            vocab_size=config.enc_vocab_size,
            hidden_dim=config.enc_hidden_dim,
            num_layers=config.enc_num_layers,
            num_heads=config.enc_num_heads,
            ff_dim=config.enc_ff_dim,
            max_seq_len=config.enc_max_seq_len,
            dropout=config.enc_dropout,
            pad_token_id=config.pad_token_id,
        )

        self.target_encoder = EMATargetEncoder(self.encoder)

        self.predictor = Predictor(
            enc_hidden_dim=config.enc_hidden_dim,
            pred_hidden_dim=config.pred_hidden_dim,
            pred_num_layers=config.pred_num_layers,
            pred_num_heads=config.pred_num_heads,
            pred_ff_dim=config.pred_ff_dim,
            dropout=config.pred_dropout,
        )

        self.decoder = LightweightDecoder(
            vocab_size=config.enc_vocab_size,
            hidden_dim=config.dec_hidden_dim,
            num_layers=config.dec_num_layers,
            num_heads=config.dec_num_heads,
            ff_dim=config.dec_ff_dim,
            max_seq_len=config.enc_max_seq_len,
            dropout=config.dec_dropout,
            pad_token_id=config.pad_token_id,
        )

    def forward(self, input_ids, return_logits=False):
        """Forward pass for training.

        Args:
            input_ids: (B, N) — input token ids
            return_logits: whether to return decoder logits

        Returns:
            dict with losses and metrics
        """
        # Step 1: Create span masks
        mask_data = create_span_mask(
            input_ids,
            mask_prob=self.config.mask_prob,
            mean_span_length=self.config.mean_span_length,
            num_spans_to_sample=self.config.num_spans_to_sample,
            mask_token_id=self.config.mask_token_id,
            pad_token_id=self.config.pad_token_id,
        )

        masked_input_ids = mask_data["masked_input_ids"]
        mask_labels = mask_data["mask_labels"]
        mask_positions = mask_data["mask_positions"]
        mask_valid = mask_data["mask_valid"]
        unmasked_ids = mask_data["unmasked_ids"]
        attention_mask = mask_data["attention_mask"]

        # Step 2: Encode masked input
        encoder_hidden = self.encoder(masked_input_ids, attention_mask)

        # Step 3: Encode full input with target encoder (no gradients)
        with torch.no_grad():
            target_hidden = self.target_encoder(unmasked_ids, attention_mask)

        # Step 4: Extract target latents at masked positions
        B, N, D = target_hidden.shape
        T = mask_positions.shape[1]

        # Gather target latents at masked positions
        target_latents = torch.gather(
            target_hidden, 1, mask_positions.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, T, D)

        # Step 5: Predict latents for masked positions
        predicted_latents = self.predictor(encoder_hidden, mask_positions)  # (B, T, D)

        # Step 6: Compute JEPA loss
        jepa_loss_val, cosine_sim = jepa_loss(predicted_latents, target_latents, mask_valid)

        # Step 7: Compute NTP loss (weak, secondary)
        ntp_loss_val = torch.tensor(0.0, device=input_ids.device)
        if return_logits:
            # Decode predicted latents to tokens
            logits = self.decoder(masked_input_ids, predicted_latents, attention_mask)
            ntp_loss_val = ntp_loss(logits, mask_labels)

        # Step 8: Combine losses
        total = total_loss(jepa_loss_val, ntp_loss_val,
                          self.config.lambda_jepa, self.config.lambda_ntp)

        results = {
            "loss": total,
            "jepa_loss": jepa_loss_val.item(),
            "ntp_loss": ntp_loss_val.item(),
            "cosine_similarity": cosine_sim.item(),
            "num_masked": mask_valid.sum().item(),
        }

        if return_logits:
            results["logits"] = logits

        return results

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """Update target encoder via EMA."""
        self.target_encoder.update(self.encoder, momentum)

    @torch.no_grad()
    def predict_masked(self, input_ids, mask_positions):
        """Predict latent representations for specific positions.

        Used for inference and analysis.
        """
        attention_mask = (input_ids != self.config.pad_token_id).long()
        encoder_hidden = self.encoder(input_ids, attention_mask)
        predicted_latents = self.predictor(encoder_hidden, mask_positions)
        return predicted_latents

    def decode_from_latents(self, input_ids, latent_vectors, attention_mask=None):
        """Decode text from latent vectors.

        Used for text generation from predicted latents.
        """
        return self.decoder(input_ids, latent_vectors, attention_mask)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        target_params = sum(p.numel() for p in self.target_encoder.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_params,
            "predictor": predictor_params,
            "decoder": decoder_params,
            "target_encoder": target_params,
        }
