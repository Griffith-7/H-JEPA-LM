"""Tests for JEPA-LM and H-JEPA-LM core components."""

import pytest
import torch
from jepalm.config import JEPAConfig
from jepalm.model import JEPELM
from jepalm.hjepa import HJEPELM, HConfig
from jepalm.encoder import BidirectionalEncoder
from jepalm.target_encoder import EMATargetEncoder
from jepalm.predictor import Predictor
from jepalm.decoder import LightweightDecoder
from jepalm.masking import create_span_mask
from jepalm.loss import jepa_loss, ntp_loss, total_loss


@pytest.fixture
def tiny_config():
    return JEPAConfig(
        enc_hidden_dim=64, enc_num_layers=2, enc_num_heads=4, enc_ff_dim=128,
        pred_hidden_dim=32, pred_num_layers=1, pred_num_heads=4, pred_ff_dim=64,
        dec_hidden_dim=64, dec_num_layers=1, dec_num_heads=4, dec_ff_dim=128,
        enc_max_seq_len=64, mask_prob=0.15, mean_span_length=3, num_spans_to_sample=2,
    )


@pytest.fixture
def tiny_hconfig():
    return HConfig(
        vocab_size=1000, dim=64, layers=4, heads=4, ff_dim=128,
        max_len=64, num_levels=2, action_dim=32, dropout=0.1,
    )


class TestEncoder:
    def test_forward(self, tiny_config):
        enc = BidirectionalEncoder(
            vocab_size=tiny_config.enc_vocab_size, hidden_dim=tiny_config.enc_hidden_dim,
            num_layers=tiny_config.enc_num_layers, num_heads=tiny_config.enc_num_heads,
            ff_dim=tiny_config.enc_ff_dim, max_seq_len=tiny_config.enc_max_seq_len,
        )
        x = torch.randint(0, 1000, (2, 32))
        out = enc(x)
        assert out.shape == (2, 32, tiny_config.enc_hidden_dim)

    def test_not_collapsed(self, tiny_config):
        enc = BidirectionalEncoder(
            vocab_size=tiny_config.enc_vocab_size, hidden_dim=tiny_config.enc_hidden_dim,
            num_layers=tiny_config.enc_num_layers, num_heads=tiny_config.enc_num_heads,
            ff_dim=tiny_config.enc_ff_dim, max_seq_len=tiny_config.enc_max_seq_len,
        )
        x = torch.randint(0, 1000, (2, 32))
        out = enc(x)
        cos = torch.nn.functional.cosine_similarity(out[0, 0:1], out[1, 0:1])
        assert cos.item() < 0.99


class TestTargetEncoder:
    def test_ema_update(self, tiny_config):
        enc = BidirectionalEncoder(
            vocab_size=tiny_config.enc_vocab_size, hidden_dim=tiny_config.enc_hidden_dim,
            num_layers=tiny_config.enc_num_layers, num_heads=tiny_config.enc_num_heads,
            ff_dim=tiny_config.enc_ff_dim, max_seq_len=tiny_config.enc_max_seq_len,
        )
        target = EMATargetEncoder(enc)
        x = torch.randint(0, 1000, (2, 16))
        out = target(x)
        assert out.shape == (2, 16, tiny_config.enc_hidden_dim)

    def test_no_grad(self, tiny_config):
        enc = BidirectionalEncoder(
            vocab_size=tiny_config.enc_vocab_size, hidden_dim=tiny_config.enc_hidden_dim,
            num_layers=tiny_config.enc_num_layers, num_heads=tiny_config.enc_num_heads,
            ff_dim=tiny_config.enc_ff_dim, max_seq_len=tiny_config.enc_max_seq_len,
        )
        target = EMATargetEncoder(enc)
        x = torch.randint(0, 1000, (2, 16))
        out = target(x)
        assert not out.requires_grad


class TestPredictor:
    def test_forward(self, tiny_config):
        pred = Predictor(
            enc_hidden_dim=tiny_config.enc_hidden_dim, pred_hidden_dim=tiny_config.pred_hidden_dim,
            pred_num_layers=tiny_config.pred_num_layers, pred_num_heads=tiny_config.pred_num_heads,
            pred_ff_dim=tiny_config.pred_ff_dim,
        )
        z = torch.randn(2, 16, tiny_config.enc_hidden_dim)
        mask_pos = torch.arange(16).unsqueeze(0).expand(2, -1)
        out = pred(z, mask_pos)
        assert out.shape[0] == 2
        assert out.shape[2] == tiny_config.enc_hidden_dim


class TestDecoder:
    def test_forward(self, tiny_config):
        dec = LightweightDecoder(
            vocab_size=tiny_config.enc_vocab_size, hidden_dim=tiny_config.dec_hidden_dim,
            num_layers=tiny_config.dec_num_layers, num_heads=tiny_config.dec_num_heads,
            ff_dim=tiny_config.dec_ff_dim, max_seq_len=tiny_config.enc_max_seq_len,
        )
        x = torch.randint(0, 1000, (2, 16))
        logits = dec(x)
        assert logits.shape == (2, 16, tiny_config.enc_vocab_size)


class TestMasking:
    def test_create_span_mask(self, tiny_config):
        result = create_span_mask(
            torch.randint(0, 1000, (2, 32)),
            mask_prob=tiny_config.mask_prob, mean_span_length=tiny_config.mean_span_length,
            num_spans_to_sample=tiny_config.num_spans_to_sample,
            mask_token_id=tiny_config.mask_token_id, pad_token_id=tiny_config.pad_token_id,
        )
        assert "masked_input_ids" in result
        assert "mask_positions" in result
        assert result["masked_input_ids"].shape == (2, 32)


class TestLoss:
    def test_jepa_loss(self):
        pred = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        mask = torch.ones(4, 16, dtype=torch.bool)
        loss, cos_sim = jepa_loss(pred, target, mask)
        assert loss.ndim == 0 and loss.item() >= 0
        assert cos_sim.ndim == 0

    def test_ntp_loss(self):
        logits = torch.randn(4, 32, 1000)
        targets = torch.randint(0, 1000, (4, 32))
        loss = ntp_loss(logits, targets)
        assert loss.ndim == 0 and loss.item() >= 0

    def test_total_loss(self):
        t = total_loss(torch.tensor(1.0), torch.tensor(0.5), lambda_jepa=1.0, lambda_ntp=0.1)
        assert abs(t.item() - 1.05) < 1e-5


class TestJEPELM:
    def test_forward(self, tiny_config):
        model = JEPELM(tiny_config)
        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        assert "loss" in out and "jepa_loss" in out and "cosine_similarity" in out

    def test_count_parameters(self, tiny_config):
        counts = JEPELM(tiny_config).count_parameters()
        assert counts["total"] > 0 and counts["encoder"] > 0

    def test_trainable(self, tiny_config):
        model = JEPELM(tiny_config)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        out["loss"].backward()
        opt.step()


class TestHJEPELM:
    def test_forward(self, tiny_hconfig):
        model = HJEPELM(tiny_hconfig)
        x = torch.randint(0, tiny_hconfig.vocab_size, (2, 32))
        out = model(x)
        assert isinstance(out, list)
        assert out[-1].shape == (2, 32, tiny_hconfig.dim)

    def test_compute_loss(self, tiny_hconfig):
        model = HJEPELM(tiny_hconfig)
        x = torch.randint(0, tiny_hconfig.vocab_size, (2, 32))
        loss, val = model.compute_loss(x)
        assert loss.ndim == 0 and loss.item() >= 0

    def test_compute_loss_with_actions(self, tiny_hconfig):
        model = HJEPELM(tiny_hconfig)
        x = torch.randint(0, tiny_hconfig.vocab_size, (2, 32))
        actions = torch.randn(2, tiny_hconfig.action_dim)
        loss, val = model.compute_loss(x, actions=actions)
        assert loss.ndim == 0

    def test_plan_actions(self, tiny_hconfig):
        model = HJEPELM(tiny_hconfig)
        x = torch.randint(0, tiny_hconfig.vocab_size, (2, 32))
        goal = torch.randn(2, tiny_hconfig.dim)
        planned = model.plan_actions(x, goal, num_steps=2, num_candidates=3)
        assert planned.shape[0] == 2
        assert planned.shape[1] == 2

    def test_count_parameters(self, tiny_hconfig):
        counts = HJEPELM(tiny_hconfig).count_parameters()
        assert counts["total"] > 0

    def test_trainable(self, tiny_hconfig):
        model = HJEPELM(tiny_hconfig)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randint(0, tiny_hconfig.vocab_size, (2, 32))
        loss, _ = model.compute_loss(x)
        loss.backward()
        opt.step()


class TestCLI:
    def test_import_cli(self):
        from jepalm.cli import main
        assert callable(main)

    def test_import_all(self):
        from jepalm import JEPELM, JEPAConfig, HJEPELM, HConfig
        assert JEPELM is not None
        assert HJEPELM is not None
