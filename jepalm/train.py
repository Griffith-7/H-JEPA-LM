"""Training loop for JEPA-LM."""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

from .config import JEPAConfig
from .model import JEPELM


class SimpleTextDataset(Dataset):
    """Simple text dataset that tokenizes on-the-fly.

    For now, uses random token sequences for testing.
    Replace with real tokenizer later.
    """

    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random tokens (for testing)
        tokens = torch.randint(1, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: JEPAConfig = None):
    """Main training function."""
    if config is None:
        config = JEPAConfig()

    print("=" * 60)
    print("JEPA-LM Training")
    print("=" * 60)

    # Seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    print("\nInitializing model...")
    model = JEPELM(config).to(device)
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"  Encoder:      {params['encoder']:,}")
    print(f"  Predictor:    {params['predictor']:,}")
    print(f"  Decoder:      {params['decoder']:,}")
    print(f"  Target Enc:   {params['target_encoder']:,} (not trained)")

    # Dataset
    print("\nPreparing dataset...")
    dataset = SimpleTextDataset(
        vocab_size=config.enc_vocab_size,
        seq_len=config.enc_max_seq_len,
        num_samples=10000,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    # Mixed precision
    scaler = GradScaler("cuda") if config.fp16 and device.type == "cuda" else None

    # Output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"Batch size: {config.batch_size}")
    print(f"Total steps: {total_steps}")
    print(f"Lambda JEPA: {config.lambda_jepa}")
    print(f"Lambda NTP: {config.lambda_ntp}")
    print("-" * 60)

    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_jepa = 0
        epoch_cosine = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast("cuda"):
                    results = model(input_ids, return_logits=True)
                    loss = results["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                results = model(input_ids, return_logits=True)
                loss = results["loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            # Update target encoder via EMA
            momentum = config.ema_momentum_start + (
                config.ema_momentum_end - config.ema_momentum_start
            ) * (global_step / max(1, total_steps - 1))
            model.update_target_encoder(momentum)

            # Track metrics
            epoch_loss += results["loss"].item()
            epoch_jepa += results["jepa_loss"]
            epoch_cosine += results["cosine_similarity"]
            num_batches += 1
            global_step += 1

            # Log
            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / num_batches
                avg_jepa = epoch_jepa / num_batches
                avg_cosine = epoch_cosine / num_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time

                print(
                    f"Step {global_step:6d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"JEPA: {avg_jepa:.4f} | "
                    f"CosSim: {avg_cosine:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"EMA: {momentum:.4f} | "
                    f"Time: {elapsed:.0f}s"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # Save checkpoint
            if global_step % config.save_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": config,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

        # End of epoch
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch + 1}/{config.num_epochs} complete | "
              f"Avg Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
        print("-" * 60)

    # Save final model
    final_path = os.path.join(config.output_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "loss": best_loss,
        "config": config,
    }, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    return model


if __name__ == "__main__":
    train()
