"""Main entry point for JEPA-LM training."""

import argparse
import torch

from jepalm.config import JEPAConfig
from jepalm.train import train


def main():
    parser = argparse.ArgumentParser(description="Train JEPA-LM")

    # Model
    parser.add_argument("--enc_hidden_dim", type=int, default=768)
    parser.add_argument("--enc_num_layers", type=int, default=12)
    parser.add_argument("--enc_num_heads", type=int, default=12)
    parser.add_argument("--pred_hidden_dim", type=int, default=384)
    parser.add_argument("--pred_num_layers", type=int, default=6)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_jepa", type=float, default=1.0)
    parser.add_argument("--lambda_ntp", type=float, default=0.1)
    parser.add_argument("--mask_prob", type=float, default=0.15)

    # General
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)

    # Quick presets
    parser.add_argument("--preset", type=str, default=None,
                       choices=["tiny", "small", "medium"],
                       help="Use a preset configuration")

    args = parser.parse_args()

    # Apply preset
    if args.preset == "tiny":
        config = JEPAConfig(
            enc_hidden_dim=256, enc_num_layers=4, enc_num_heads=4, enc_ff_dim=1024,
            pred_hidden_dim=128, pred_num_layers=2, pred_num_heads=4, pred_ff_dim=512,
            dec_hidden_dim=256, dec_num_layers=2, dec_num_heads=4, dec_ff_dim=1024,
            batch_size=64, num_epochs=5,
        )
        print("Using TINY preset (~10M params)")
    elif args.preset == "small":
        config = JEPAConfig(
            enc_hidden_dim=512, enc_num_layers=6, enc_num_heads=8, enc_ff_dim=2048,
            pred_hidden_dim=256, pred_num_layers=3, pred_num_heads=8, pred_ff_dim=1024,
            dec_hidden_dim=512, dec_num_layers=3, dec_num_heads=8, dec_ff_dim=2048,
            batch_size=32, num_epochs=10,
        )
        print("Using SMALL preset (~50M params)")
    elif args.preset == "medium":
        config = JEPAConfig(
            enc_hidden_dim=768, enc_num_layers=12, enc_num_heads=12, enc_ff_dim=3072,
            pred_hidden_dim=384, pred_num_layers=6, pred_num_heads=12, pred_ff_dim=1536,
            dec_hidden_dim=768, dec_num_layers=6, dec_num_heads=12, dec_ff_dim=3072,
            batch_size=16, num_epochs=10,
        )
        print("Using MEDIUM preset (~125M params)")
    else:
        config = JEPAConfig()

    # Override with args
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.lambda_jepa = args.lambda_jepa
    config.lambda_ntp = args.lambda_ntp
    config.mask_prob = args.mask_prob
    config.device = args.device
    config.seed = args.seed
    config.output_dir = args.output_dir
    config.log_every = args.log_every
    config.save_every = args.save_every

    # Print config
    print("\nConfiguration:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    # Train
    model = train(config)

    print("\nDone!")


if __name__ == "__main__":
    main()
