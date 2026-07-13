"""CLI entry points for H-JEPA-LM."""

import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(
        prog="hjepa",
        description="H-JEPA-LM: Hierarchical Joint-Embedding Predictive Language Model",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Train H-JEPA-LM")
    sub.add_parser("bench", help="Run 5-way benchmark comparison")
    sub.add_parser("test", help="Run quick smoke test")
    sub.add_parser("info", help="Show model info")

    args = parser.parse_args()

    if args.command == "train":
        train_cmd()
    elif args.command == "bench":
        bench_cmd()
    elif args.command == "test":
        test_cmd()
    elif args.command == "info":
        info_cmd()
    else:
        parser.print_help()


def train_cmd():
    parser = argparse.ArgumentParser(description="Train H-JEPA-LM")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args(sys.argv[2:])

    from jepalm.config import JEPAConfig

    presets = {
        "tiny": dict(enc_hidden_dim=256, enc_num_layers=4, enc_num_heads=4,
                     pred_hidden_dim=128, pred_num_layers=2),
        "small": dict(enc_hidden_dim=512, enc_num_layers=6, enc_num_heads=8,
                      pred_hidden_dim=256, pred_num_layers=3),
        "medium": dict(enc_hidden_dim=768, enc_num_layers=12, enc_num_heads=12,
                       pred_hidden_dim=384, pred_num_layers=6),
    }

    config = JEPAConfig(**presets[args.preset])
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = args.device
    config.max_samples = args.max_samples
    config.output_dir = args.output_dir

    print(f"Training JEPA-LM ({args.preset}) on {args.device}...")
    from jepalm.train import train
    train(config)


def bench_cmd():
    print("Running 5-way benchmark...")
    import subprocess
    subprocess.run([sys.executable, "benchmark_hjepa.py"])


def test_cmd():
    print("Running smoke test...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def info_cmd():
    from jepalm import JEPELM, JEPAConfig, HJEPELM, HConfig

    print("=" * 50)
    print("JEPA-LM Models")
    print("=" * 50)

    models = [
        ("JEPA-LM (base)", JEPELM, JEPAConfig()),
        ("H-JEPA-LM", HJEPELM, HConfig()),
    ]

    for name, cls, config in models:
        try:
            model = cls(config)
            n = sum(p.numel() for p in model.parameters())
            print(f"\n  {name}: {n:,} parameters")
        except Exception as e:
            print(f"\n  {name}: Error - {e}")

    print()


if __name__ == "__main__":
    main()
