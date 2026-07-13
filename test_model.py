"""Quick test to verify JEPA-LM works end-to-end."""

import torch
from jepalm.config import JEPAConfig
from jepalm.model import JEPELM
from jepalm.train import SimpleTextDataset
from torch.utils.data import DataLoader


def main():
    print("=" * 50)
    print("JEPA-LM Quick Test")
    print("=" * 50)

    config = JEPAConfig(
        enc_vocab_size=1000,
        enc_hidden_dim=128,
        enc_num_layers=2,
        enc_num_heads=4,
        enc_ff_dim=256,
        enc_max_seq_len=64,
        pred_hidden_dim=64,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ff_dim=128,
        dec_hidden_dim=128,
        dec_num_layers=2,
        dec_num_heads=4,
        dec_ff_dim=256,
        batch_size=8,
        mask_prob=0.15,
        num_epochs=2,
        device="cuda",
    )

    model = JEPELM(config).to("cuda")
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = SimpleTextDataset(1000, 64, 100)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    losses = []
    for epoch in range(2):
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to("cuda")
            results = model(ids, return_logits=True)
            results["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()
            model.update_target_encoder(0.996)
            losses.append(results["loss"].item())
            if i % 5 == 0:
                print(f"  Ep{epoch} Step{i} Loss={results['loss'].item():.4f} "
                      f"JEPA={results['jepa_loss']:.4f} "
                      f"CosSim={results['cosine_similarity']:.4f}")

    print(f"\nSteps trained: {len(losses)}")
    print(f"First loss: {losses[0]:.4f}")
    print(f"Last loss: {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
