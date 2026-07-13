"""Evaluation metrics for JEPA-LM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


class LinearProbe(nn.Module):
    """Linear probe for testing embedding quality."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def evaluate_embeddings(model, eval_loader, device, num_batches=20):
    """Evaluate embedding quality.

    Works with any model that has .encoder, .get_embedding(), or .encoder.get_embedding().
    """
    model.eval()
    all_embeddings = []
    all_norms = []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = (input_ids != getattr(model, "config", type("C", (), {"pad_token_id": 0})()).pad_token_id).long()

            # Get embeddings — handle both JEPA-LM and baseline
            if hasattr(model, "encoder") and hasattr(model.encoder, "get_embedding"):
                hidden = model.encoder(input_ids, attention_mask)
                embedding = model.encoder.get_embedding(hidden, attention_mask)
            elif hasattr(model, "get_embedding"):
                hidden = model.forward(input_ids) if callable(model.forward) else model(input_ids)
                if isinstance(hidden, dict):
                    hidden = hidden.get("hidden_states", hidden.get("logits"))
                embedding = model.get_embedding(hidden, attention_mask)
            else:
                output = model(input_ids)
                if isinstance(output, dict):
                    hidden = output.get("hidden_states", output.get("logits"))
                else:
                    hidden = output
                embedding = hidden.mean(dim=1)

            all_embeddings.append(embedding.cpu())
            all_norms.append(embedding.norm(dim=-1).cpu())

    all_emb = torch.cat(all_embeddings, dim=0).numpy()
    norms = torch.cat(all_norms, dim=0).numpy()

    # Embedding stats
    cos_sims = []
    all_emb_list = all_embeddings

    # Metrics
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    collapse_score = std_norm / (mean_norm + 1e-8)

    try:
        U, S, V = np.linalg.svd(all_emb - all_emb.mean(axis=0), full_matrices=False)
        sv_ratio = float(S[0] / (S.sum() + 1e-8))
    except:
        sv_ratio = 1.0

    try:
        n = min(500, len(all_emb))
        sil_score = float(silhouette_score(all_emb[:n], np.zeros(n)))
    except:
        sil_score = 0.0

    return {
        "mean_cosine_sim": float(np.mean(cos_sims)) if cos_sims else 0.0,
        "std_cosine_sim": float(np.std(cos_sims)) if cos_sims else 0.0,
        "mean_embedding_norm": mean_norm,
        "std_embedding_norm": std_norm,
        "collapse_score": collapse_score,
        "sv_ratio": sv_ratio,
        "silhouette_score": sil_score,
        "embeddings": all_emb,
        "labels": np.zeros(len(all_emb)),
    }


def visualize_embeddings(embeddings, labels, save_path, title="Embeddings"):
    """Create t-SNE visualization of embeddings."""
    n = min(len(embeddings), 2000)
    emb = embeddings[:n]
    lab = labels[:n]

    perplexity = min(30, n - 1) if n > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


def plot_training_curves(losses, save_path):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(losses.get("total", []))
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Step")

    axes[1].plot(losses.get("jepa", []), color="blue")
    axes[1].set_title("JEPA Loss")
    axes[1].set_xlabel("Step")

    axes[2].plot(losses.get("cosine_sim", []), color="green")
    axes[2].set_title("Cosine Similarity")
    axes[2].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")
