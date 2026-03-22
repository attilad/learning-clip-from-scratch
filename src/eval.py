"""Zero-shot evaluation: recall@K and embedding space diagnostics."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def compute_recall_at_k(
    model: nn.Module,
    dataloader: DataLoader,
    ks: tuple[int, ...] = (1, 5),
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute image→text and text→image recall@K on a held-out set.

    Args:
        model: CLIP model in eval mode.
        dataloader: Yields (image_batch, text_batch).
        ks: Which K values to compute recall for.
        device: Target device.

    Returns:
        Dict with keys like "i2t_recall@1", "t2i_recall@5", etc.
    """
    if device is None:
        device = torch.device("cuda")

    # Free training VRAM before encoding the eval set
    torch.cuda.empty_cache()

    all_image_features = []
    all_text_features = []

    for images, texts in dataloader:
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        # Use BF16 for eval encoding too — saves VRAM and we don't
        # need FP32 precision for retrieval ranking
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_features = F.normalize(model.encode_image(images), dim=-1)
            text_features = F.normalize(model.encode_text(texts), dim=-1)

        # Move to CPU immediately to free GPU memory for next batch
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())

    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    # Similarity matrix: [N_images, N_texts]
    # Computed on CPU to avoid GPU OOM on large eval sets
    sim_matrix = image_features.float() @ text_features.float().t()

    n = sim_matrix.shape[0]
    labels = torch.arange(n)

    results = {}
    for k in ks:
        # Image → Text: for each image, are the correct text(s) in top-K?
        i2t_topk = sim_matrix.topk(k, dim=1).indices
        i2t_correct = (i2t_topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"i2t_recall@{k}"] = i2t_correct

        # Text → Image: for each text, are the correct image(s) in top-K?
        t2i_topk = sim_matrix.t().topk(k, dim=1).indices
        t2i_correct = (t2i_topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"t2i_recall@{k}"] = t2i_correct

    return results


def log_eval_results(
    results: dict[str, float],
    step: int,
    writer: SummaryWriter | None = None,
) -> None:
    """Print and optionally log eval metrics to TensorBoard."""
    parts = [f"[eval step {step}]"]
    for key, val in results.items():
        parts.append(f"{key}={val:.4f}")
        if writer is not None:
            writer.add_scalar(f"eval/{key}", val, step)
    print("  ".join(parts))


def generate_umap_visualization(
    model: nn.Module,
    dataloader: DataLoader,
    output_path: str | Path,
    max_samples: int = 1000,
    device: torch.device | None = None,
) -> Path:
    """Generate UMAP projection of image + text embeddings.

    This is a diagnostic tool, not an optimization target. Use it to
    visually check whether the embedding space has meaningful structure.
    """
    import matplotlib.pyplot as plt
    import umap

    if device is None:
        device = torch.device("cuda")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_features = []
    text_features = []
    collected = 0

    with torch.no_grad():
        for images, texts in dataloader:
            if collected >= max_samples:
                break
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            img_feat = F.normalize(model.encode_image(images), dim=-1)
            txt_feat = F.normalize(model.encode_text(texts), dim=-1)

            image_features.append(img_feat.cpu())
            text_features.append(txt_feat.cpu())
            collected += images.shape[0]

    image_features = torch.cat(image_features, dim=0)[:max_samples]
    text_features = torch.cat(text_features, dim=0)[:max_samples]

    # Combine for joint UMAP — lets us see if images and texts
    # are mapping to the same regions of embedding space
    combined = torch.cat([image_features, text_features], dim=0).numpy()
    n_images = image_features.shape[0]

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        embedding[:n_images, 0], embedding[:n_images, 1],
        c="steelblue", alpha=0.5, s=10, label="images",
    )
    ax.scatter(
        embedding[n_images:, 0], embedding[n_images:, 1],
        c="coral", alpha=0.5, s=10, label="texts",
    )
    ax.legend()
    ax.set_title("UMAP: Image + Text Embeddings")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"UMAP visualization saved to {output_path}")
    return output_path
