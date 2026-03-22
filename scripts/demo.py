"""Interactive demo: see your trained CLIP model in action.

Three demos that show what CLIP actually learned:

1. Zero-shot classification: give it an image + candidate labels,
   see which label it picks (no training on those labels!)
2. Text-to-image retrieval: give it a text query, find the best
   matching images from the eval set
3. Similarity heatmap: visualize the full similarity matrix between
   a batch of images and captions

Usage:
    uv run --no-sync python -m scripts.demo \
        --checkpoint checkpoints/step_020000.pt
"""

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

from src.dataset import CC3MDataset
from src.model import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ImageNet normalization inverse (for display)
IMAGENET_MEAN = torch.tensor([0.4815, 0.4578, 0.4082])
IMAGENET_STD = torch.tensor([0.2686, 0.2613, 0.2758])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display."""
    img = tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
):
    """Load model + checkpoint, return (model, preprocess, tokenizer)."""
    model, preprocess, tokenizer = create_model(
        model_name="ViT-B-32", device=device
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt["step"]
    logger.info(f"Loaded checkpoint from step {step}")
    return model, preprocess, tokenizer, step


@torch.no_grad()
def demo_zero_shot(
    model,
    preprocess,
    tokenizer,
    image_paths: list[Path],
    candidate_labels: list[str],
    device: torch.device,
    output_path: Path,
) -> None:
    """Zero-shot classification: rank candidate labels for each image.

    This is the magic of CLIP — the model has never been trained to
    classify these specific categories, but it can rank them by how
    well they match the image, using the shared embedding space it
    learned from image-caption pairs.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Zero-Shot Classification")
    print("=" * 60)
    print(f"Candidate labels: {candidate_labels}")

    # Encode all candidate labels once
    prompts = [f"a photo of {label}" for label in candidate_labels]
    text_tokens = tokenizer(prompts).to(device)
    with autocast("cuda", dtype=torch.bfloat16):
        text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

    n_images = len(image_paths)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    if n_images == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            img_features = F.normalize(model.encode_image(img_tensor), dim=-1)

        # Cosine similarities → probabilities via softmax
        similarities = (img_features @ text_features.t()).squeeze(0).float()
        probs = similarities.softmax(dim=0).cpu().numpy()

        row, col = divmod(i, cols)
        ax = axes[row][col]
        ax.imshow(img)
        ax.axis("off")

        # Build label with probabilities
        ranked = sorted(zip(candidate_labels, probs), key=lambda x: -x[1])
        title_lines = [f"Top: {ranked[0][0]} ({ranked[0][1]:.1%})"]
        for label, prob in ranked[1:4]:
            title_lines.append(f"  {label}: {prob:.1%}")
        ax.set_title("\n".join(title_lines), fontsize=9, ha="left",
                     x=0.02, fontfamily="monospace")

        print(f"\n{img_path.name}:")
        for label, prob in ranked:
            bar = "█" * int(prob * 30)
            print(f"  {prob:5.1%} {bar} {label}")

    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].axis("off")

    fig.suptitle("Zero-Shot Classification (no training on these labels!)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {output_path}")


@torch.no_grad()
def demo_text_retrieval(
    model,
    preprocess,
    tokenizer,
    dataset: CC3MDataset,
    queries: list[str],
    device: torch.device,
    output_path: Path,
    n_candidates: int = 500,
    top_k: int = 5,
) -> None:
    """Text-to-image retrieval: find the best images for a text query.

    Encodes a pool of candidate images, then ranks them by similarity
    to each text query. This is what CLIP enables for search engines.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Text → Image Retrieval")
    print("=" * 60)

    # Encode a pool of candidate images
    indices = random.sample(range(len(dataset)), min(n_candidates, len(dataset)))
    all_images = []
    all_paths = []
    valid_indices = []

    for idx in indices:
        try:
            img_tensor, _ = dataset[idx]
            all_images.append(img_tensor)
            all_paths.append(dataset.samples[idx][1])
            valid_indices.append(idx)
        except Exception:
            continue
        if len(all_images) >= n_candidates:
            break

    image_batch = torch.stack(all_images).to(device)
    with autocast("cuda", dtype=torch.bfloat16):
        image_features = F.normalize(model.encode_image(image_batch), dim=-1)

    n_queries = len(queries)
    fig = plt.figure(figsize=(4 * top_k, 4.5 * n_queries))
    gs = gridspec.GridSpec(n_queries, top_k, hspace=0.4, wspace=0.1)

    for q_idx, query in enumerate(queries):
        print(f"\nQuery: \"{query}\"")

        # Encode the query
        text_tokens = tokenizer([query]).to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

        similarities = (text_features @ image_features.t()).squeeze(0).float()
        topk_vals, topk_idx = similarities.topk(top_k)

        for rank, (sim, img_idx) in enumerate(zip(topk_vals, topk_idx)):
            img_path = all_paths[img_idx.item()]
            caption = dataset.samples[valid_indices[img_idx.item()]][0]

            print(f"  #{rank+1} (sim={sim.item():.3f}): {caption[:70]}")

            ax = fig.add_subplot(gs[q_idx, rank])
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"sim={sim.item():.3f}", fontsize=8)
            if rank == 0:
                ax.set_ylabel(f'"{query}"', fontsize=10, fontweight="bold",
                              rotation=0, labelpad=80, va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Text → Image Retrieval (top 5 matches from 500 candidates)",
                 fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {output_path}")


@torch.no_grad()
def demo_similarity_heatmap(
    model,
    preprocess,
    tokenizer,
    dataset: CC3MDataset,
    device: torch.device,
    output_path: Path,
    n_samples: int = 8,
) -> None:
    """Visualize the similarity matrix between images and captions.

    The ideal matrix has high values on the diagonal (correct matches)
    and low values everywhere else. This shows you exactly what the
    model "sees" when comparing images and texts.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Similarity Heatmap")
    print("=" * 60)

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    images = []
    captions = []
    img_paths = []

    for idx in indices:
        try:
            img_tensor, _ = dataset[idx]
            caption = dataset.samples[idx][0]
            img_path = dataset.samples[idx][1]
            images.append(img_tensor)
            captions.append(caption[:50])  # Truncate for display
            img_paths.append(img_path)
        except Exception:
            continue
        if len(images) >= n_samples:
            break

    image_batch = torch.stack(images).to(device)
    text_tokens = tokenizer(captions).to(device)

    with autocast("cuda", dtype=torch.bfloat16):
        image_features = F.normalize(model.encode_image(image_batch), dim=-1)
        text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

    sim_matrix = (image_features @ text_features.t()).float().cpu().numpy()
    n = len(images)

    # Layout: images on top, heatmap below
    fig = plt.figure(figsize=(max(12, n * 2), max(10, n * 1.5 + 4)))
    gs = gridspec.GridSpec(2, n, height_ratios=[1, n * 0.6], hspace=0.3)

    # Show thumbnail images across the top
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        img = Image.open(img_paths[i]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"img_{i}", fontsize=8)

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, :])
    im = ax_heat.imshow(sim_matrix, cmap="RdYlGn", aspect="auto",
                         vmin=sim_matrix.min(), vmax=sim_matrix.max())

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            color = "white" if abs(sim_matrix[i, j]) > 0.3 else "black"
            ax_heat.text(j, i, f"{sim_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8, color=color)

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels([f"txt_{i}: {c[:25]}..." for i, c in enumerate(captions)],
                             rotation=45, ha="right", fontsize=7)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels([f"img_{i}" for i in range(n)], fontsize=8)
    ax_heat.set_xlabel("Text captions")
    ax_heat.set_ylabel("Images")

    fig.colorbar(im, ax=ax_heat, shrink=0.6, label="Cosine similarity")
    fig.suptitle("Image-Text Similarity Matrix\n(diagonal = correct pairs)",
                 fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {output_path}")

    # Print the matrix
    print("\nSimilarity matrix (rows=images, cols=texts):")
    print("Diagonal should be highest in each row/column if model works well.\n")
    header = "        " + "  ".join(f"txt_{i:>2}" for i in range(n))
    print(header)
    for i in range(n):
        row = f"img_{i}  " + "  ".join(
            f" {sim_matrix[i,j]:.2f}" if i != j else f"[{sim_matrix[i,j]:.2f}]"
            for j in range(n)
        )
        print(row)

    # Score: how often is the diagonal the max in its row?
    correct = sum(sim_matrix[i].argmax() == i for i in range(n))
    print(f"\nDiagonal is highest in row: {correct}/{n} ({100*correct/n:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="CLIP model demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-tsv", type=str,
                        default="data/cc3m/Validation_GCC-1.1.0-Validation.tsv")
    parser.add_argument("--eval-image-dir", type=str, default="data/eval/images")
    parser.add_argument("--output-dir", type=str, default="data/demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, preprocess, tokenizer, step = load_model_from_checkpoint(
        args.checkpoint, device
    )

    # Load eval dataset (for retrieval and heatmap demos)
    dataset = CC3MDataset(
        tsv_path=args.eval_tsv,
        image_dir=args.eval_image_dir,
        transform=preprocess,
        tokenizer=tokenizer,
    )

    # --- Demo 1: Zero-shot classification ---
    # Pick some random eval images and classify them with made-up categories
    sample_paths = [dataset.samples[i][1] for i in random.sample(range(len(dataset)), 8)]
    zero_shot_labels = [
        "a dog", "a cat", "a car", "food", "a building",
        "a person", "nature", "art", "sports", "an animal",
    ]
    demo_zero_shot(
        model, preprocess, tokenizer,
        image_paths=sample_paths,
        candidate_labels=zero_shot_labels,
        device=device,
        output_path=output_dir / "zero_shot.png",
    )

    # --- Demo 2: Text retrieval ---
    queries = [
        "a dog playing in the park",
        "a sunset over the ocean",
        "a plate of food",
        "a city skyline at night",
    ]
    demo_text_retrieval(
        model, preprocess, tokenizer,
        dataset=dataset,
        queries=queries,
        device=device,
        output_path=output_dir / "retrieval.png",
    )

    # --- Demo 3: Similarity heatmap ---
    demo_similarity_heatmap(
        model, preprocess, tokenizer,
        dataset=dataset,
        device=device,
        output_path=output_dir / "heatmap.png",
    )

    print("\n" + "=" * 60)
    print(f"All demos saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
