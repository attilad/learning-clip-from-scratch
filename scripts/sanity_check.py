"""Sanity check: load a few CC3M samples and display images + captions.

Verifies the full data pipeline end-to-end:
  TSV → Dataset (with image hashing) → transform → tokenizer → batch

Also prints dataset stats and timing for a single DataLoader iteration
so you can catch bottlenecks before committing to a full training run.

Usage:
    uv run python -m scripts.sanity_check \
        --tsv data/cc3m/Train_GCC-training.tsv \
        --image-dir data/cc3m/images \
        --num-samples 8
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import CC3MDataset, create_dataloader
from src.model import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ImageNet stats — open_clip's preprocessing uses these for normalization.
# We need the inverse to undo it for visualization.
IMAGENET_MEAN = torch.tensor([0.4815, 0.4578, 0.4082])
IMAGENET_STD = torch.tensor([0.2686, 0.2613, 0.2758])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display. [C,H,W] → [H,W,C] in [0,1]."""
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return img.clamp(0, 1).permute(1, 2, 0)


def show_samples(
    dataset: CC3MDataset,
    num_samples: int = 8,
    output_path: Path = Path("data/sanity_check.png"),
) -> None:
    """Display a grid of sample images with their captions."""
    num_samples = min(num_samples, len(dataset))
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(num_samples):
        image_tensor, token_ids = dataset[i]
        caption = dataset.samples[i][0]  # Raw caption text

        row, col = divmod(i, cols)
        ax = axes[row][col]

        img = denormalize(image_tensor)
        ax.imshow(img.numpy())
        # Truncate long captions so they fit under the image
        title = caption[:80] + "..." if len(caption) > 80 else caption
        ax.set_title(title, fontsize=8, wrap=True)
        ax.axis("off")

    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sample grid saved to {output_path}")


def benchmark_dataloader(
    dataset: CC3MDataset,
    batch_size: int = 256,
    num_batches: int = 10,
) -> None:
    """Time a few DataLoader iterations to catch pipeline bottlenecks."""
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(
        f"\nBenchmarking DataLoader: batch_size={batch_size}, "
        f"num_workers={loader.num_workers}, {num_batches} batches..."
    )

    times = []
    for i, (images, texts) in enumerate(loader):
        if i >= num_batches:
            break
        t0 = time.perf_counter()
        # Simulate what training does: move to GPU
        images = images.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    if times:
        avg = sum(times) / len(times)
        logger.info(f"Avg batch transfer time: {avg*1000:.1f}ms")
        logger.info(f"Throughput: ~{batch_size / avg:.0f} samples/s (transfer only)")


def main() -> None:
    parser = argparse.ArgumentParser(description="CC3M data pipeline sanity check")
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("data/sanity_check.png"))
    parser.add_argument("--benchmark", action="store_true", help="Run DataLoader benchmark")
    args = parser.parse_args()

    # Use open_clip's preprocessing — this is the exact transform used
    # during training, so we're testing the real pipeline, not a toy version
    logger.info("Initializing model (for transforms + tokenizer)...")
    _, preprocess, tokenizer = create_model(model_name="ViT-B-32")

    logger.info(f"Loading dataset from {args.tsv}...")
    dataset = CC3MDataset(
        tsv_path=args.tsv,
        image_dir=args.image_dir,
        transform=preprocess,
        tokenizer=tokenizer,
    )

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Show sample images + captions
    show_samples(dataset, num_samples=args.num_samples, output_path=args.output)

    # Print token shapes for a single sample to verify tokenizer output
    image_tensor, token_ids = dataset[0]
    logger.info(f"Image tensor shape: {image_tensor.shape}")
    logger.info(f"Image tensor range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
    logger.info(f"Token IDs shape: {token_ids.shape}")
    logger.info(f"Token IDs dtype: {token_ids.dtype}")

    if args.benchmark:
        benchmark_dataloader(dataset)

    logger.info("\nSanity check complete.")


if __name__ == "__main__":
    main()
