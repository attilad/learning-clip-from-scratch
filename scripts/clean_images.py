"""Clean the CC3M image dataset: remove corrupted and placeholder images.

Two-pass approach:
  Pass 1 (fast, no GPU): Try to decode every image with PIL. Remove files
    that can't be opened, are truncated, or have degenerate dimensions.
  Pass 2 (GPU, uses CLIP): Detect placeholder/stock images. These are
    generic "image not available" pages that websites serve instead of
    real content. They're poison for training because they pair random
    captions with identical visual content.

    Detection strategy:
    - Encode all images with pretrained CLIP
    - Placeholder images cluster tightly (high mutual similarity)
    - Real images matching their captions have high image-text similarity
    - Flag images that are both: similar to many others AND poorly matched
      to their caption

Usage:
    # Pass 1 only (fast, no GPU needed)
    uv run --no-sync python -m scripts.clean_images \
        --tsv data/cc3m/Train_GCC-training.tsv \
        --image-dir data/cc3m/images \
        --pass1-only

    # Both passes (needs GPU, uses pretrained CLIP)
    uv run --no-sync python -m scripts.clean_images \
        --tsv data/cc3m/Train_GCC-training.tsv \
        --image-dir data/cc3m/images

    # Dry run (report only, don't delete)
    uv run --no-sync python -m scripts.clean_images \
        --tsv data/cc3m/Train_GCC-training.tsv \
        --image-dir data/cc3m/images \
        --dry-run
"""

import argparse
import csv
import logging
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

from src.dataset import url_to_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---- Pass 1: Filesystem/PIL validation ----

def validate_image(path: Path) -> str | None:
    """Try to fully decode an image. Returns error string or None if OK."""
    try:
        with Image.open(path) as img:
            # .verify() checks for truncation/corruption without decoding pixels
            img.verify()

        # Re-open to check dimensions (verify() invalidates the object)
        with Image.open(path) as img:
            w, h = img.size
            if w < 10 or h < 10:
                return f"too small ({w}x{h})"
            if w > 20000 or h > 20000:
                return f"too large ({w}x{h})"
            # Force full decode to catch truncation that verify() misses
            img.load()

    except Exception as e:
        return str(e)[:100]

    return None


def pass1_validate(image_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Validate all images in the directory. Returns (valid, removed)."""
    image_files = list(image_dir.glob("*"))
    total = len(image_files)
    logger.info(f"Pass 1: Validating {total} images...")

    removed = 0
    errors_by_type: dict[str, int] = {}

    for i, path in enumerate(image_files):
        if (i + 1) % 50000 == 0:
            logger.info(f"  [{i+1}/{total}] checked, {removed} bad so far")

        error = validate_image(path)
        if error is not None:
            # Categorize the error
            if "truncat" in error.lower():
                err_type = "truncated"
            elif "too small" in error:
                err_type = "too_small"
            elif "too large" in error:
                err_type = "too_large"
            elif "cannot identify" in error.lower():
                err_type = "not_an_image"
            else:
                err_type = "other"
            errors_by_type[err_type] = errors_by_type.get(err_type, 0) + 1

            if not dry_run:
                path.unlink()
            removed += 1

    logger.info(f"Pass 1 complete: {removed}/{total} removed "
                f"({100*removed/total:.1f}%)")
    for err_type, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
        logger.info(f"  {err_type}: {count}")

    return total - removed, removed


# ---- Pass 2: CLIP-based placeholder detection ----

def pass2_detect_placeholders(
    tsv_path: Path,
    image_dir: Path,
    dry_run: bool,
    similarity_threshold: float = 0.85,
    caption_threshold: float = 0.15,
    min_cluster_size: int = 20,
    batch_size: int = 256,
) -> tuple[int, int]:
    """Detect placeholder images using pretrained CLIP.

    Strategy:
    1. Encode all images in batches
    2. Find images with very high mutual similarity (placeholder clusters)
    3. Among those, confirm they have low caption similarity (not just
       legitimately similar images like "blue sky" photos)
    4. Remove images that are both: in a high-similarity cluster AND
       poorly matched to their caption
    """
    from src.model import create_model

    device = torch.device("cuda")
    model, preprocess, tokenizer = create_model(
        model_name="ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    logger.info("Loaded pretrained CLIP for placeholder detection")

    # Build index of (caption, image_path) for existing images
    samples = []
    with open(tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            caption, url = row[0], row[1]
            img_path = image_dir / url_to_filename(url)
            if img_path.exists():
                samples.append((caption, img_path))

    logger.info(f"Pass 2: Encoding {len(samples)} images with pretrained CLIP...")

    # Encode all images in batches
    all_img_features = []
    all_txt_features = []
    valid_indices = []
    t0 = time.monotonic()

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        imgs = []
        txts = []
        batch_valid = []

        for j, (caption, img_path) in enumerate(batch_samples):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                imgs.append(img_tensor)
                txts.append(caption)
                batch_valid.append(i + j)
            except Exception:
                continue

        if not imgs:
            continue

        img_batch = torch.stack(imgs).to(device)
        txt_tokens = tokenizer(txts).to(device)

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            img_feat = F.normalize(model.encode_image(img_batch), dim=-1)
            txt_feat = F.normalize(model.encode_text(txt_tokens), dim=-1)

        all_img_features.append(img_feat.cpu())
        all_txt_features.append(txt_feat.cpu())
        valid_indices.extend(batch_valid)

        if (i + batch_size) % (batch_size * 100) == 0:
            elapsed = time.monotonic() - t0
            rate = len(valid_indices) / elapsed
            logger.info(f"  [{len(valid_indices)}/{len(samples)}] encoded @ {rate:.0f} img/s")

    img_features = torch.cat(all_img_features, dim=0).float()
    txt_features = torch.cat(all_txt_features, dim=0).float()
    n = img_features.shape[0]
    logger.info(f"Encoded {n} images in {time.monotonic() - t0:.0f}s")

    # Per-sample caption similarity (how well does this image match its text?)
    caption_sims = (img_features * txt_features).sum(dim=-1)

    # Find potential placeholders: images with low caption similarity
    low_caption_mask = caption_sims < caption_threshold
    n_low_caption = low_caption_mask.sum().item()
    logger.info(f"Images with caption similarity < {caption_threshold}: {n_low_caption}")

    # Among low-caption images, find clusters of near-duplicates
    # Only compute pairwise similarity for the low-caption subset to save memory
    low_caption_indices = torch.where(low_caption_mask)[0]

    if len(low_caption_indices) == 0:
        logger.info("No low-caption images found. Skipping cluster detection.")
        return n, 0

    low_features = img_features[low_caption_indices]

    # Process in chunks to avoid OOM on the similarity matrix
    placeholder_flags = torch.zeros(len(low_caption_indices), dtype=torch.bool)
    chunk_size = 5000

    for start in range(0, len(low_caption_indices), chunk_size):
        end = min(start + chunk_size, len(low_caption_indices))
        chunk = low_features[start:end]

        # Similarity of this chunk against ALL low-caption images
        sims = chunk @ low_features.t()

        # Count how many other images each one is highly similar to
        high_sim_counts = (sims > similarity_threshold).sum(dim=1) - 1  # exclude self

        # Flag images that are similar to many others
        placeholder_flags[start:end] = high_sim_counts >= min_cluster_size

    n_placeholders = placeholder_flags.sum().item()
    logger.info(f"Detected {n_placeholders} placeholder images "
                f"(>{min_cluster_size} near-duplicates with sim>{similarity_threshold})")

    # Map back to original indices and delete
    removed = 0
    for i, is_placeholder in enumerate(placeholder_flags):
        if is_placeholder:
            orig_idx = valid_indices[low_caption_indices[i].item()]
            _, img_path = samples[orig_idx]
            if not dry_run:
                img_path.unlink()
            removed += 1

    logger.info(f"Pass 2 complete: {removed} placeholder images "
                f"{'would be ' if dry_run else ''}removed")

    # Show some examples
    if n_placeholders > 0:
        example_indices = torch.where(placeholder_flags)[0][:5]
        logger.info("Example placeholder captions (should NOT match the image):")
        for idx in example_indices:
            orig_idx = valid_indices[low_caption_indices[idx].item()]
            caption, path = samples[orig_idx]
            sim = caption_sims[low_caption_indices[idx]].item()
            logger.info(f"  sim={sim:.3f} | {caption[:80]} | {path.name}")

    return n - removed, removed


def main():
    parser = argparse.ArgumentParser(description="Clean CC3M image dataset")
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, don't delete files")
    parser.add_argument("--pass1-only", action="store_true",
                        help="Only run PIL validation, skip CLIP detection")
    parser.add_argument("--sim-threshold", type=float, default=0.85,
                        help="Image similarity threshold for placeholder clusters")
    parser.add_argument("--caption-threshold", type=float, default=0.15,
                        help="Caption similarity below which images are suspect")
    parser.add_argument("--min-cluster", type=int, default=20,
                        help="Minimum cluster size to flag as placeholder")
    args = parser.parse_args()

    action = "Would remove" if args.dry_run else "Removing"
    logger.info(f"{'DRY RUN — ' if args.dry_run else ''}Cleaning {args.image_dir}")

    # Pass 1: PIL validation
    valid_after_p1, removed_p1 = pass1_validate(args.image_dir, args.dry_run)

    if args.pass1_only:
        logger.info(f"\nDone. {valid_after_p1} valid images remain.")
        return

    # Pass 2: CLIP placeholder detection
    valid_after_p2, removed_p2 = pass2_detect_placeholders(
        args.tsv, args.image_dir, args.dry_run,
        similarity_threshold=args.sim_threshold,
        caption_threshold=args.caption_threshold,
        min_cluster_size=args.min_cluster,
    )

    total_removed = removed_p1 + removed_p2
    logger.info(f"\nSummary:")
    logger.info(f"  Pass 1 (corrupted): {removed_p1} removed")
    logger.info(f"  Pass 2 (placeholders): {removed_p2} removed")
    logger.info(f"  Total removed: {total_removed}")
    logger.info(f"  Remaining: {valid_after_p2} images")


if __name__ == "__main__":
    main()
