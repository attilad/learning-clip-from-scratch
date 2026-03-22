"""CC3M dataset: downloading, loading, and preprocessing for CLIP training."""

import csv
import hashlib
import io
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CC3MDataset(Dataset):
    """Conceptual Captions 3M dataset for CLIP training.

    Two-phase design:
      1. Download phase: run `scripts/download_cc3m.py` to async-download
         images from URLs into `image_dir/` using content-hashed filenames.
      2. Training phase: this Dataset loads the index of successfully
         downloaded images and serves (image_tensor, token_ids) pairs.

    The TSV format is: caption<TAB>url (the original CC3M format).
    We build a filtered index at init time — only samples whose images
    actually exist on disk are included. This makes the Dataset robust
    to partial downloads without needing a separate manifest.
    """

    def __init__(
        self,
        tsv_path: str | Path,
        image_dir: str | Path,
        transform=None,
        tokenizer=None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}. "
                f"Run `uv run python -m scripts.download_cc3m` first."
            )

        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        # Build index: only include samples where the image was downloaded
        self.samples: list[tuple[str, Path]] = []
        skipped = 0

        with open(tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    skipped += 1
                    continue
                caption, url = row[0], row[1]
                image_path = self.image_dir / url_to_filename(url)
                if image_path.exists():
                    self.samples.append((caption, image_path))
                else:
                    skipped += 1

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found. TSV has entries but no images in "
                f"{self.image_dir}. Run the downloader first."
            )

        logger.info(
            f"CC3M dataset: {len(self.samples)} samples loaded, "
            f"{skipped} skipped (missing images or malformed rows)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        caption, image_path = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Corrupt file on disk — return a random other sample rather
            # than crashing the DataLoader. Log so we can clean up later.
            logger.warning(f"Failed to load {image_path}: {e}")
            return self[torch.randint(len(self), (1,)).item()]

        if self.transform is not None:
            image = self.transform(image)

        if self.tokenizer is not None:
            text = self.tokenizer([caption])[0]
        else:
            text = caption

        return image, text


def url_to_filename(url: str) -> str:
    """Deterministic mapping from URL to filename.

    Uses a content hash so we get a flat directory structure with no
    path-separator issues, and can deduplicate if the same image appears
    under multiple URLs. The extension is preserved for image format hints.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    # Try to preserve the original extension for PIL format detection
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        suffix = ".jpg"  # Safe default — PIL will read by magic bytes anyway
    return f"{url_hash}{suffix}"


def create_dataloader(
    dataset: CC3MDataset,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader tuned for CLIP training on a 4090.

    Worker count rationale: 8 workers saturate the PCIe bus for image
    decoding without hitting WSL2 shared memory limits. If you see
    "RuntimeError: DataLoader worker ... exited unexpectedly", try
    reducing to 4 or adding --shm-size to your WSL config.

    pin_memory=True: pre-stages tensors in page-locked RAM so the
    CUDA transfer is a single async DMA copy instead of a pageable
    memcpy. Free performance on dedicated GPU systems.

    drop_last=True: contrastive loss builds an [B,B] similarity matrix,
    so the last incomplete batch would have different geometry. Dropping
    it costs <1 batch per epoch — negligible on 3M samples.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
