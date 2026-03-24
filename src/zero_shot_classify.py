"""Zero-shot classification on CIFAR-100 — measures general visual knowledge.

This is the out-of-distribution metric for exp 008. A pretrained CLIP model
should get ~60% top-1 on CIFAR-100 zero-shot. If fine-tuning on CC3M drops
this significantly, we've measured catastrophic forgetting.

The approach: encode all 100 class names as text ("a photo of a {class}"),
encode all test images, classify each image by its nearest text embedding.
No training involved — pure inference.
"""

import logging
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

logger = logging.getLogger(__name__)

# Prompt ensembling improves zero-shot accuracy by ~3-5% over a single template.
# These templates are from the original CLIP paper (Radford et al., 2021).
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of a {} in a video game.",
    "a photo of many {}.",
    "a photo of the hard to see {}.",
]


@torch.no_grad()
def cifar100_zero_shot(
    model: nn.Module,
    tokenizer: open_clip.tokenizer,
    preprocess,
    data_dir: str | Path = "data/cifar100",
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate zero-shot classification accuracy on CIFAR-100 test set.

    Args:
        model: CLIP model in eval mode.
        tokenizer: open_clip tokenizer for encoding class names.
        preprocess: Image transform (from open_clip).
        data_dir: Where to download/cache CIFAR-100.
        device: Target device.

    Returns:
        Dict with "cifar100_top1" and "cifar100_top5" accuracy.
    """
    if device is None:
        device = torch.device("cuda")

    torch.cuda.empty_cache()

    # Download CIFAR-100 test set (first run downloads ~160MB)
    dataset = CIFAR100(
        root=str(data_dir),
        train=False,
        download=True,
        transform=preprocess,
    )
    class_names = dataset.classes  # 100 fine-grained class names

    logger.info(f"CIFAR-100: {len(dataset)} test images, {len(class_names)} classes")

    # --- Build text classifier weights via prompt ensembling ---
    # For each class, average the text embeddings across all prompt templates.
    # This creates a single "centroid" embedding per class that's more robust
    # than any single prompt.
    text_features_per_class = []
    for class_name in class_names:
        prompts = [template.format(class_name) for template in PROMPT_TEMPLATES]
        tokens = tokenizer(prompts).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.encode_text(tokens)
            features = F.normalize(features, dim=-1)

        # Average across templates, then re-normalize
        class_embedding = features.mean(dim=0)
        class_embedding = F.normalize(class_embedding, dim=0)
        text_features_per_class.append(class_embedding)

    # [100, embed_dim] — one row per class
    text_classifier = torch.stack(text_features_per_class)

    # --- Classify all test images ---
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        # [batch, 100] — cosine similarity to each class
        logits = image_features @ text_classifier.t()

        # Top-1
        preds = logits.argmax(dim=1)
        correct_top1 += (preds == labels).sum().item()

        # Top-5
        top5_preds = logits.topk(5, dim=1).indices
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.shape[0]

    top1 = correct_top1 / total
    top5 = correct_top5 / total

    logger.info(f"CIFAR-100 zero-shot: top1={top1:.4f}, top5={top5:.4f}")

    return {"cifar100_top1": top1, "cifar100_top5": top5}
