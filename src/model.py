"""CLIP model initialization via open_clip."""

import open_clip
import torch
import torch.nn as nn


def create_model(
    model_name: str = "ViT-B-32",
    pretrained: str | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, open_clip.transform.image_transform, open_clip.tokenizer]:
    """Create a CLIP model, image transform, and tokenizer.

    Args:
        model_name: Architecture name (open_clip format, e.g. "ViT-B-32").
        pretrained: Pretrained weights tag, or None for random init.
        device: Target device. Raises if CUDA is unavailable.

    Returns:
        (model, preprocess, tokenizer) tuple.
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This project requires a GPU — "
                "no silent fallback to CPU."
            )
        device = torch.device("cuda")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer
