"""Smoke test: verify GPU, model init, and a forward pass with synthetic data.

Run this before any real training to catch environment issues early.
Usage: uv run python scripts/smoke_test.py
"""

import sys
import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.model import create_model
from src.loss import CLIPLoss


def main() -> None:
    print("=" * 60)
    print("CLIP Smoke Test")
    print("=" * 60)

    # --- GPU check ---
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device_name} ({vram:.1f} GB)")

    # --- Model init ---
    print("\nInitializing ViT-B-32 (random weights)...")
    model, preprocess, tokenizer = create_model(model_name="ViT-B-32")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.1f}M")

    # --- Synthetic forward pass ---
    print("\nRunning forward pass with synthetic batch (batch_size=4)...")
    batch_size = 4
    device = torch.device("cuda")

    # Fake images: random tensors matching expected input shape
    dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_texts = tokenizer(["a dog", "a cat", "a sunset", "a building"])
    dummy_texts = dummy_texts.to(device)

    model.eval()
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        image_features = model.encode_image(dummy_images)
        text_features = model.encode_text(dummy_texts)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

    assert image_features.shape == (batch_size, image_features.shape[1]), (
        f"Unexpected image feature shape: {image_features.shape}"
    )
    assert text_features.shape == image_features.shape, (
        f"Shape mismatch: {text_features.shape} vs {image_features.shape}"
    )
    print(f"Image features: {image_features.shape}")
    print(f"Text features:  {text_features.shape}")

    # --- Loss computation ---
    print("\nComputing contrastive loss...")
    loss_fn = CLIPLoss().to(device)
    loss, metrics = loss_fn(image_features, text_features)
    print(f"Loss: {loss.item():.4f}")
    print(f"Temperature: {metrics['logit_scale']:.2f}")
    print(f"I2T accuracy: {metrics['i2t_acc']:.3f}")
    print(f"T2I accuracy: {metrics['t2i_acc']:.3f}")

    # --- Cosine similarity matrix ---
    sim = image_features @ text_features.t()
    print(f"\nCosine similarity matrix:\n{sim.cpu().float()}")

    # --- Memory usage ---
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU memory used: {mem:.2f} GB")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
