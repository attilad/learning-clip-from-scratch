"""Export exp 008 checkpoints to clip_benchmark-compatible format.

Our training checkpoints save {"model_state_dict": ..., "optimizer_state_dict": ...},
but clip_benchmark expects raw open_clip state dicts. This script extracts and
converts each checkpoint, including merging LoRA deltas into base weights.

Usage:
    uv run python -m scripts.export_checkpoints
"""

import logging
from pathlib import Path

import torch

from src.adapt import merge_lora_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path("experiments/008_adaptation")
OUTPUT_DIR = Path("experiments/009_benchmark/exported")


def extract_model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load a training checkpoint and return just the model state dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    # wise_ft.pt already has model_state_dict at top level
    return ckpt


def export_standard(name: str, checkpoint_path: Path) -> Path:
    """Export a standard (non-LoRA) checkpoint."""
    state_dict = extract_model_state(checkpoint_path)
    output_path = OUTPUT_DIR / f"{name}.pt"
    torch.save(state_dict, output_path)
    logger.info(f"Exported {name}: {len(state_dict)} keys -> {output_path}")
    return output_path


def export_lora(name: str, checkpoint_path: Path, rank: int = 4) -> Path:
    """Export a LoRA checkpoint by merging adapters into base weights."""
    state_dict = extract_model_state(checkpoint_path)
    merged = merge_lora_state_dict(state_dict, rank=rank)
    output_path = OUTPUT_DIR / f"{name}.pt"
    torch.save(merged, output_path)
    logger.info(f"Exported {name} (merged LoRA): {len(merged)} keys -> {output_path}")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exports = {
        "full_ft": (EXP_DIR / "checkpoints_a" / "step_005000.pt", "standard"),
        "freeze": (EXP_DIR / "checkpoints_b" / "step_005000.pt", "standard"),
        "lora_r4": (EXP_DIR / "checkpoints_d" / "step_005000.pt", "lora"),
        "wise_ft": (EXP_DIR / "checkpoints_e" / "wise_ft.pt", "standard"),
    }

    for name, (path, method) in exports.items():
        if not path.exists():
            logger.warning(f"Skipping {name}: {path} not found")
            continue

        if method == "lora":
            export_lora(name, path)
        else:
            export_standard(name, path)

    # Verify all exported checkpoints load into a standard open_clip model
    logger.info("\nVerifying exported checkpoints...")
    import open_clip
    for pt_file in sorted(OUTPUT_DIR.glob("*.pt")):
        state_dict = torch.load(pt_file, map_location="cpu", weights_only=False)
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=None, device="cpu"
        )
        # strict=False allows minor mismatches (e.g., logit_scale)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        status = "OK" if not unexpected else f"unexpected: {unexpected[:3]}"
        logger.info(f"  {pt_file.name}: {status} (missing: {len(missing)})")

    logger.info(f"\nAll exports in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
