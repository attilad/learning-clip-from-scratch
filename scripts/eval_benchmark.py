"""Multi-dataset zero-shot evaluation using clip_benchmark datasets.

Uses clip_benchmark's dataset infrastructure (standardized class names and
prompt templates) but our own zero-shot classification logic — avoids a
numpy 2.x compatibility bug in clip_benchmark's accuracy function.

Usage:
    # Evaluate one model on one dataset
    uv run python -m scripts.eval_benchmark --pretrained openai --dataset cifar100

    # Evaluate all exp 008 checkpoints (called by run_009.sh)
    uv run python -m scripts.eval_benchmark --pretrained openai --dataset all
"""

import argparse
import json
import logging
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from clip_benchmark.datasets.builder import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = [
    "cifar100",
    "caltech101",
    "food101",
    "oxford_flowers102",
    "fgvc_aircraft",
    "sun397",
    "dtd",
    "eurosat",
    "gtsrb",
]


@torch.no_grad()
def zero_shot_classify(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    """Zero-shot classification on a dataset with class names and templates.

    Uses the same approach as our src/zero_shot_classify.py but generalized
    to any dataset that has .classes and .templates attributes (which
    clip_benchmark's build_dataset provides).
    """
    class_names = dataset.classes
    templates = dataset.templates

    # Build text classifier: average embeddings across templates per class
    text_features_per_class = []
    for class_name in class_names:
        prompts = [t(class_name) if callable(t) else t.format(c=class_name) for t in templates]
        tokens = tokenizer(prompts).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.encode_text(tokens)
            features = F.normalize(features, dim=-1)

        class_embedding = F.normalize(features.mean(dim=0), dim=0)
        text_features_per_class.append(class_embedding)

    text_classifier = torch.stack(text_features_per_class)  # [num_classes, embed_dim]

    # Classify all images
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
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

        logits = image_features @ text_classifier.t()

        correct_top1 += (logits.argmax(dim=1) == labels).sum().item()

        k = min(5, logits.shape[1])
        top5 = logits.topk(k, dim=1).indices
        correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.shape[0]

    return {
        "acc1": correct_top1 / total,
        "acc5": correct_top5 / total,
        "num_samples": total,
        "num_classes": len(class_names),
    }


def load_model(pretrained: str, device: torch.device):
    """Load ViT-B-32 with either an open_clip tag or a local checkpoint path."""
    if pretrained == "openai" or pretrained.startswith("laion"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=pretrained, device=device
        )
    else:
        # Local checkpoint: create model, then load state dict
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=None, device=device
        )
        state_dict = torch.load(pretrained, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-dataset zero-shot eval")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="open_clip pretrained tag or path to exported checkpoint")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all' for all datasets")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="experiments/009_benchmark/data")
    parser.add_argument("--output-dir", type=str, default="experiments/009_benchmark/results")
    parser.add_argument("--name", type=str, default=None,
                        help="Model name for output files (derived from pretrained if not set)")
    args = parser.parse_args()

    device = torch.device("cuda")
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    # Derive model name for output files
    if args.name:
        model_name = args.name
    elif args.pretrained in ("openai",):
        model_name = "baseline"
    else:
        model_name = Path(args.pretrained).stem

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.pretrained}")
    model, preprocess, tokenizer = load_model(args.pretrained, device)

    for dataset_name in datasets:
        output_file = output_dir / f"{model_name}_{dataset_name}.json"

        if output_file.exists():
            logger.info(f"  {dataset_name}: already done, skipping")
            continue

        logger.info(f"  Evaluating on {dataset_name}...")
        try:
            dataset = build_dataset(
                dataset_name, root=args.data_root,
                transform=preprocess, split="test",
            )
        except Exception as e:
            logger.warning(f"  {dataset_name}: failed to load ({e}), skipping")
            continue

        torch.cuda.empty_cache()
        metrics = zero_shot_classify(
            model, tokenizer, dataset, device,
            batch_size=args.batch_size,
        )

        logger.info(
            f"  {dataset_name}: top1={metrics['acc1']:.4f} top5={metrics['acc5']:.4f} "
            f"({metrics['num_samples']} samples, {metrics['num_classes']} classes)"
        )

        with open(output_file, "w") as f:
            json.dump({"metrics": metrics, "dataset": dataset_name, "model": model_name}, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
