"""Semantic UMAP: visualize the embedding space colored by auto-detected category.

Instead of just "image vs text", this script detects rough semantic categories
from captions (e.g. "dog" → animal, "car" → vehicle) and colors the UMAP
projection accordingly. This reveals whether the model learned meaningful
structure — do animals cluster together? Do buildings end up near vehicles
or near nature?

Usage:
    uv run --no-sync python -m scripts.semantic_umap \
        --checkpoint checkpoints/step_020000.pt
"""

import argparse
import logging
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.dataset import CC3MDataset
from src.model import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# Category detection rules: if any keyword appears in the caption,
# assign that category. Order matters — first match wins.
# These are intentionally broad to get good coverage on CC3M captions.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("person/people", [
        "person", "people", "man", "woman", "boy", "girl", "child",
        "player", "athlete", "actor", "actress", "model", "celebrity",
        "team", "crowd", "portrait", "selfie", "face",
    ]),
    ("animal", [
        "dog", "cat", "bird", "horse", "fish", "animal", "puppy", "kitten",
        "deer", "bear", "lion", "tiger", "elephant", "monkey", "rabbit",
        "cow", "sheep", "duck", "eagle", "whale", "insect", "butterfly",
        "pet", "wildlife",
    ]),
    ("vehicle", [
        "car", "truck", "bus", "train", "airplane", "plane", "boat",
        "ship", "bicycle", "bike", "motorcycle", "vehicle", "taxi",
        "helicopter", "jet",
    ]),
    ("food/drink", [
        "food", "meal", "dish", "plate", "bowl", "cake", "pizza",
        "bread", "fruit", "vegetable", "salad", "soup", "coffee",
        "wine", "beer", "restaurant", "kitchen", "cook", "chef",
        "dessert", "chocolate", "cheese",
    ]),
    ("nature/landscape", [
        "sunset", "sunrise", "mountain", "ocean", "sea", "beach",
        "forest", "tree", "flower", "garden", "lake", "river",
        "sky", "cloud", "snow", "rain", "landscape", "valley",
        "field", "park", "waterfall", "island",
    ]),
    ("building/architecture", [
        "building", "house", "church", "castle", "tower", "bridge",
        "skyscraper", "hotel", "museum", "temple", "cathedral",
        "architecture", "roof", "window", "door", "hall", "palace",
        "monument", "statue",
    ]),
    ("sports", [
        "sport", "game", "soccer", "football", "basketball", "tennis",
        "golf", "baseball", "hockey", "swimming", "running", "race",
        "stadium", "ball", "goal", "match", "championship",
    ]),
    ("art/design", [
        "art", "painting", "drawing", "design", "illustration",
        "pattern", "color", "abstract", "poster", "logo", "graphic",
        "tattoo", "graffiti",
    ]),
]


def categorize_caption(caption: str) -> str:
    """Assign a semantic category based on keyword matching."""
    caption_lower = caption.lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            # Word boundary matching to avoid false positives
            # (e.g., "cart" shouldn't match "car")
            if re.search(rf'\b{re.escape(kw)}\b', caption_lower):
                return category
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic UMAP visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-tsv", type=str,
                        default="data/cc3m/Validation_GCC-1.1.0-Validation.tsv")
    parser.add_argument("--eval-image-dir", type=str, default="data/eval/images")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/demo/semantic_umap.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Load model
    model, preprocess, tokenizer = create_model("ViT-B-32", device=device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt["step"]
    logger.info(f"Loaded checkpoint from step {step}")

    # Load dataset (raw captions needed for category detection)
    dataset = CC3MDataset(
        tsv_path=args.eval_tsv,
        image_dir=args.eval_image_dir,
        transform=preprocess,
        tokenizer=tokenizer,
    )

    # Sample and categorize
    indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))
    image_tensors = []
    text_tensors = []
    captions = []
    categories = []

    for idx in indices:
        try:
            img_tensor, txt_tensor = dataset[idx]
            caption = dataset.samples[idx][0]
            category = categorize_caption(caption)
            image_tensors.append(img_tensor)
            text_tensors.append(txt_tensor)
            captions.append(caption)
            categories.append(category)
        except Exception:
            continue

    logger.info(f"Collected {len(image_tensors)} samples")

    # Print category distribution
    from collections import Counter
    cat_counts = Counter(categories)
    logger.info("Category distribution:")
    for cat, count in cat_counts.most_common():
        logger.info(f"  {cat}: {count} ({100*count/len(categories):.0f}%)")

    # Encode in batches
    all_img_feat = []
    all_txt_feat = []
    batch_size = 64

    for i in range(0, len(image_tensors), batch_size):
        img_batch = torch.stack(image_tensors[i:i+batch_size]).to(device)
        txt_batch = torch.stack(text_tensors[i:i+batch_size]).to(device)
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            all_img_feat.append(F.normalize(model.encode_image(img_batch), dim=-1).cpu())
            all_txt_feat.append(F.normalize(model.encode_text(txt_batch), dim=-1).cpu())

    img_features = torch.cat(all_img_feat, dim=0).float().numpy()
    txt_features = torch.cat(all_txt_feat, dim=0).float().numpy()
    n = img_features.shape[0]

    # --- UMAP ---
    import umap

    # Run UMAP on images, texts, and combined separately
    logger.info("Running UMAP (this takes ~30s)...")

    combined = np.concatenate([img_features, txt_features], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(combined)

    img_emb = embedding[:n]
    txt_emb = embedding[n:]

    # Category colors
    unique_cats = sorted(set(categories))
    cmap = plt.cm.get_cmap("tab10", len(unique_cats))
    cat_to_color = {cat: cmap(i) for i, cat in enumerate(unique_cats)}
    colors = [cat_to_color[c] for c in categories]

    # --- Plot ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.4], wspace=0.3)

    # Panel 1: Images colored by semantic category
    ax1 = fig.add_subplot(gs[0])
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        ax1.scatter(
            img_emb[mask, 0], img_emb[mask, 1],
            c=[cat_to_color[cat]], alpha=0.6, s=15, label=cat,
        )
    ax1.set_title("Image Embeddings\n(colored by semantic category)", fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: Images + Texts with matched-pair lines
    ax2 = fig.add_subplot(gs[1])
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        ax2.scatter(img_emb[mask, 0], img_emb[mask, 1],
                    c=[cat_to_color[cat]], alpha=0.5, s=12, marker="o")
        ax2.scatter(txt_emb[mask, 0], txt_emb[mask, 1],
                    c=[cat_to_color[cat]], alpha=0.5, s=12, marker="x")

    # Draw lines for a subset of matched pairs
    for i in range(0, n, max(1, n // 80)):
        ax2.plot([img_emb[i, 0], txt_emb[i, 0]],
                 [img_emb[i, 1], txt_emb[i, 1]],
                 c=colors[i], alpha=0.2, linewidth=0.5)

    ax2.set_title("Images (●) + Texts (×)\n(lines connect matched pairs)", fontsize=13)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Legend
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    for i, cat in enumerate(unique_cats):
        count = cat_counts[cat]
        pct = 100 * count / n
        ax3.scatter([], [], c=[cat_to_color[cat]], s=60, label=f"{cat} ({count}, {pct:.0f}%)")
    ax3.legend(loc="center left", fontsize=11, frameon=False, title="Categories",
               title_fontsize=13)

    fig.suptitle(
        f"Semantic UMAP — Step {step} — {n} samples\n"
        f"Does the model organize the world into meaningful clusters?",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {output_path}")

    # --- Cluster quality metric ---
    # For each category with >10 samples, compute the average intra-category
    # cosine similarity vs average inter-category similarity.
    # If intra > inter, the category forms a real cluster.
    print("\n" + "=" * 60)
    print("Cluster Quality (image embeddings)")
    print("=" * 60)
    print(f"{'Category':<22} {'Intra-sim':>10} {'Inter-sim':>10} {'Gap':>8} {'Clustered?':>10}")
    print("-" * 62)

    img_feat_tensor = torch.from_numpy(img_features)
    for cat in unique_cats:
        mask = torch.tensor([c == cat for c in categories])
        if mask.sum() < 10:
            continue

        cat_features = img_feat_tensor[mask]
        other_features = img_feat_tensor[~mask]

        # Intra-category: average pairwise similarity within the category
        intra_sim = (cat_features @ cat_features.t())
        # Exclude self-similarity (diagonal)
        intra_mask = ~torch.eye(intra_sim.shape[0], dtype=torch.bool)
        intra_avg = intra_sim[intra_mask].mean().item()

        # Inter-category: average similarity to samples outside the category
        inter_sim = (cat_features @ other_features.t())
        inter_avg = inter_sim.mean().item()

        gap = intra_avg - inter_avg
        clustered = "✓ yes" if gap > 0.02 else "~ weak" if gap > 0 else "✗ no"

        print(f"{cat:<22} {intra_avg:>10.4f} {inter_avg:>10.4f} {gap:>+8.4f} {clustered:>10}")


if __name__ == "__main__":
    main()
