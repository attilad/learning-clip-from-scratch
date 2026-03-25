"""Analyze exp 009 CLIP Benchmark results.

Parses JSON output files from clip_benchmark, builds comparison tables,
and computes forgetting metrics across all methods and datasets.

Usage:
    uv run python -m scripts.analyze_009
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/009_benchmark/results")

# Datasets ordered by expected distance from CC3M distribution
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

METHODS = ["baseline", "full_ft", "freeze", "lora_r4", "wise_ft"]

METHOD_LABELS = {
    "baseline": "Pretrained",
    "full_ft": "Full FT",
    "freeze": "Freeze",
    "lora_r4": "LoRA r=4",
    "wise_ft": "WiSE-FT",
}

# Rough categorization by distance from CC3M's distribution
DATASET_CATEGORIES = {
    "cifar100": "near",
    "caltech101": "near",
    "food101": "near",
    "oxford_flowers102": "medium",
    "fgvc_aircraft": "medium",
    "sun397": "medium",
    "dtd": "far",
    "eurosat": "far",
    "gtsrb": "far",
}


def load_results() -> dict[str, dict[str, float]]:
    """Load all results into {method: {dataset: accuracy}} nested dict."""
    results = {}
    for method in METHODS:
        results[method] = {}
        for dataset in DATASETS:
            path = RESULTS_DIR / f"{method}_{dataset}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                # clip_benchmark outputs metrics dict; extract accuracy
                if isinstance(data, dict):
                    acc = data.get("metrics", data).get("acc1", data.get("acc", None))
                    if acc is not None:
                        results[method][dataset] = acc
                elif isinstance(data, list) and len(data) > 0:
                    acc = data[0].get("metrics", {}).get("acc1", None)
                    if acc is not None:
                        results[method][dataset] = acc
    return results


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table."""
    # Header
    header = f"{'Dataset':<20}"
    for method in METHODS:
        header += f" {METHOD_LABELS[method]:>12}"
    header += f" {'Best Δ':>8}"
    print(header)
    print("-" * len(header))

    baseline_scores = results.get("baseline", {})

    for dataset in DATASETS:
        cat = DATASET_CATEGORIES[dataset]
        row = f"{dataset:<20}"
        deltas = []

        for method in METHODS:
            acc = results.get(method, {}).get(dataset)
            if acc is not None:
                row += f" {acc:>11.1%}"
                if method != "baseline" and dataset in baseline_scores:
                    delta = acc - baseline_scores[dataset]
                    deltas.append(delta)
            else:
                row += f" {'—':>12}"

        # Show max delta (positive = improved, negative = forgot)
        if deltas:
            worst = min(deltas)
            row += f" {worst:>+7.1%}"
        print(f"{row}  [{cat}]")

    # Summary statistics
    print()
    print("--- Forgetting Summary (Δ vs pretrained baseline) ---")
    print()

    for method in METHODS:
        if method == "baseline":
            continue

        deltas_all = []
        deltas_near = []
        deltas_far = []

        for dataset in DATASETS:
            if dataset in results.get(method, {}) and dataset in baseline_scores:
                delta = results[method][dataset] - baseline_scores[dataset]
                deltas_all.append(delta)
                cat = DATASET_CATEGORIES[dataset]
                if cat == "near":
                    deltas_near.append(delta)
                elif cat == "far":
                    deltas_far.append(delta)

        if deltas_all:
            avg_all = sum(deltas_all) / len(deltas_all)
            avg_far = sum(deltas_far) / len(deltas_far) if deltas_far else 0
            worst = min(deltas_all)
            worst_ds = DATASETS[[
                i for i, d in enumerate(DATASETS)
                if d in results.get(method, {}) and d in baseline_scores
            ][deltas_all.index(worst)]]

            print(
                f"  {METHOD_LABELS[method]:<12}: "
                f"avg={avg_all:>+.2%}  "
                f"far-domain={avg_far:>+.2%}  "
                f"worst={worst:>+.2%} ({worst_ds})"
            )


def main() -> None:
    if not RESULTS_DIR.exists():
        logger.error(f"No results directory found at {RESULTS_DIR}")
        logger.error("Run: bash scripts/run_009.sh")
        return

    results = load_results()

    total_evals = sum(len(v) for v in results.values())
    expected = len(METHODS) * len(DATASETS)
    logger.info(f"Loaded {total_evals}/{expected} evaluation results")

    if total_evals == 0:
        logger.error("No results found. Run: bash scripts/run_009.sh")
        return

    print()
    print("=" * 80)
    print("Experiment 009: Multi-Dataset CLIP Benchmark Results")
    print("=" * 80)
    print()
    print_results_table(results)


if __name__ == "__main__":
    main()
