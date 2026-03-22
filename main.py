"""Entry point: wire up dataset → model → training loop.

Usage:
    # Full training run
    uv run python -m main

    # Quick test (100 steps, small batch)
    uv run python -m main --total-steps 100 --batch-size 32

    # Resume from checkpoint
    uv run python -m main --resume checkpoints/step_001000.pt
"""

import argparse
import logging

import torch
from src.dataset import CC3MDataset, create_dataloader
from src.eval import compute_recall_at_k, log_eval_results
from src.model import create_model
from src.train import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP on CC3M")
    parser.add_argument("--tsv", type=str, default="data/cc3m/Train_GCC-training.tsv")
    parser.add_argument("--image-dir", type=str, default="data/cc3m/images")
    parser.add_argument("--eval-tsv", type=str, default="data/cc3m/Validation_GCC-1.1.0-Validation.tsv")
    parser.add_argument("--eval-image-dir", type=str, default="data/eval/images")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--accum-freq", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum_freq)")
    parser.add_argument("--loss", type=str, default="clip", choices=["clip", "siglip"],
                        help="Loss function: 'clip' (InfoNCE) or 'siglip' (sigmoid pairwise)")
    parser.add_argument("--eval-samples", type=int, default=1000,
                        help="Max eval samples (caps similarity matrix size)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # --- Model ---
    model, preprocess, tokenizer = create_model(model_name="ViT-B-32")

    # --- Training data ---
    train_dataset = CC3MDataset(
        tsv_path=args.tsv,
        image_dir=args.image_dir,
        transform=preprocess,
        tokenizer=tokenizer,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Eval data ---
    # Eval encodes all samples then builds an [N, N] similarity matrix,
    # so we cap N to avoid OOM. 1000 samples → 1000x1000 matrix ≈ 4MB.
    eval_loader = None
    try:
        eval_dataset = CC3MDataset(
            tsv_path=args.eval_tsv,
            image_dir=args.eval_image_dir,
            transform=preprocess,
            tokenizer=tokenizer,
        )
        if len(eval_dataset) > args.eval_samples:
            eval_dataset = torch.utils.data.Subset(
                eval_dataset,
                indices=list(range(args.eval_samples)),
            )
        eval_loader = create_dataloader(
            eval_dataset,
            batch_size=256,  # Fixed small batch for eval encoding
            num_workers=4,
            shuffle=False,
        )
        logging.info(f"Eval set: {len(eval_dataset)} samples")
    except (FileNotFoundError, RuntimeError) as e:
        logging.warning(f"No eval set available ({e}). Training without eval.")

    # --- Eval callback ---
    def eval_fn(model, step, writer):
        if eval_loader is None:
            return
        results = compute_recall_at_k(model, eval_loader)
        log_eval_results(results, step, writer)

    # --- Train ---
    # total_steps counts micro-batches, not optimizer steps.
    # With accum_freq=4 and total_steps=20000, there are 20000 micro-batches
    # = 5000 optimizer steps. This keeps wall-clock time roughly constant
    # across accum_freq values for fair comparison.
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        total_steps=args.total_steps * args.accum_freq,
        accum_freq=args.accum_freq,
        loss_type=args.loss,
    )

    train(
        model=model,
        train_loader=train_loader,
        config=config,
        eval_fn=eval_fn,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
