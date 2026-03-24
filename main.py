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
from src.adapt import apply_lora, freeze_backbone, save_pretrained_state, wise_ft_interpolate
from src.dataset import CC3MDataset, create_dataloader
from src.eval import compute_recall_at_k, log_eval_results
from src.model import create_model
from src.train import TrainConfig, train
from src.zero_shot_classify import cifar100_zero_shot

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
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained weights tag (e.g. 'openai', 'laion2b_s34b_b79k')")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze encoders, only train projection layers")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank (0=disabled). Injects low-rank adapters into attention.")
    parser.add_argument("--cifar100-eval", action="store_true",
                        help="Run CIFAR-100 zero-shot classification at each eval step")
    parser.add_argument("--wise-ft-alpha", type=float, default=None,
                        help="WiSE-FT: interpolate final weights with pretrained (0-1)")
    args = parser.parse_args()

    # --- Model ---
    model, preprocess, tokenizer = create_model(
        model_name="ViT-B-32",
        pretrained=args.pretrained,
    )

    # --- Adaptation setup ---
    # Snapshot pretrained weights before any modification (needed for WiSE-FT)
    pretrained_state = None
    if args.wise_ft_alpha is not None:
        pretrained_state = save_pretrained_state(model)

    if args.lora_rank > 0:
        apply_lora(model, rank=args.lora_rank)
    elif args.freeze_backbone:
        freeze_backbone(model)

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
        if eval_loader is not None:
            results = compute_recall_at_k(model, eval_loader)
            log_eval_results(results, step, writer)

        if args.cifar100_eval:
            zs_results = cifar100_zero_shot(model, tokenizer, preprocess)
            for key, val in zs_results.items():
                print(f"  [zero-shot] {key}={val:.4f}")
                if writer is not None:
                    writer.add_scalar(f"eval/{key}", val, step)

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
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )

    train(
        model=model,
        train_loader=train_loader,
        config=config,
        eval_fn=eval_fn,
        resume_from=args.resume,
    )

    # --- WiSE-FT post-processing ---
    if args.wise_ft_alpha is not None and pretrained_state is not None:
        logging.info(f"Applying WiSE-FT interpolation (alpha={args.wise_ft_alpha})")
        wise_ft_interpolate(model, pretrained_state, alpha=args.wise_ft_alpha)

        # Re-run eval on the interpolated model
        model.eval()
        print("\n=== WiSE-FT Evaluation ===")
        if eval_loader is not None:
            results = compute_recall_at_k(model, eval_loader)
            log_eval_results(results, step=-1, writer=None)
        if args.cifar100_eval:
            zs_results = cifar100_zero_shot(model, tokenizer, preprocess)
            for key, val in zs_results.items():
                print(f"  [zero-shot] {key}={val:.4f}")

        # Save interpolated weights
        from pathlib import Path
        wise_path = Path(config.checkpoint_dir) / "wise_ft.pt"
        wise_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, wise_path)
        logging.info(f"WiSE-FT checkpoint saved: {wise_path}")


if __name__ == "__main__":
    main()
