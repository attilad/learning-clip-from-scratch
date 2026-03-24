"""Main CLIP training loop.

This is the heart of the project. The training loop implements the procedure
from Radford et al. (2021) "Learning Transferable Visual Models From Natural
Language Supervision" — simplified for readability but mechanically correct.

Key concepts to understand:
  - Contrastive learning: we don't predict labels, we learn a distance metric
  - Mixed precision: BF16 halves memory and doubles throughput on Ada GPUs
  - Gradient scaling: necessary because BF16 has limited dynamic range
  - Warmup: prevents early gradient explosions when params are random
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.loss import CLIPLoss, SigLIPLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters — all defaults from CLAUDE.md.

    These are reasonable starting points for ViT-B/32 on CC3M. The values
    come from a mix of the original CLIP paper, OpenCLIP's defaults, and
    what fits in 24GB VRAM.
    """

    batch_size: int = 256
    # lr=1e-3 is aggressive but works with warmup. The original CLIP used
    # lr=5e-4 on 400M pairs; we use higher lr because our dataset is smaller
    # and we need to learn faster before overfitting.
    lr: float = 1e-3
    # weight_decay=0.1 is standard for ViTs. It acts as L2 regularization
    # but decoupled from the gradient (AdamW vs Adam+L2 matters here).
    weight_decay: float = 0.1
    # Warmup the first 10% of training. Random init → huge gradients on
    # step 1. Warmup lets the optimizer build up good second-moment estimates
    # (Adam's running variance) before taking full-size steps.
    warmup_fraction: float = 0.1
    total_steps: int = 10_000
    # Loss function: "clip" (InfoNCE) or "siglip" (sigmoid pairwise).
    # SigLIP outperforms InfoNCE at batch sizes below ~16K.
    loss_type: str = "clip"
    # Gradient accumulation: simulate larger effective batch sizes without
    # more VRAM. With accum_freq=4 and batch_size=512, the effective batch
    # is 2048 — but the contrastive loss still sees 512 per forward pass.
    # This gives better gradient estimates, not more negatives per loss call.
    accum_freq: int = 1
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"


def build_optimizer_and_scheduler(
    model: nn.Module,
    loss_fn: CLIPLoss | SigLIPLoss,
    config: TrainConfig,
) -> tuple[AdamW, SequentialLR]:
    """Set up AdamW with cosine schedule + linear warmup.

    Why separate param groups for the temperature?
      - weight_decay=0 for logit_scale: it's a scalar controlling the
        softmax sharpness, not a weight matrix. Regularizing it toward
        zero would force the temperature toward infinity (flat softmax),
        destroying the contrastive signal.
      - Same lr for now, but having it in a separate group lets us
        experiment with different lr for temperature later.
    """
    # Only optimize params that require gradients — freeze_backbone() and
    # apply_lora() set requires_grad=False on frozen params, and passing
    # frozen params to AdamW wastes memory on unused optimizer states.
    model_params = [p for p in model.parameters() if p.requires_grad]
    loss_params = [p for p in loss_fn.parameters()]

    trainable_count = sum(p.numel() for p in model_params)
    logger.info(f"Optimizer: {trainable_count:,} trainable model params")

    param_groups = [
        {
            "params": model_params,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
        },
        {
            "params": loss_params,
            "lr": config.lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups)

    warmup_steps = int(config.total_steps * config.warmup_fraction)

    # LinearLR: lr ramps from lr*start_factor to lr over warmup_steps.
    # start_factor=0.01 means we begin at 1% of target lr.
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)

    # CosineAnnealingLR: lr decays following a cosine curve to ~0.
    # This gives the model most of its learning budget in the middle
    # of training, with a gentle ramp-down at the end.
    cosine = CosineAnnealingLR(optimizer, T_max=config.total_steps - warmup_steps)

    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler


def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizer: AdamW,
    loss_fn: CLIPLoss,
    config: TrainConfig,
    checkpoint_dir: Path,
) -> Path:
    """Save model, optimizer, loss state, and config snapshot.

    We save the full optimizer state (not just model weights) so we can
    resume training with the same Adam momentum estimates. Without this,
    resuming from a checkpoint would effectively reset the optimizer,
    causing a spike in loss as it re-estimates second moments.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "logit_scale": loss_fn.logit_scale.item(),
            "config": asdict(config),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: AdamW | None = None,
    loss_fn: CLIPLoss | None = None,
) -> int:
    """Load a checkpoint and return the step number."""
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None:
        # Skip optimizer state if param count changed (e.g., LP-FT phase 2
        # resumes from a frozen-backbone checkpoint with a new optimizer).
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            logger.warning(
                "Optimizer state mismatch — rebuilding optimizer from scratch. "
                "This is expected when changing trainable params between phases (LP-FT)."
            )
    if loss_fn is not None:
        loss_fn.logit_scale.data.fill_(
            torch.tensor(ckpt["logit_scale"]).log()
        )
    logger.info(f"Resumed from checkpoint {path} at step {ckpt['step']}")
    return ckpt["step"]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainConfig,
    eval_fn: Callable[[nn.Module, int, SummaryWriter], None] | None = None,
    resume_from: str | Path | None = None,
) -> None:
    """Run the CLIP training loop.

    The loop structure is step-based (not epoch-based) because:
      1. CC3M is large enough that we may not finish a full epoch
      2. Checkpointing/eval on step boundaries is more predictable
      3. The LR schedule is defined in steps, not epochs

    Args:
        model: CLIP model (open_clip), will be moved to GPU.
        train_loader: Yields (image_batch, text_batch) tensors.
        config: Training hyperparameters.
        eval_fn: Optional callable(model, step, writer) for periodic eval.
        resume_from: Path to checkpoint to resume from.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — no silent CPU fallback.")

    device = torch.device("cuda")
    model = model.to(device)

    if config.loss_type == "siglip":
        loss_fn = SigLIPLoss().to(device)
        logger.info("Using SigLIP sigmoid loss")
    else:
        loss_fn = CLIPLoss().to(device)
        logger.info("Using CLIP InfoNCE loss")
    optimizer, scheduler = build_optimizer_and_scheduler(model, loss_fn, config)

    # GradScaler adjusts loss magnitude before backward() to prevent
    # BF16 underflow. BF16 has the same exponent range as FP32 (unlike FP16),
    # so scaling is less critical — but it's cheap insurance and PyTorch
    # handles it automatically.
    scaler = GradScaler()
    writer = SummaryWriter(config.log_dir)

    # Save config to the log dir for reproducibility
    config_path = Path(config.log_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2))

    checkpoint_dir = Path(config.checkpoint_dir)

    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from, model, optimizer, loss_fn)

    step = start_step
    model.train()

    accum_freq = config.accum_freq
    effective_batch = config.batch_size * accum_freq

    logger.info(
        f"Starting training: {config.total_steps} steps, "
        f"batch_size={config.batch_size}, accum_freq={accum_freq}, "
        f"effective_batch={effective_batch}, lr={config.lr}"
    )

    # Outer loop re-iterates the DataLoader for multi-epoch training.
    while step < config.total_steps:
        for images, texts in train_loader:
            if step >= config.total_steps:
                break

            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            # --- Gradient Accumulation ---
            # With accum_freq > 1, we accumulate gradients over multiple
            # micro-batches before updating weights. This simulates a
            # larger effective batch for better gradient estimates.
            #
            # Important subtlety: the contrastive loss still computes its
            # N×N similarity matrix within each micro-batch. So accum_freq=4
            # with B=512 gives gradients averaged over 4 batches of 512,
            # NOT one batch of 2048. The negatives-per-sample stay at 511.
            # What improves is the gradient estimate quality — each optimizer
            # step is informed by 4x more data.

            is_accum_step = (step % accum_freq != 0) if accum_freq > 1 else False
            is_update_step = ((step + 1) % accum_freq == 0) or (step + 1 >= config.total_steps)

            if not is_accum_step:
                t0 = time.perf_counter()
                optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                loss, metrics = loss_fn(image_features, text_features)

            # Scale loss by accum_freq so the accumulated gradient has the
            # same magnitude as a single-batch gradient. Without this,
            # accum_freq=4 would have 4x larger gradients than accum_freq=1.
            scaled_loss = loss / accum_freq
            scaler.scale(scaled_loss).backward()

            if is_update_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                step_time = time.perf_counter() - t0

                # Count optimizer steps (not micro-batch steps) for logging
                opt_step = step // accum_freq

                # --- Logging ---
                if opt_step % config.log_every == 0:
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    grad_norm = _compute_grad_norm(model)

                    print(
                        f"step {opt_step:>6d} | loss {loss.item():.4f} | "
                        f"i2t_acc {metrics['i2t_acc']:.3f} | "
                        f"t2i_acc {metrics['t2i_acc']:.3f} | "
                        f"temp {metrics['logit_scale']:.1f} | "
                        f"gpu {gpu_mem:.1f}GB | {step_time:.3f}s"
                    )

                    writer.add_scalar("train/loss", loss.item(), opt_step)
                    writer.add_scalar("train/i2t_accuracy", metrics["i2t_acc"], opt_step)
                    writer.add_scalar("train/t2i_accuracy", metrics["t2i_acc"], opt_step)
                    writer.add_scalar("train/logit_scale", metrics["logit_scale"], opt_step)
                    writer.add_scalar("train/temperature", metrics["temperature"], opt_step)
                    writer.add_scalar("train/grad_norm", grad_norm, opt_step)
                    writer.add_scalar("train/gpu_memory_gb", gpu_mem, opt_step)
                    writer.add_scalar("train/step_time", step_time, opt_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], opt_step)
                    if "logit_bias" in metrics:
                        writer.add_scalar("train/logit_bias", metrics["logit_bias"], opt_step)

                # --- Eval ---
                if eval_fn is not None and opt_step % config.eval_every == 0 and opt_step > 0:
                    model.eval()
                    eval_fn(model, opt_step, writer)
                    model.train()

                # --- Checkpoint ---
                if opt_step % config.checkpoint_every == 0 and opt_step > 0:
                    path = save_checkpoint(
                        opt_step, model, optimizer, loss_fn, config, checkpoint_dir
                    )
                    logger.info(f"Checkpoint saved: {path}")

            step += 1

    # Final checkpoint
    final_opt_step = step // accum_freq
    path = save_checkpoint(final_opt_step, model, optimizer, loss_fn, config, checkpoint_dir)
    writer.close()
    logger.info(f"Training complete at step {final_opt_step}. Final checkpoint: {path}")


def _compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters.

    This is a key diagnostic: if grad_norm spikes, training is unstable.
    If it goes to zero, the model has stopped learning (vanishing gradients).
    Healthy training has grad_norm that starts high, drops during warmup,
    then stays roughly stable.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
