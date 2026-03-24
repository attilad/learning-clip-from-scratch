#!/bin/bash
# Experiment 008: Adaptation methods & catastrophic forgetting
#
# Compares 5 adaptation strategies on two axes:
#   - CC3M retrieval (in-distribution)
#   - CIFAR-100 zero-shot classification (out-of-distribution / forgetting)
#
# 4 training runs + 2 eval-only passes ≈ 3 hours total

set -e
export HF_HOME=.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_DIR=experiments/008_adaptation
mkdir -p "$EXP_DIR"

COMMON_FLAGS="--pretrained openai --batch-size 512 --total-steps 5000 \
    --eval-every 500 --checkpoint-every 5000 --cifar100-eval"

echo "=========================================="
echo "Exp 008: Adaptation Methods & Forgetting"
echo "=========================================="

# --- Baseline: pretrained zero-shot (no training) ---
echo ""
echo "--- Baseline: pretrained zero-shot ---"
uv run --no-sync python -c "
import torch, logging
from src.model import create_model
from src.dataset import CC3MDataset, create_dataloader
from src.eval import compute_recall_at_k
from src.zero_shot_classify import cifar100_zero_shot

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda')
model, preprocess, tokenizer = create_model('ViT-B-32', pretrained='openai', device=device)
model.eval()

# CC3M recall
dataset = CC3MDataset(
    tsv_path='data/cc3m/Validation_GCC-1.1.0-Validation.tsv',
    image_dir='data/eval/images',
    transform=preprocess,
    tokenizer=tokenizer,
)
subset = torch.utils.data.Subset(dataset, list(range(1000)))
loader = create_dataloader(subset, batch_size=256, num_workers=4, shuffle=False)

print('=== PRETRAINED BASELINE ===')
results = compute_recall_at_k(model, loader)
for k, v in results.items():
    print(f'  {k} = {v:.4f}')

# CIFAR-100 zero-shot
zs = cifar100_zero_shot(model, tokenizer, preprocess)
for k, v in zs.items():
    print(f'  {k} = {v:.4f}')
" 2>&1 | tee "$EXP_DIR/baseline.log"

# --- Run A: Full fine-tune (reproduce exp 007 best) ---
echo ""
echo "--- Run A: Full fine-tune (lr=1e-5) ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    $COMMON_FLAGS --lr 1e-5 \
    2>&1 | tee "$EXP_DIR/run_a_full_ft.log"

cp -r runs/ "$EXP_DIR/runs_a"
cp -r checkpoints/ "$EXP_DIR/checkpoints_a"

# --- Run B: Freeze backbone (projection layers only) ---
echo ""
echo "--- Run B: Freeze backbone (lr=1e-5) ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    $COMMON_FLAGS --lr 1e-5 --freeze-backbone \
    2>&1 | tee "$EXP_DIR/run_b_freeze.log"

cp -r runs/ "$EXP_DIR/runs_b"
cp -r checkpoints/ "$EXP_DIR/checkpoints_b"

# --- Run C: LP-FT (two-phase) ---
# Phase 1: Freeze backbone for 2500 steps
# Phase 2: Unfreeze all for 2500 more steps (resume from phase 1)
echo ""
echo "--- Run C: LP-FT Phase 1 (frozen, 2500 steps) ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    $COMMON_FLAGS --lr 1e-5 --freeze-backbone --total-steps 2500 \
    2>&1 | tee "$EXP_DIR/run_c_lpft_phase1.log"

echo ""
echo "--- Run C: LP-FT Phase 2 (unfrozen, 2500 more steps) ---"
# Find the last checkpoint from phase 1
PHASE1_CKPT=$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1)
if [ -z "$PHASE1_CKPT" ]; then
    echo "ERROR: No phase 1 checkpoint found"
    exit 1
fi
echo "Resuming from: $PHASE1_CKPT"

# Phase 2: unfreeze all params, continue training
# Note: --resume loads weights, and without --freeze-backbone all params train
uv run --no-sync python -m main \
    $COMMON_FLAGS --lr 1e-5 --total-steps 2500 --resume "$PHASE1_CKPT" \
    2>&1 | tee "$EXP_DIR/run_c_lpft_phase2.log"

cp -r runs/ "$EXP_DIR/runs_c"
cp -r checkpoints/ "$EXP_DIR/checkpoints_c"

# --- Run D: LoRA rank=4 ---
echo ""
echo "--- Run D: LoRA rank=4 (lr=1e-4) ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    $COMMON_FLAGS --lr 1e-4 --lora-rank 4 \
    2>&1 | tee "$EXP_DIR/run_d_lora.log"

cp -r runs/ "$EXP_DIR/runs_d"
cp -r checkpoints/ "$EXP_DIR/checkpoints_d"

# --- Run E: WiSE-FT (full FT + weight interpolation) ---
# This re-runs full FT with --wise-ft-alpha, which:
# 1. Snapshots pretrained weights
# 2. Trains normally
# 3. Interpolates final weights: 0.5*FT + 0.5*pretrained
echo ""
echo "--- Run E: WiSE-FT (alpha=0.5) ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    $COMMON_FLAGS --lr 1e-5 --wise-ft-alpha 0.5 \
    2>&1 | tee "$EXP_DIR/run_e_wiseft.log"

cp -r runs/ "$EXP_DIR/runs_e"
cp -r checkpoints/ "$EXP_DIR/checkpoints_e"

echo ""
echo "=========================================="
echo "Exp 008 complete! Check $EXP_DIR/"
echo "=========================================="
