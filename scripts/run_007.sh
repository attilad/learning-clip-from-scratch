#!/bin/bash
# Experiment 007: Fine-tune pretrained CLIP at three learning rates
# Run overnight — total ~2 hours (3 x ~40 min each)

set -e
export HF_HOME=.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Exp 007: Fine-tuning pretrained CLIP"
echo "=========================================="

# --- Zero-shot baseline (no fine-tuning) ---
echo ""
echo "--- Zero-shot baseline (pretrained, no FT) ---"
uv run --no-sync python -c "
import torch, logging
from src.model import create_model
from src.dataset import CC3MDataset, create_dataloader
from src.eval import compute_recall_at_k

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda')
model, preprocess, tokenizer = create_model('ViT-B-32', pretrained='openai', device=device)
model.eval()

dataset = CC3MDataset(
    tsv_path='data/cc3m/Validation_GCC-1.1.0-Validation.tsv',
    image_dir='data/eval/images',
    transform=preprocess,
    tokenizer=tokenizer,
)
subset = torch.utils.data.Subset(dataset, list(range(1000)))
loader = create_dataloader(subset, batch_size=256, num_workers=4, shuffle=False)

results = compute_recall_at_k(model, loader)
print('ZERO-SHOT BASELINE:')
for k, v in results.items():
    print(f'  {k} = {v:.4f}')
"

# --- Run A: lr=1e-6 ---
echo ""
echo "--- Run A: lr=1e-6 ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    --pretrained openai \
    --batch-size 512 --total-steps 5000 --lr 1e-6 --eval-every 250 --checkpoint-every 1000 \
    2>&1 | tee experiments/007_finetune/run_a.log

cp -r runs/ experiments/007_finetune/runs_a
cp -r checkpoints/ experiments/007_finetune/checkpoints_a

# --- Run B: lr=5e-6 ---
echo ""
echo "--- Run B: lr=5e-6 ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    --pretrained openai \
    --batch-size 512 --total-steps 5000 --lr 5e-6 --eval-every 250 --checkpoint-every 1000 \
    2>&1 | tee experiments/007_finetune/run_b.log

cp -r runs/ experiments/007_finetune/runs_b
cp -r checkpoints/ experiments/007_finetune/checkpoints_b

# --- Run C: lr=1e-5 ---
echo ""
echo "--- Run C: lr=1e-5 ---"
rm -rf runs/ checkpoints/
uv run --no-sync python -m main \
    --pretrained openai \
    --batch-size 512 --total-steps 5000 --lr 1e-5 --eval-every 250 --checkpoint-every 1000 \
    2>&1 | tee experiments/007_finetune/run_c.log

cp -r runs/ experiments/007_finetune/runs_c
cp -r checkpoints/ experiments/007_finetune/checkpoints_c

echo ""
echo "=========================================="
echo "Exp 007 complete! Check experiments/007_finetune/"
echo "=========================================="
