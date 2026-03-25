#!/bin/bash
# Experiment 009: Multi-dataset evaluation via CLIP Benchmark
#
# Evaluates all exp 008 checkpoints across 9 zero-shot classification datasets
# to properly measure catastrophic forgetting beyond CIFAR-100.
#
# Prerequisites:
#   uv run python -m scripts.export_checkpoints  (converts checkpoints)
#
# Runtime: ~30-45 min total (inference only, no training)

set -e
export HF_HOME=.cache/huggingface

EXPORT_DIR=experiments/009_benchmark/exported

echo "=========================================="
echo "Exp 009: Multi-Dataset CLIP Benchmark"
echo "=========================================="

# Check that exported checkpoints exist
if [ ! -d "$EXPORT_DIR" ] || [ -z "$(ls $EXPORT_DIR/*.pt 2>/dev/null)" ]; then
    echo "No exported checkpoints found. Running export..."
    uv run --no-sync python -m scripts.export_checkpoints
fi

# --- Evaluate each model on all datasets ---

echo ""
echo "--- Baseline (pretrained openai) ---"
uv run --no-sync python -m scripts.eval_benchmark \
    --pretrained openai --dataset all --name baseline

echo ""
echo "--- Full FT ---"
uv run --no-sync python -m scripts.eval_benchmark \
    --pretrained "$EXPORT_DIR/full_ft.pt" --dataset all --name full_ft

echo ""
echo "--- Freeze backbone ---"
uv run --no-sync python -m scripts.eval_benchmark \
    --pretrained "$EXPORT_DIR/freeze.pt" --dataset all --name freeze

echo ""
echo "--- LoRA r=4 ---"
uv run --no-sync python -m scripts.eval_benchmark \
    --pretrained "$EXPORT_DIR/lora_r4.pt" --dataset all --name lora_r4

echo ""
echo "--- WiSE-FT ---"
uv run --no-sync python -m scripts.eval_benchmark \
    --pretrained "$EXPORT_DIR/wise_ft.pt" --dataset all --name wise_ft

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo ""
echo "Run analysis: uv run python -m scripts.analyze_009"
echo "=========================================="
