# Learning CLIP from Scratch

Training OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (ViT-B/32, 151M params) from scratch on a single RTX 4090, then systematically exploring what makes contrastive vision-language training work.

This isn't a reproduction of the original paper — it's a learning project. Each experiment isolates one variable, documents a hypothesis, and records what actually happened. The goal is building intuition for training dynamics, not achieving SOTA.

## What's here

```
src/                    Training code
  model.py              open_clip ViT-B/32 wrapper
  loss.py               InfoNCE (CLIPLoss) + SigLIP (SigLIPLoss)
  dataset.py            CC3M dataset with content-hashed image storage
  train.py              Training loop: BF16, gradient accumulation, cosine schedule
  eval.py               Recall@K evaluation
  adapt.py              Adaptation: freeze backbone, LoRA, WiSE-FT interpolation
  zero_shot_classify.py CIFAR-100 zero-shot classification eval

scripts/                Utilities
  download_cc3m.py      Async image downloader (fault-tolerant, resumable)
  smoke_test.py         GPU + model + forward pass sanity check
  sanity_check.py       Visual data pipeline verification
  demo.py               Zero-shot classification, retrieval, similarity heatmaps
  semantic_umap.py      Embedding space visualization colored by category
  postmortem.py         Training forensics: parse TensorBoard, analyze gradients

notebooks/
  explore.ipynb         Interactive model exploration (Jupyter)

experiments/            Structured experiment documentation
  001_baseline/         B=256, lr=1e-3, 10K steps
  002_large_batch/      B=512, lr=1e-3, 20K steps
  003_lower_lr/         B=512, lr=3e-4, 20K steps      <-- best from-scratch result
  004_grad_accum/       Effective B=2048 via accumulation
  005_siglip/           SigLIP sigmoid loss
  007_finetune/         Pretrained fine-tuning at 3 learning rates
  008_adaptation/       LoRA, WiSE-FT, frozen backbone, LP-FT comparison
```

## Key findings

### Phase 1: From-scratch training (exp 001–005)

| # | What changed | Loss | Train Acc | Recall@1 | Key insight |
|---|---|---|---|---|---|
| 001 | Baseline (B=256) | 4.15 | 12% | 0.006 | It learns, but slowly |
| 002 | B=512, 20K steps | 2.55 | 44% | 0.137 | LR=1e-3 destabilizes after warmup; cosine decay recovers |
| 003 | **lr=3e-4** | **0.30** | **90%** | **0.276** | LR was the single biggest lever |
| 004 | Grad accum (eff B=2048) | 0.09 | 97% | 0.200 | More memorization, not better generalization |
| 005 | SigLIP loss | 0.002 | 90% | 0.259 | Same ceiling — it's a data limit, not loss function |

**The recall ceiling (~0.26-0.28) on 1M CC3M pairs is a data/model limit**, confirmed across three approaches.

### Phase 2: Fine-tuning pretrained CLIP (exp 007–008)

Starting from OpenAI's pretrained ViT-B/32 (trained on 400M pairs), then adapting to CC3M:

| # | Method | CC3M R@1 | CIFAR-100 ZS | Trainable params | Key insight |
|---|---|---|---|---|---|
| 007 | Full fine-tune (lr=1e-5) | **0.760** | 0.630 | 151M (100%) | 2.8× better than from-scratch |
| 008 | Frozen backbone | 0.626 | 0.596 | 655K (0.4%) | Counterproductive — hurts OOD |
| 008 | LoRA rank=4 | 0.729 | 0.636 | 901K (0.6%) | 96% of full FT's gain, preserves OOD |
| 008 | **WiSE-FT (α=0.5)** | 0.734 | **0.663** | 151M | Best OOD — beats even pretrained |

CIFAR-100 zero-shot accuracy measures whether adaptation destroys general visual knowledge (pretrained baseline: 62.3%).

### Things that surprised me

- **Temperature is the best diagnostic signal.** Increasing = model gaining confidence. Decreasing = struggling. Exp 002's V-shaped temperature curve (down then up) perfectly tracked the LR destabilization and recovery. In exp 008, LoRA's temperature spiked to 17.3 (vs 14.6 for full FT) — a signal of adapter capacity saturation.
- **Gradient accumulation hurt generalization.** It gave the optimizer smoother gradients, which it used to memorize the training set more efficiently. More data passes over the same 1M pairs = more overfitting.
- **No catastrophic forgetting on CIFAR-100.** Full fine-tuning on CC3M didn't destroy CIFAR-100 zero-shot accuracy — it actually improved slightly. CC3M's visual concepts overlap enough with CIFAR-100 that adaptation is complementary. Measuring forgetting requires a more distant OOD benchmark.
- **WiSE-FT creates performance that neither model had.** Interpolating fine-tuned and pretrained weights (50/50) yielded the best CIFAR-100 score across all configurations — better than the pretrained model it was interpolated with.
- **Freezing the backbone is the worst strategy.** Projection-only training can't compensate for a frozen encoder. It was the only method to hurt both CC3M and CIFAR-100.
- **Semantic structure emerges naturally.** Despite training on noisy web-scraped captions, the model organized its embeddings into meaningful clusters (animals, buildings, vehicles, food) — verified quantitatively with intra/inter-class similarity.

## Setup

Requires an NVIDIA GPU with CUDA. Developed on RTX 4090 (24GB) with WSL2 Ubuntu 24.04.

```bash
# Install dependencies
uv sync

# Verify GPU
uv run python -m scripts.smoke_test
```

### Get training data

```bash
# Download CC3M TSV from HuggingFace
HF_HOME=.cache/huggingface uv run python -c "
from datasets import load_dataset
import csv
ds = load_dataset('google-research-datasets/conceptual_captions', 'unlabeled', split='train')
with open('data/cc3m/Train_GCC-training.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for row in ds:
        writer.writerow([row['caption'], row['image_url']])
"

# Download images (expect ~50% failure rate — many CC3M URLs are dead)
uv run python -m scripts.download_cc3m \
    --tsv data/cc3m/Train_GCC-training.tsv \
    --output-dir data/cc3m/images \
    --max-concurrent 64
```

### Train

```bash
# From scratch (exp 003)
uv run python -m main --batch-size 512 --total-steps 20000 --lr 3e-4

# Fine-tune pretrained (exp 007)
uv run python -m main --pretrained openai --batch-size 512 --total-steps 5000 --lr 1e-5

# LoRA adaptation (exp 008)
uv run python -m main --pretrained openai --batch-size 512 --total-steps 5000 --lr 1e-4 --lora-rank 4

# WiSE-FT: fine-tune then interpolate with pretrained (exp 008)
uv run python -m main --pretrained openai --batch-size 512 --total-steps 5000 --lr 1e-5 --wise-ft-alpha 0.5

# Add CIFAR-100 zero-shot eval to any run
uv run python -m main --pretrained openai --batch-size 512 --total-steps 5000 --lr 1e-5 --cifar100-eval
```

### Explore the model

```bash
# Visual demo: zero-shot classification, retrieval, similarity heatmaps
uv run python -m scripts.demo --checkpoint checkpoints/step_020000.pt

# Semantic UMAP: does the model organize the world into clusters?
uv run python -m scripts.semantic_umap --checkpoint checkpoints/step_020000.pt

# Interactive notebook
uv run jupyter notebook notebooks/explore.ipynb

# Training forensics
uv run python -m scripts.postmortem \
    --log-dir runs/ --checkpoint-dir checkpoints/
```

## What's next

From-scratch training and pretrained adaptation are thoroughly explored. The [lesson plan](LESSON_PLAN.md) outlines next experiments:

- **Layer-wise LR decay (LLRD)** — assign lower LR to earlier transformer layers to protect generic features
- **Data-centric methods** — hard negative mining, LLM caption augmentation, DFN data filtering
- **Distillation** — compress a larger CLIP teacher into a smaller student

## License

MIT
