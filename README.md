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
```

## Key findings

Five experiments, each building on the last:

| # | What changed | Loss | Train Acc | Recall@1 | Key insight |
|---|---|---|---|---|---|
| 001 | Baseline (B=256) | 4.15 | 12% | 0.006 | It learns, but slowly |
| 002 | B=512, 20K steps | 2.55 | 44% | 0.137 | LR=1e-3 destabilizes after warmup; cosine decay recovers |
| 003 | **lr=3e-4** | **0.30** | **90%** | **0.276** | LR was the single biggest lever |
| 004 | Grad accum (eff B=2048) | 0.09 | 97% | 0.200 | More memorization, not better generalization |
| 005 | SigLIP loss | 0.002 | 90% | 0.259 | Same ceiling — it's a data limit, not loss function |

**The recall ceiling (~0.26-0.28) on 1M CC3M pairs is a data/model limit**, confirmed across three approaches. Loss function, gradient quality, and effective batch size are not the bottleneck.

### Things that surprised me

- **Temperature is the best diagnostic signal.** Increasing = model gaining confidence. Decreasing = struggling. Exp 002's V-shaped temperature curve (down then up) perfectly tracked the LR destabilization and recovery.
- **Gradient accumulation hurt generalization.** It gave the optimizer smoother gradients, which it used to memorize the training set more efficiently. More data passes over the same 1M pairs = more overfitting.
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
# Reproduce the best experiment (exp 003)
uv run python -m main --batch-size 512 --total-steps 20000 --lr 3e-4

# Try SigLIP loss
uv run python -m main --batch-size 512 --total-steps 20000 --lr 3e-4 --loss siglip

# Gradient accumulation (effective batch = batch_size * accum_freq)
uv run python -m main --batch-size 512 --total-steps 20000 --lr 3e-4 --accum-freq 4
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

The from-scratch path on 1M pairs is thoroughly explored. The [lesson plan](LESSON_PLAN.md) outlines next experiments:

- **Fine-tuning pretrained CLIP** — sidesteps the data ceiling by starting from 400M+ pairs of learned features
- **LP-FT / WiSE-FT / LoRA** — controlled comparison of adaptation methods and catastrophic forgetting
- **Data-centric methods** — hard negative mining, LLM caption augmentation, DFN data filtering

## License

MIT
