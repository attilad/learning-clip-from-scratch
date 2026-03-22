# Experiment 001 — Baseline: First CLIP Training Run

## Hypothesis

A randomly initialized ViT-B/32 trained on ~1M CC3M image-text pairs with
standard hyperparameters (AdamW, lr=1e-3, cosine schedule, BF16) should:

- Show loss dropping from ~5.5 (ln(256) — random chance for batch size 256)
  to somewhere in the 2.0–3.0 range within 10K steps
- Develop non-trivial in-batch accuracy (>10% i2t and t2i, up from 0.4% random)
- Learn a temperature (logit_scale) that increases from init (~14) as the
  model gains confidence in its similarity estimates
- Produce recall@1 > random (>0.01) on the held-out eval set

This is our "does it learn anything at all?" baseline. No tuning — just
verify the full pipeline works end-to-end and produces measurable learning.

## Method

**Config** (all defaults from `TrainConfig`):
| Parameter       | Value   |
|-----------------|---------|
| Model           | ViT-B/32 (random init) |
| Batch size      | 256     |
| LR              | 1e-3    |
| Weight decay    | 0.1     |
| Warmup          | 10% of steps |
| Scheduler       | Cosine  |
| Precision       | BF16    |
| Total steps     | 10,000  |
| Eval every      | 500 steps |
| Checkpoint every| 1,000 steps |

**Data**:
- Training: ~1.07M CC3M image-text pairs
- Eval: ~9.3K CC3M validation pairs
- One epoch ≈ 4,180 steps (1.07M / 256)
- 10K steps ≈ 2.4 epochs

**Command**:
```bash
uv run --no-sync python -m main
```

## Observations

**Loss curve**: Dropped from 5.65 (step 0) to ~4.1-4.3 (step 10K). The decline
was steepest in the first 500 steps (5.65→4.7), then slowed significantly.
The loss did NOT reach the predicted 2.0-3.0 range — it plateaued around 4.2.

**In-batch accuracy**: Rose from 0% to ~10-12% by step 10K. This is 25-30x
better than random chance (0.4% at B=256), so the model IS learning meaningful
correspondences, but slowly.

**Temperature**: Went DOWN from 14.3 → 12.4, opposite to our prediction.
This means the model is *softening* its similarity distribution over time,
not sharpening it. Interpretation: the embeddings are still noisy enough
that a softer distribution produces better cross-entropy loss. A confident
model would push temperature up to be more decisive.

**Eval recall**: Improved but stayed very low:
- recall@1 went from ~0.002 → ~0.005 (i2t) and ~0.002 → ~0.003 (t2i)
- recall@5 went from ~0.008 → ~0.018 (i2t) and ~0.006 → ~0.015 (t2i)
- These are above random (1/9355 ≈ 0.0001), but far from useful.

**Throughput**: ~0.2s/step (after step 0 warmup), ~1,280 samples/sec.
Only 2.6GB VRAM used — the 4090 has massive headroom.

**Training progression by phase**:
| Phase      | Steps     | Loss  | i2t_acc | temp | Notes                    |
|------------|-----------|-------|---------|------|--------------------------|
| Start      | 0         | 5.65  | 0.0%    | 14.3 | Random init              |
| Early      | 500       | 4.65  | 7%      | 14.1 | Fastest improvement      |
| Mid        | 2000      | 4.85  | 5%      | 12.4 | Briefly got worse (epoch boundary?) |
| Late       | 5000      | 4.55  | 6%      | 11.5 | Slow steady improvement  |
| Final      | 10000     | 4.15  | 10%     | 12.4 | Still improving          |

## Results

| Metric          | Value    |
|-----------------|----------|
| Initial loss    | 5.65     |
| Final loss      | ~4.15    |
| Best i2t_acc    | ~12%     |
| Best t2i_acc    | ~12%     |
| i2t_recall@1    | 0.0058 (step 8000) |
| t2i_recall@1    | 0.0044 (step 8000) |
| i2t_recall@5    | 0.0187 (step 9000) |
| t2i_recall@5    | 0.0153 (step 8000) |
| Steps trained   | 10,000   |
| Wall time       | ~47 min  |
| Peak GPU mem    | 2.6 GB   |
| Throughput      | ~0.2s/step, ~1,280 samples/sec |

## Conclusions

**What we learned:**
1. The pipeline works end-to-end — model trains, eval runs, checkpoints save.
2. The model IS learning (10% accuracy vs 0.4% random), but progress is slow.
3. Loss plateaued around 4.2 instead of the predicted 2.0-3.0. This suggests
   10K steps on 1M pairs is not enough, OR the learning rate / batch size
   needs tuning.

**Hypothesis scorecard:**
- ✅ Loss drops from ~5.5 — confirmed (5.65 → 4.15)
- ✅ In-batch accuracy > 10% — barely met at the very end
- ❌ Temperature increases — it DECREASED (14.3 → 12.4)
- ❌ Loss reaches 2.0-3.0 — plateaued at 4.2
- ⚠️ recall@1 > 0.01 — achieved 0.006, borderline

**What to try next (experiment 002 candidates):**
- **More steps**: The loss curve was still decreasing at step 10K. We may
  just need more training time (20-50K steps).
- **Larger batch size**: Contrastive learning benefits strongly from more
  negatives per batch. B=256 gives 255 negatives; B=1024 gives 1023.
  We're only using 2.6GB of 24GB — we have headroom.
- **Gradient accumulation**: If batch size is the bottleneck, we could
  accumulate gradients over 4 steps to simulate B=1024 without needing
  more VRAM (though we have it).
- **Lower learning rate**: 1e-3 may be too aggressive, causing oscillation
  rather than convergence.
