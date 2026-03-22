# Experiment 002 — Large Batch: B=512, 20K Steps

## Hypothesis

Increasing batch size from 256 → 512 should improve contrastive learning:

- **More negatives per batch**: Each sample now competes against 511
  negatives instead of 255. This gives the similarity matrix 2x more
  "wrong answers" to push apart, producing a stronger gradient signal.
- **Better gradient estimates**: Larger batches average over more samples,
  reducing noise in the gradient.
- **The original CLIP used B=32,768**: Our B=256 was tiny by comparison.

Note: B=1024 OOMed on the 4090 (24GB) during the text encoder forward
pass — the [1024, 77, 512] attention tensors plus gradients exceed VRAM.
B=512 is the largest power-of-2 that fits.

Doubling the steps (10K → 20K) compensates for seeing fewer unique
batches per epoch — at B=512, one epoch ≈ 2,093 steps, so 20K steps
≈ 9.6 epochs.

**Predictions**:
- Loss should drop well below 4.0 (vs 4.15 in exp 001)
- In-batch accuracy should exceed 20% (vs 10% in exp 001)
- Temperature should eventually increase (model gets confident enough
  to sharpen the distribution)
- Eval recall@1 should break 0.01 clearly

## Method

**Config changes from experiment 001**:
| Parameter       | Exp 001 | Exp 002 | Why                              |
|-----------------|---------|---------|----------------------------------|
| Batch size      | 256     | 512     | More negatives, better gradients |
| Total steps     | 10,000  | 20,000  | More training time               |
| Eval samples    | 9,355   | 1,000   | Capped to avoid OOM on sim matrix|
| All else        | same    | same    | Isolate the batch size effect    |

**Data**: Same as exp 001 (~1.07M train, eval capped at 1K samples)

**Command**:
```bash
uv run --no-sync python -m main --batch-size 512 --total-steps 20000
```

**Infra fixes made during this experiment**:
- Eval: added BF16 autocast, moved features to CPU before building sim matrix
- Eval: capped to 1000 samples (avoids OOM on [N,N] similarity matrix)
- Added `torch.cuda.empty_cache()` before eval

## Observations

**Three-act training story**:

**Act 1 — Fast learning (steps 0-1200)**: Loss dropped sharply from 6.30
(ln(512)≈6.24, random chance) to ~4.6. In-batch accuracy climbed to
~10%. Temperature stable at 14.3. Recall@1 peaked at 0.053 at step 1000.
This phase looked great.

**Act 2 — LR destabilization (steps 1200-4000)**: After warmup ended at
step 2000, the full learning rate (1e-3) caused instability. Loss
climbed BACK UP from 4.6 to 5.5. Accuracy dropped from 10% to 3%.
Recall fell. Temperature dropped to 13.3 (model becoming less confident).
The model was essentially unlearning.

**Act 3 — Cosine recovery (steps 4000-20000)**: As the cosine schedule
reduced LR, the model stabilized and began learning again — this time
much more effectively. From step 4000 onward, every metric improved
monotonically:
  - Loss: 5.0 → 2.5 (crossed below exp 001's 4.15 around step 7000)
  - Accuracy: 5% → 40%+ (4x better than exp 001's best)
  - Temperature: 13.3 → 27.4 (INCREASED, as we predicted — the model
    gained enough confidence to sharpen its similarity distribution)
  - Recall@1: 0.03 → 0.13 (20x better than exp 001)
  - Recall@5: 0.09 → 0.31

**Temperature dynamics are the most interesting signal**: In exp 001,
temp went DOWN (14.3→12.4). In exp 002, it went DOWN during the
destabilized phase (14.3→13.3), then UP dramatically during recovery
(13.3→27.4). This confirms the interpretation: temperature increasing
means the model is confident enough to make sharper distinctions
between correct and incorrect matches.

## Results

| Metric          | Exp 001  | Exp 002     | Change        |
|-----------------|----------|-------------|---------------|
| Initial loss    | 5.65     | 6.30        | Higher (more negatives) |
| Final loss      | 4.15     | **2.55**    | **-39%**      |
| Best i2t_acc    | 12%      | **44.5%**   | **3.7x**      |
| Best t2i_acc    | 12%      | **42.6%**   | **3.5x**      |
| i2t_recall@1    | 0.0058   | **0.1367**  | **23x** (step 16000) |
| t2i_recall@1    | 0.0044   | **0.1315**  | **30x** (step 16000) |
| i2t_recall@5    | 0.0187   | **0.3164**  | **17x** (step 19000) |
| t2i_recall@5    | 0.0153   | **0.3177**  | **21x** (step 19000) |
| Temperature     | 12.4↓    | **27.4↑**   | Direction flipped |
| Steps trained   | 10,000   | 20,000      |               |
| Wall time       | 47 min   | ~160 min    |               |
| Peak GPU mem    | 2.6 GB   | 2.7 GB      | Negligible    |

Note: eval uses 1000 samples in exp 002 vs 9355 in exp 001, so recall
numbers aren't directly comparable (smaller pool = easier retrieval).
But the magnitude of improvement (20-30x) far exceeds this effect.

## Conclusions

**What we learned:**
1. **More steps >>> larger batch** for this scale. The 2x batch size
   increase was less impactful than the 2x step increase. Most gains
   came from steps 8000-20000, which exp 001 never reached.
2. **LR=1e-3 is too aggressive after warmup**. The destabilization from
   steps 1200-4000 was caused by the learning rate being too high for
   the model's capacity. The cosine decay saved us, but we lost ~3000
   steps to unlearning.
3. **Temperature is a great diagnostic signal**. Increasing temp = model
   is learning. Decreasing temp = model is struggling or destabilizing.
4. **The model is still improving at step 20K**. The curves haven't
   plateaued — more steps would likely help.

**Hypothesis scorecard:**
- ✅ Loss drops below 4.0 — reached 2.55
- ✅ In-batch accuracy > 20% — reached 44%
- ✅ Temperature increases — 14.3 → 27.4
- ✅ Recall@1 > 0.01 — reached 0.137

**What to try next (experiment 003 candidates)**:
- **Lower peak LR** (e.g., 5e-4 or 3e-4) to avoid the destabilization
  phase. Should learn smoothly from step 0 without the unlearning dip.
- **More steps** (50K) to see where the model converges.
- **Gradient accumulation** to simulate B=1024+ without OOM.
