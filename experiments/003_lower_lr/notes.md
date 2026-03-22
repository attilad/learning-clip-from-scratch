# Experiment 003 — Lower LR: 3e-4, B=512, 20K Steps

## Hypothesis

LR=1e-3 caused destabilization after warmup in exp 002 (steps 1200-4000,
loss went backwards from 4.6 to 5.5). Reducing peak LR to 3e-4 should:

- Eliminate the destabilization phase entirely
- Let the model use all 20K steps productively
- Converge to a better final loss since it never gets knocked off track
- Temperature should increase steadily from the start

**Risk**: LR might be too conservative, learning too slowly to match
exp 002's final metrics in only 20K steps.

**Predictions**:
- No mid-training accuracy dip
- Final loss < 2.5 (vs 2.55 in exp 002)
- recall@1 > 0.15 (vs 0.137 in exp 002)
- Smooth temperature increase from step 0

## Method

**Config changes from experiment 002**:
| Parameter       | Exp 002 | Exp 003 | Why                              |
|-----------------|---------|---------|----------------------------------|
| LR              | 1e-3    | 3e-4    | Avoid post-warmup destabilization|
| All else        | same    | same    | Isolate the LR effect            |

**Command**:
```bash
uv run --no-sync python -m main --batch-size 512 --total-steps 20000 --lr 3e-4
```

## Observations

**No destabilization at all.** Every metric improved monotonically from
step 0. This confirms the hypothesis: lr=1e-3 was the root cause of
exp 002's mid-training regression.

**Dramatically better results across the board.** The model reached
exp 002's final recall@1 (0.137) by step 3500 — only 17% of the way
through training. By the end it nearly doubled it.

**Loss reached 0.3** — down from 2.55 in exp 002. This is an 8x
improvement in final loss. The model is classifying 90% of in-batch
pairs correctly at B=512.

**Temperature climbed to 39.7** — monotonically increasing from 14.3.
Compare to exp 002's V-shaped dip-then-recovery to 27.4. The model
was confident from the start and kept sharpening.

**Recall plateaued around step 12K-13K** at ~0.26 recall@1 and ~0.51
recall@5. The last 7K steps showed diminishing returns, suggesting
we're approaching the ceiling for this model/data combination at
this scale.

**Training progression**:
| Step  | Loss  | i2t_acc | temp | i2t_R@1 | i2t_R@5 |
|-------|-------|---------|------|---------|---------|
| 0     | 6.30  | 0.0%    | 14.3 | —       | —       |
| 1000  | 4.85  | 7%      | 14.3 | 0.065   | 0.172   |
| 2000  | 4.05  | 14%     | 14.7 | 0.098   | 0.229   |
| 5000  | 2.65  | 38%     | 18.5 | 0.167   | 0.361   |
| 10000 | 0.78  | 75%     | 30.5 | 0.231   | 0.491   |
| 15000 | 0.37  | 88%     | 37.7 | 0.257   | 0.497   |
| 20000 | 0.30  | 90%     | 39.7 | 0.262   | 0.489   |

## Results

| Metric          | Exp 001  | Exp 002  | Exp 003       |
|-----------------|----------|----------|---------------|
| Initial loss    | 5.65     | 6.30     | 6.30          |
| Final loss      | 4.15     | 2.55     | **0.30**      |
| Best i2t_acc    | 12%      | 44%      | **92%**       |
| Best t2i_acc    | 12%      | 43%      | **90%**       |
| i2t_recall@1    | 0.006    | 0.137    | **0.276**     |
| t2i_recall@1    | 0.004    | 0.132    | **0.269**     |
| i2t_recall@5    | 0.019    | 0.316    | **0.525**     |
| t2i_recall@5    | 0.015    | 0.318    | **0.527**     |
| Temperature     | 12.4↓    | 27.4↑    | **39.7↑↑**    |
| Destabilized?   | no*      | yes      | **no**        |
| Steps trained   | 10,000   | 20,000   | 20,000        |
| Wall time       | 47 min   | ~160 min | ~160 min      |

*Exp 001 didn't destabilize because it stopped before the LR could cause damage.

## Conclusions

**What we learned:**
1. **LR was the single biggest lever.** Reducing from 1e-3 to 3e-4
   improved final loss by 8.5x (2.55 → 0.30) and recall by 2x
   (0.137 → 0.276). The batch size change from exp 001→002 was
   important, but the LR fix was transformative.

2. **The model is overfitting.** Loss of 0.3 with 90% in-batch accuracy
   means the model memorizes most of the batch. But recall@1 plateaued
   at ~0.26 — the model doesn't generalize perfectly to held-out data.
   This is the classic overfitting signal: training metric keeps improving,
   eval metric saturates.

3. **We're near the ceiling for this setup.** With 1M training pairs,
   B=512, ViT-B/32, the model has learned most of what it can learn.
   Further improvements would need:
   - More data (download more CC3M, or add another dataset)
   - Larger model (ViT-B/16 or ViT-L/14)
   - Regularization (augmentation, dropout, weight decay tuning)

**Hypothesis scorecard:**
- ✅ No mid-training accuracy dip — confirmed, smooth learning throughout
- ✅ Final loss < 2.5 — far exceeded: 0.30
- ✅ recall@1 > 0.15 — reached 0.276
- ✅ Smooth temperature increase — 14.3 → 39.7, monotonic
