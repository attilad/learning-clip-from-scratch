# Experiment 004 — Gradient Accumulation: Effective B=2048

## Hypothesis

Exp 003's recall@1 plateaued at ~0.276 after step 12K. One possible cause:
the gradient estimate from B=512 is noisy. Accumulating gradients over 4
micro-batches (effective B=2048) should:

- Provide smoother, more reliable gradient estimates
- Potentially break through the recall plateau
- NOT increase the number of negatives per contrastive loss call (still 511)

**Predictions**:
- Recall@1 > 0.276 (exp 003's plateau)
- Smoother loss curve (less batch-to-batch variance)
- Wall time ~4x longer per optimizer step

## Method

**Config**:
| Parameter       | Exp 003  | Exp 004    |
|-----------------|----------|------------|
| Batch size      | 512      | 512        |
| Accum freq      | 1        | 4          |
| Effective batch | 512      | 2048       |
| LR              | 3e-4     | 3e-4       |
| Optimizer steps | 20,000   | 20,000     |
| Micro-steps     | 20,000   | 80,000     |

**Command**:
```bash
uv run --no-sync python -m main --batch-size 512 --total-steps 20000 --lr 3e-4 --accum-freq 4
```

## Observations

**Training metrics improved dramatically over exp 003**:
- Final loss: 0.09 (vs 0.30 in exp 003)
- In-batch accuracy: 97% (vs 90%)
- Temperature: 59.1 (vs 39.7)

**But eval recall DECREASED — the opposite of what we predicted**:
- Best recall@1: ~0.200 at step 17500-18500 (vs 0.276 in exp 003)
- Best recall@5: ~0.410 at step 8500 (vs 0.525 in exp 003)
- Recall plateaued around step 5000-6000 at ~0.17-0.19

**This is a clear overfitting signal**: the model memorizes the training
data even more aggressively (97% vs 90% train accuracy) while generalizing
WORSE to held-out data. Gradient accumulation gave the optimizer more
precise gradient estimates, which it used to memorize more efficiently.

**Why this happened**: With accum_freq=4, the model sees 4x more data per
optimizer step (80K micro-steps vs 20K). At ~1M training samples with
B=512, that's ~9.6 epochs of unique data per epoch of optimizer steps —
effectively 38 epochs of data exposure over 20K optimizer steps (vs ~9.6
epochs in exp 003). The model had much more opportunity to memorize.

**Temperature at 59.1 confirms extreme confidence**: The model is making
very sharp, decisive predictions — but on training data, not on held-out
data. This is the textbook symptom of overfitting in contrastive learning.

## Results

| Metric          | Exp 003 (B=512) | Exp 004 (eff B=2048) | Change    |
|-----------------|-----------------|----------------------|-----------|
| Final loss      | 0.30            | **0.09**             | Better    |
| Best i2t_acc    | 92%             | **97%**              | Better    |
| Best t2i_acc    | 90%             | **97%**              | Better    |
| i2t_recall@1    | 0.276           | **0.200**            | **Worse** |
| t2i_recall@1    | 0.269           | **0.200**            | **Worse** |
| i2t_recall@5    | 0.525           | **0.410**            | **Worse** |
| t2i_recall@5    | 0.527           | **0.400**            | **Worse** |
| Temperature     | 39.7            | 59.1                 |           |
| Wall time       | ~160 min        | ~600 min             | 4x slower |

## Conclusions

**What we learned:**

1. **Gradient accumulation ≠ larger batch for contrastive learning.**
   True batch size increases give more negatives per loss call (bigger
   similarity matrix). Gradient accumulation only smooths the gradient
   estimate. At our scale (~1M pairs), gradient noise from B=512 was
   NOT the bottleneck — data diversity and model capacity are.

2. **More data exposure without more unique data → overfitting.** The
   80K micro-steps gave the model 4x more passes over the training set,
   which it used to memorize rather than generalize. This confirms exp
   003's finding: we're at the data ceiling for this setup.

3. **Temperature as a diagnostic**: 59.1 in exp 004 vs 39.7 in exp 003.
   Both increased (good), but the extreme 59.1 combined with declining
   eval recall means the model is confident but wrong on new data.

**Hypothesis scorecard:**
- ❌ Recall@1 > 0.276 — actually decreased to 0.200
- ✅ Smoother loss curve — confirmed
- ✅ Wall time ~4x — confirmed (~600 min vs 160 min)

**Key insight for the curriculum**: To improve generalization at this
scale, we need either:
- More unique training data (not more passes over the same data)
- A loss that's less susceptible to memorization (→ exp 005: SigLIP)
- Regularization (augmentation, dropout)
- Fine-tuning a pretrained model instead of training from scratch
