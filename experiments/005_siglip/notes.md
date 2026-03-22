# Experiment 005 — SigLIP Sigmoid Loss

## Hypothesis

Replace InfoNCE with SigLIP's sigmoid pairwise loss. SigLIP evaluates
each image-text pair independently ("match or not?") rather than as a
global ranking ("which of B texts matches?"). The literature shows it
outperforms InfoNCE at batch sizes below ~16K — exactly our regime.

**Predictions**:
- recall@1 > 0.276 (exp 003's best with InfoNCE)
- Less overfitting gap (train vs eval metrics closer together)
- Different temperature dynamics (SigLIP also has a learnable bias)

## Method

**Config**:
| Parameter       | Exp 003 (InfoNCE) | Exp 005 (SigLIP) |
|-----------------|-------------------|-------------------|
| Loss function   | CLIP (InfoNCE)    | SigLIP (sigmoid)  |
| Batch size      | 512               | 512               |
| LR              | 3e-4              | 3e-4              |
| Steps           | 20,000            | 20,000            |
| All else        | same              | same              |

**Command**:
```bash
uv run --no-sync python -m main --batch-size 512 --total-steps 20000 --lr 3e-4 --loss siglip
```

## Observations

**SigLIP matched InfoNCE's generalization almost exactly.** The recall
curves track each other remarkably closely:

| Step  | InfoNCE R@1 | SigLIP R@1 | InfoNCE R@5 | SigLIP R@5 |
|-------|-------------|------------|-------------|------------|
| 2000  | 0.098       | 0.085      | 0.229       | 0.215      |
| 5000  | 0.167       | 0.164      | 0.361       | 0.353      |
| 10000 | 0.231       | 0.231      | 0.491       | 0.473      |
| 15000 | 0.257       | 0.250      | 0.497       | 0.490      |
| 20000 | 0.262       | 0.258      | 0.489       | 0.481      |

**SigLIP overfit LESS on training metrics.** Final in-batch accuracy
was 89-90% vs InfoNCE's 90-92%. Final temperature was 29.0 vs 39.7.
The model was less aggressively confident, which is consistent with
the sigmoid loss providing a different optimization landscape.

**Loss values are not comparable.** SigLIP loss (per-pair BCE averaged
over B² pairs) started at 0.020 and ended at 0.002. InfoNCE (B-way
cross-entropy) started at 6.3 and ended at 0.30. Different scales,
different interpretations.

**The logit_bias settled around its initial value.** It started at -10.0
and didn't change dramatically, suggesting the default initialization
was reasonable for this dataset/scale.

**No destabilization**, same as exp 003. Smooth learning throughout.

## Results

| Metric          | Exp 003 (InfoNCE) | Exp 005 (SigLIP)  | Delta     |
|-----------------|--------------------|--------------------|-----------|
| Final loss      | 0.30               | 0.002              | Different scales |
| Best i2t_acc    | 92%                | 90%                | ~same     |
| Best t2i_acc    | 90%                | 90%                | ~same     |
| i2t_recall@1    | 0.276              | 0.259              | -0.017    |
| t2i_recall@1    | 0.269              | 0.270              | +0.001    |
| i2t_recall@5    | 0.525              | 0.529              | +0.004    |
| t2i_recall@5    | 0.527              | 0.525              | -0.002    |
| Temperature     | 39.7               | 29.0               |           |
| Logit bias      | n/a                | ~-10               |           |
| Wall time       | ~160 min           | ~160 min           | same      |

## Conclusions

**What we learned:**

1. **SigLIP ≈ InfoNCE at this scale.** The recall numbers are within
   noise of each other. The literature's claim that SigLIP outperforms
   InfoNCE at small batch sizes may require B < 512 or much larger
   datasets to manifest. At our B=512 / 1M pairs scale, the loss
   function is NOT the bottleneck.

2. **The recall ceiling (~0.26-0.28 R@1) is confirmed as a data/model
   limit, not a loss function artifact.** Both InfoNCE and SigLIP hit
   the same wall. This rules out loss function as the cause of our
   plateau.

3. **SigLIP shows slightly less overfitting** (lower train accuracy,
   lower temperature) but this didn't translate to better eval metrics.
   The overfitting gap is real but narrow at this scale.

4. **SigLIP is a viable drop-in replacement.** Same speed, same memory,
   comparable results. For future experiments at much larger batch sizes
   (via multi-GPU or different hardware), SigLIP's memory efficiency
   advantages would matter more.

**Hypothesis scorecard:**
- ❌ recall@1 > 0.276 — matched but didn't exceed (0.259-0.270)
- ⚠️ Less overfitting — slightly less, but didn't help eval
- ✅ Different temp dynamics — 29.0 vs 39.7, less aggressive

**What this means for the curriculum:** The from-scratch path on 1M CC3M
pairs has been thoroughly explored. Experiments 003-005 all converge to
recall@1 ≈ 0.26-0.28 regardless of loss function, gradient accumulation,
or batch size. The next real gains require either more data or starting
from a pretrained checkpoint (fine-tuning path in the lesson plan).
