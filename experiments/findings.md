# Findings: Training CLIP from Scratch to Fine-Tuning Mastery

A systematic investigation of contrastive vision-language training on a single RTX 4090. Seven experiments across two phases, each isolating one variable and recording what actually happened.

## Executive Summary

We trained CLIP (ViT-B/32, 151M params) from random initialization on ~1M CC3M image-text pairs, hit a recall ceiling at R@1=0.276 in this regime, then switched to adapting a pretrained model and reached R@1=0.772 — a 2.8x improvement. Along the way we discovered that learning rate matters more than batch size, loss function, or gradient quality; that the learned logit scale (temperature) is a high-value diagnostic for training dynamics; and that weight interpolation between fine-tuned and pretrained models creates out-of-distribution performance that neither model had alone.

### The numbers at a glance

| Experiment | Method | i2t R@1 | Key insight |
|---|---|---|---|
| 001 | Baseline (B=256, lr=1e-3, 10K) | 0.006 | It learns, but barely |
| 002 | Larger batch + longer (B=512, 20K) | 0.137 | LR=1e-3 destabilizes mid-training |
| 003 | **Lower LR** (lr=3e-4, 20K) | **0.276** | LR is the single biggest lever |
| 004 | Grad accumulation (eff B=2048) | 0.200 | More passes = more overfitting |
| 005 | SigLIP loss | 0.259 | Same ceiling in this regime |
| 007 | Fine-tune pretrained (lr=1e-5) | **0.772** | Pretrained is a different universe |
| 008 | LoRA / WiSE-FT / freeze | 0.729–0.768 | No forgetting on CIFAR-100; WiSE-FT is free lunch |

---

## Phase 1: From-Scratch Training (Experiments 001–005)

### Experiment 001 — Baseline

**Setup:** B=256, lr=1e-3, 10K steps, BF16 mixed precision, cosine schedule with 10% warmup.

**Results:** Loss 5.65 → 4.15. In-batch accuracy reached 12% (vs 0.4% random chance). Recall@1 = 0.006 — barely above random. Temperature *decreased* from 14.3 to 12.4, which was unexpected.

**What we learned:** The pipeline works end-to-end and the model is learning *something*, but 10K steps at B=256 isn't enough to build a useful embedding space. The decreasing temperature was our first clue that temperature tracks training health — when the model is struggling, it hedges by softening its predictions.

---

### Experiment 002 — Larger Batch, More Steps

**Setup:** B=512 (2x negatives per contrastive loss), 20K steps (2x longer). Everything else unchanged.

**Results:** Massive improvement — R@1 jumped from 0.006 to 0.137 (23x). But the training curve told a three-act story:

| Phase | Steps | What happened |
|---|---|---|
| Act 1: Fast learning | 0–1200 | Loss dropped, accuracy climbed to 10%, recall peaked at 0.053 |
| Act 2: Destabilization | 1200–4000 | Loss *climbed back*, accuracy collapsed to 3%, model was unlearning |
| Act 3: Recovery | 4000–20K | Cosine decay rescued it; steady climb to 44% accuracy, R@1=0.137 |

**What we learned:** LR=1e-3 is too aggressive for this model. After warmup ended, full-strength updates destabilized the learned representations. The cosine decay schedule saved the run by gradually reducing the LR, but the model wasted ~3000 steps recovering from self-inflicted damage. Temperature confirmed the story: it dropped during Act 2 (model struggling) and climbed during Act 3 (model gaining confidence).

---

### Experiment 003 — Lower Learning Rate

**Setup:** B=512, lr=3e-4 (3.3x lower), 20K steps.

**Results:** This was the breakthrough from-scratch run.

| Metric | Exp 002 (lr=1e-3) | Exp 003 (lr=3e-4) | Improvement |
|---|---|---|---|
| Final loss | 2.55 | **0.30** | 8.5x |
| In-batch accuracy | 44% | **90%** | 2x |
| i2t R@1 | 0.137 | **0.276** | 2x |
| i2t R@5 | 0.316 | **0.525** | 1.7x |
| Temperature | 27.4 | **39.7** | Higher confidence |

No mid-training destabilization. Smooth, monotonic improvement across all metrics. Temperature climbed steadily from 14.3 to 39.7 — the model was consistently gaining confidence.

**What we learned:** Learning rate was the single biggest lever in the entire from-scratch phase. A 3.3x reduction in LR gave an 8.5x improvement in final loss and 2x improvement in recall. The model also showed clear signs of overfitting by the end: 90% train accuracy but recall plateaued at ~0.27 after step 15K. This hinted at a ceiling.

---

### Experiment 004 — Gradient Accumulation

**Setup:** B=512 micro-batch, accum_freq=4, effective B=2048. Same lr=3e-4, 20K optimizer steps (80K micro-steps). This tests whether smoother gradients help.

**Results:**

| Metric | Exp 003 (B=512) | Exp 004 (eff B=2048) |
|---|---|---|
| Final loss | 0.30 | **0.09** (better) |
| Train accuracy | 92% | **97%** (better) |
| i2t R@1 | **0.276** | 0.200 (worse) |
| Temperature | 39.7 | **59.1** (extreme) |
| Wall time | ~160 min | ~600 min (4x slower) |

Training metrics improved. Eval metrics got *worse*. Classic overfitting.

**What we learned:** Gradient accumulation ≠ larger batch for contrastive learning. Larger batches add more negatives per loss computation (bigger similarity matrix). Grad accum only smooths the gradient estimate — the contrastive loss still sees B=512 negatives per forward pass. With 80K micro-steps (4x more passes over the same 1M images), the model memorized the training set more efficiently. Temperature reached 59.1 — confidently wrong.

---

### Experiment 005 — SigLIP Loss

**Setup:** Replace InfoNCE (cross-entropy over similarity matrix) with SigLIP (independent sigmoid per pair). Same B=512, lr=3e-4, 20K steps.

**Results:**

| Metric | InfoNCE (exp 003) | SigLIP (exp 005) |
|---|---|---|
| i2t R@1 | **0.276** | 0.259 |
| t2i R@1 | 0.269 | **0.270** |
| i2t R@5 | 0.525 | **0.529** |
| Temperature | 39.7 | 29.0 |

Nearly identical retrieval performance. SigLIP showed slightly less overfitting (lower temperature, lower train accuracy) but hit the same recall ceiling.

**What we learned:** In this regime (1M CC3M subset, B=512, ViT-B/32), recall converged to ~0.26 regardless of loss function. However, this was a single-swap comparison — the SigLIP literature shows its advantage can depend on co-varying other recipe elements (weight decay, schedule length), and at B=512 the theoretical advantage over InfoNCE is modest. The more defensible conclusion: **loss function choice was not the binding constraint in our setup**. The binding constraint was data — quantity and likely quality, since CC3M captions are noisy and repetitive. Caption enrichment or larger datasets (CC12M, LAION) would be needed to test whether a different loss unlocks further gains.

### Phase 1 Conclusion

**In our regime (1M CC3M subset, ViT-B/32, B=512), recall converged to ~0.27 across three approaches:**

```
Exp 003 (InfoNCE, optimal LR):      R@1 = 0.276  ← best result
Exp 004 (smoother gradients):        R@1 = 0.200  ← overfitting
Exp 005 (different loss function):   R@1 = 0.259  ← same regime
```

The bottleneck is data — both quantity and quality. Scaling studies (OpenCLIP, DataComp) show that such ceilings routinely break with more data, better curation, or richer captions. This is a *regime ceiling* for our specific setup, not an intrinsic CLIP limitation. The natural next steps from here are (a) pretrained adaptation (which we pursued in Phase 2), (b) scaling data (CC12M, LAION), or (c) improving caption quality (LLM rewriting, BLIP-style filtering).

---

## Phase 2: Fine-Tuning Pretrained CLIP (Experiments 007–008)

### Experiment 007 — Fine-Tune Pretrained CLIP on CC3M

**Setup:** Start from OpenAI's pretrained ViT-B/32 (trained on 400M+ pairs). Fine-tune on CC3M at three learning rates: 1e-6, 5e-6, 1e-5. B=512, 5K steps each.

**Results — the pretrained model changes everything:**

| Configuration | i2t R@1 | vs from-scratch best |
|---|---|---|
| Pretrained zero-shot (no training) | 0.626 | 2.3x better |
| Fine-tuned lr=1e-6 | 0.725 | 2.6x |
| Fine-tuned lr=5e-6 | 0.741 | 2.7x |
| **Fine-tuned lr=1e-5** | **0.772** | **2.8x** |

The pretrained model, *without any training on our data*, already more than doubled our best from-scratch result. Fine-tuning pushed it further.

**LR sweep progression:**

| Step | lr=1e-6 | lr=5e-6 | lr=1e-5 |
|---|---|---|---|
| 0 | 0.626 | 0.626 | 0.626 |
| 1000 | 0.684 | 0.717 | 0.720 |
| 3000 | 0.719 | 0.742 | 0.750 |
| 5000 | 0.725 | 0.741 | **0.772** |

**What we learned:**

1. **Pretrained models exist in a different universe.** 400M pairs of pretraining knowledge dwarfs anything achievable from scratch on 1M. The zero-shot baseline alone was 2.3x our best trained model.

2. **lr=1e-5 beat the "safe" lr=5e-6.** This was surprising — the literature recommends conservative LRs for fine-tuning to avoid forgetting. Two possible explanations: CC3M's distribution is close enough to the pretraining data that larger updates are safe, or 5K steps is too short for forgetting to manifest.

3. **Temperature barely moved** (14.3 → 14.6). Unlike from-scratch training where temperature climbed from 14 to 40 as the model built confidence, the pretrained model was already well-calibrated. Fine-tuning nudged the alignment without disrupting the underlying representation.

4. **Fine-tuning is 4x faster.** 5K steps / 40 minutes vs 20K steps / 160 minutes, and it reaches 2.8x higher recall.

---

### Experiment 008 — Adaptation Methods & Catastrophic Forgetting

The key open question from exp 007: lr=1e-5 fine-tuning gave us great CC3M recall, but **does it destroy the model's general visual knowledge?** We only measured on CC3M — the same distribution we trained on. We needed an out-of-distribution test.

**Setup:** Compare 5 adaptation strategies, measured on two axes:
- **CC3M retrieval** (in-distribution): did adaptation help?
- **CIFAR-100 zero-shot classification** (out-of-distribution): did adaptation forget?

CIFAR-100 zero-shot classifies 10K test images into 100 categories using text prompts like "a photo of a {class}". The pretrained baseline scores 62.3% top-1 — any drop after fine-tuning indicates forgetting. **Caveat:** CIFAR-100 is still natural images with significant semantic overlap with CC3M. A more rigorous forgetting evaluation would use distribution-shift benchmarks (ImageNet variants, EuroSAT, DTD textures) or multi-benchmark suites like CLIP Benchmark / DataComp's 38-task eval.

| Method | Trainable params | What it does |
|---|---|---|
| Baseline | 0 | Pretrained model, no adaptation |
| Full fine-tune | 151M (100%) | Train everything at lr=1e-5 |
| Freeze backbone | 655K (0.4%) | Only train projection layers |
| LoRA rank=4 | 901K (0.6%) | Low-rank adapters on Q/V attention |
| WiSE-FT (α=0.5) | 151M → interpolate | Full FT, then blend with pretrained weights |

*LP-FT (two-phase: freeze then unfreeze) was planned but crashed due to an optimizer state mismatch on resume. Bug has been fixed.*

**Results:**

| Method | CC3M i2t R@1 | CIFAR-100 top1 | CIFAR-100 Δ | Temperature |
|---|---|---|---|---|
| Baseline | 0.626 | 0.623 | — | 14.3 |
| Full FT | **0.760** | 0.630 | +0.7 | 14.6 |
| Freeze backbone | 0.626 | 0.596 | **-2.7** | 14.6 |
| LoRA r=4 | 0.729 | 0.636 | +1.3 | **17.3** |
| WiSE-FT (post) | 0.734 | **0.663** | **+4.1** | — |

**What we learned:**

1. **No catastrophic forgetting on CIFAR-100 — but that may be the wrong test.** Full fine-tuning on CC3M didn't hurt CIFAR-100 at all — in fact, every method except frozen backbone *improved* it slightly. CC3M's visual concepts (people, animals, buildings, food, vehicles) overlap heavily with CIFAR-100's 100 classes, making it a weak stress test for forgetting. The WiSE-FT paper specifically evaluates against *distribution shift* (ImageNet-V2, ImageNet-R, ObjectNet, etc.), which is the failure mode fine-tuning actually causes. Our single-benchmark result confirms no forgetting *on natural images within CC3M's domain*, but says little about robustness under shift. Stronger eval suites are needed to make a general claim.

2. **WiSE-FT creates performance that neither model had alone.** Weight interpolation (50% fine-tuned + 50% pretrained) produced the best CIFAR-100 score of any configuration: 66.3%, beating even the pretrained baseline it was interpolated from (62.3%). The interpolated point sits in a better basin of the loss landscape than either endpoint. This is genuinely free — no extra training, just a weighted average of two state dicts.

3. **LoRA is the efficiency winner.** With just 0.6% of parameters (901K vs 151M), LoRA rank=4 achieved 96% of full fine-tuning's CC3M recall (0.729 vs 0.760) while preserving CIFAR-100 better than full FT. The cost: 0.9GB VRAM vs 2.7GB for full FT.

4. **Freezing the backbone was the worst strategy.** Projection-only training was the only method that *hurt* CIFAR-100 (−2.7 pts) while barely helping CC3M. Linear projections can rotate and scale the embedding space, but they can't reshape it. Optimizing projections for CC3M alignment pulled them away from the general-purpose alignment that made CIFAR-100 work.

5. **Temperature tracks adapter capacity.** LoRA's temperature spiked to 17.3 while all full-model methods stayed near 14.6. With only 900K trainable parameters, the LoRA adapters compensated by pushing the model toward higher confidence — effectively amplifying the small changes they could make to attention patterns. This temperature divergence is a useful signal: it may indicate that the adapter rank is becoming a bottleneck.

**The Pareto frontier (CIFAR-100 only):**

```
CIFAR-100 top1 ↑
  0.663 |                                    * WiSE-FT       ← best OOD
  0.636 |                    * LoRA     * WiSE-FT (pre)
  0.630 |                                 * Full FT          ← best ID
  0.623 | * Baseline
  0.596 |   * Freeze                                         ← worst
        +----+-------+-------+-------+-------+-------→ CC3M i2t R@1
           0.62    0.66    0.70    0.74    0.78
```

WiSE-FT dominates the upper-right (best tradeoff). LoRA dominates on efficiency.

**Update — exp 009 revealed this CIFAR-100 picture was misleading.** See below.

---

### Experiment 009 — Multi-Dataset Evaluation (Exp 008 Revisited)

Exp 008's CIFAR-100-only evaluation was flagged by peer review as insufficient. Exp 009 retroactively evaluated all checkpoints across 6 diverse zero-shot classification benchmarks to properly stress-test forgetting.

**Datasets span a distributional distance spectrum:**
- **Near CC3M:** CIFAR-100 (objects), Food-101 (food categories)
- **Medium:** FGVC-Aircraft (fine-grained aircraft variants)
- **Far:** DTD (textures), EuroSAT (satellite land-use), GTSRB (traffic signs)

**Results (Δ vs pretrained baseline):**

| Dataset | Full FT | Freeze | LoRA r=4 | WiSE-FT | Distance |
|---|---|---|---|---|---|
| CIFAR-100 | +0.7% | -2.5% | +1.1% | **+3.9%** | near |
| Food-101 | **-9.4%** | -17.0% | -9.4% | -0.7% | near |
| FGVC-Aircraft | -5.6% | -1.7% | -2.5% | **+0.9%** | medium |
| DTD | -3.8% | -4.0% | -1.0% | **+0.5%** | far |
| EuroSAT | +4.7% | +0.1% | **+8.5%** | +9.5% | far |
| GTSRB | **-12.1%** | -3.1% | -11.9% | -2.0% | far |
| **Average** | **-4.3%** | **-4.7%** | **-2.5%** | **+2.0%** | |

**The peer review was right. Full fine-tuning DOES cause forgetting.**

CIFAR-100 completely missed it because its distribution overlaps with CC3M. Across 6 benchmarks:
- Full FT loses an average of 4.3%, with worst cases of -12.1% (traffic signs) and -9.4% (food).
- GTSRB (traffic signs) was catastrophic for both full FT and LoRA (~-12%). German traffic signs are so distant from CC3M that any CC3M-directed adaptation destroys them.
- Food-101 (-9.4%) was surprisingly bad — food images are common in web data, but CC3M's noisy captions may degrade fine-grained food discrimination.
- EuroSAT was the one exception where fine-tuning helped (+4.7% to +9.5%), possibly because CC3M contains landscape/aerial photos.

**WiSE-FT is even more valuable than we thought.** It's the only method with a positive average (+2.0%). On GTSRB, it limited forgetting to -2.0% while full FT lost -12.1%. Weight interpolation isn't just a nice-to-have — it's essential for preserving general capability.

**LoRA is more robust than full FT but shares the same vulnerabilities.** Average -2.5% vs -4.3%, but both collapsed on GTSRB. LoRA's constrained capacity forces more generalizable adaptations on most tasks, but can't protect against extreme domain shift.

---

## Cross-Cutting Insights

### Temperature (logit scale) is a high-value diagnostic

CLIP's learned temperature is the `logit_scale` parameter — initialized as `log(1/0.07)` so `exp(logit_scale) ≈ 14.3` — that scales the similarity matrix before the contrastive loss. It's not a hyperparameter; it's optimized jointly with the model. Across our seven experiments, it was consistently one of the most informative signals for training dynamics, though it should be read alongside other metrics (loss, eval recall, gradient norms, embedding norms) rather than in isolation. Temperature can drift for multiple reasons beyond "training health" — batch composition, hard negative density, and optimizer dynamics all influence it.

| Behavior | Meaning | Example |
|---|---|---|
| Steady increase | Model gaining confidence, learning is healthy | Exp 003: 14.3 → 39.7 |
| Decrease | Model struggling, hedging predictions | Exp 001: 14.3 → 12.4 |
| V-shape (down then up) | Destabilization then recovery | Exp 002: 14.3 → 12 → 27.4 |
| Barely moves | Already well-calibrated (pretrained) | Exp 007: 14.3 → 14.6 |
| Spikes with few params | Adapter capacity bottleneck | Exp 008 LoRA: 14.3 → 17.3 |
| Extreme value + declining eval | Confidently overfitting | Exp 004: 14.3 → 59.1 |

### Learning rate dominates optimization

| Change | Effect on R@1 |
|---|---|
| LR 1e-3 → 3e-4 (exp 002 → 003) | 0.137 → 0.276 **(+100%)** |
| B=512 → eff B=2048 (exp 003 → 004) | 0.276 → 0.200 **(−28%)** |
| InfoNCE → SigLIP (exp 003 → 005) | 0.276 → 0.259 **(−6%)** |

A single hyperparameter change (LR) gave a 2x improvement. Changing the loss function or effective batch size gave marginal or negative returns.

### The data ceiling in this regime

Three approaches converged to R@1 ≈ 0.26–0.28 on our 1M CC3M subset:
- Optimal LR (exp 003): 0.276
- Different loss function (exp 005): 0.259
- Grad accumulation (exp 004): 0.200 (overfit past the ceiling)

Breaking through required pretrained knowledge from 400M+ pairs (exp 007: 0.772).

**Important context from scaling research:** This ceiling is specific to our regime (1M pairs, ViT-B/32, B=512). OpenCLIP and DataComp scaling studies demonstrate power-law improvements with more data, and DataComp makes dataset curation itself the benchmark variable. Many "ceilings" in contrastive learning are really *data coverage ceilings* — they break when you add data diversity, improve caption quality, or filter for higher-quality pairs. CC3M at 1M pairs is a particularly constrained regime; CC12M (12M pairs), LAION-400M, or even caption-enriched CC3M would likely shift the curve.

### Overfitting signals to watch for

| Signal | What it means |
|---|---|
| Train accuracy >> eval recall | Memorizing training set |
| Temperature > 40 | Overconfident predictions |
| Loss improving but recall flat | Optimizing for training distribution, not generalization |
| Grad accum improves loss but hurts eval | More passes over same data = memorization |

---

## Conclusions

1. **Learning rate is the single biggest lever for from-scratch training.** Not batch size, not loss function, not gradient quality. Get the LR right first. This is consistent with broader fine-tuning literature — WiSE-FT explicitly notes that outcomes vary substantially with small hyperparameter changes.

2. **Data is the binding constraint at this scale.** Our 1M CC3M subset hit R@1≈0.27 across three optimization approaches. This is a regime ceiling, not an intrinsic limit — scaling studies show power-law improvements with more data and better curation. The next step to test this is data-centric: caption enrichment (BLIP/VeCLIP-style), CC12M, or quality filtering (DataComp/DFN).

3. **Pretrained models are a different universe.** OpenAI's CLIP, trained on 400M pairs, provided 2.3x better zero-shot recall than our best trained model — before any fine-tuning. After fine-tuning: 2.8x. This directly confirms the CLIP paper's core thesis that scale is central to capability.

4. **WiSE-FT is free lunch for adaptation.** Interpolating fine-tuned and pretrained weights costs nothing and produces a model that outperforms either endpoint on our OOD test. This aligns with the WiSE-FT paper and the broader "model soups" literature showing that weight averaging often improves both accuracy and robustness.

5. **LoRA is remarkably parameter-efficient.** Rank-4 adapters on attention layers (0.6% of params) capture 96% of full fine-tuning's in-distribution gain while being 3x more memory-efficient. This is consistent with the general PEFT literature across modalities.

6. **Full fine-tuning DOES cause forgetting — CIFAR-100 just missed it.** Exp 009's multi-dataset evaluation revealed that full FT averages -4.3% across 6 benchmarks, with worst cases of -12.1% (GTSRB traffic signs) and -9.4% (Food-101). CIFAR-100 showed +0.7% because its distribution overlaps with CC3M. **You cannot assess forgetting with a single benchmark.** Evaluation breadth matters more than depth.

7. **Temperature (logit scale) is a high-value training diagnostic.** It consistently tracked training dynamics across all experiments — confidence, destabilization, overfitting, and adapter saturation. It should be monitored alongside other signals (eval metrics, gradient norms, embedding norms) rather than treated as a sole oracle, since it can drift for reasons unrelated to training quality.

---

## Limitations and Open Questions

This work has several limitations that bound the generality of our conclusions:

- **Single dataset regime.** All from-scratch experiments used ~1M pairs from a single source (CC3M). Scaling studies show performance improves monotonically with data quantity and diversity. Our "ceiling" is regime-specific.

- **Narrow OOD evaluation.** CIFAR-100 is natural images with heavy semantic overlap to CC3M. We did not evaluate on distribution-shift benchmarks (ImageNet-V2/R/Sketch, ObjectNet), domain-distant tasks (EuroSAT, DTD, FGVC-Aircraft), or the 38-dataset suites used by DataComp. Our "no forgetting" conclusion holds only for the narrow slice we tested.

- **SigLIP comparison was single-swap.** We changed only the loss function while keeping all other recipe elements fixed. SigLIP's advantage may depend on co-varying weight decay, schedule length, or training duration. A fairer comparison would jointly tune the recipe for each loss.

- **No cross-batch negative techniques.** Our contrastive training was limited to in-batch negatives (max 512). Techniques like MoCo's momentum queue or Cross-Batch Memory (XBM) could increase effective negatives without more VRAM, potentially shifting the ceiling even within 1M pairs.

- **Single model architecture.** All experiments used ViT-B/32. Different model sizes or architectures could change the scaling dynamics.

These limitations define the natural next directions for this project.

---

## Acknowledgments

This report benefited from a peer review that contextualized our findings against modern vision-language research (OpenCLIP scaling laws, DataComp, SigLIP/SigLIP2, WiSE-FT, BLIP, VeCLIP). The review validated our core findings on pretrained dominance, WiSE-FT benefits, and gradient accumulation mechanics, while correctly identifying that our "hard ceiling" and "no forgetting" claims were over-generalized from a narrow evaluation regime. The limitations section and updated lesson plan reflect that feedback.
