# Experiment 008: Adaptation Methods & Catastrophic Forgetting

## Date
2026-03-23

## Hypothesis
Full fine-tuning (exp 007's lr=1e-5) maximizes in-distribution CC3M retrieval
but destroys general visual knowledge. Parameter-efficient methods (LoRA, frozen
backbone) and post-hoc corrections (WiSE-FT, LP-FT) can recover the
in-distribution/OOD tradeoff.

Specific predictions:
1. Full FT wins CC3M recall, loses most CIFAR-100 zero-shot accuracy
2. Frozen backbone preserves CIFAR-100 but barely improves CC3M over baseline
3. LP-FT and WiSE-FT offer the best tradeoff (both metrics improve vs baseline)
4. LoRA matches full FT on CC3M while preserving more CIFAR-100

## Method

| ID | Method | What trains | LR | Steps | Trainable params |
|----|--------|-------------|------|-------|------------------|
| baseline | Pretrained (no adapt) | Nothing | - | 0 | 0 |
| A | Full fine-tune | All 151M params | 1e-5 | 5K | 151,277,313 |
| B | Freeze backbone | Projections only | 1e-5 | 5K | 655,360 |
| C | LP-FT | Phase 1: proj (2.5K), Phase 2: all (2.5K) | 1e-5 | 5K | 655K → 151M |
| D | LoRA r=4 | Attention adapters + projections | 1e-4 | 5K | 901,120 |
| E | WiSE-FT (α=0.5) | Full FT then interpolate with pretrained | 1e-5 | 5K | 151,277,313 |

All runs: B=512, pretrained=openai, CLIP InfoNCE loss.

### Metrics
- **In-distribution**: i2t_recall@1 on CC3M eval (1K samples)
- **Out-of-distribution**: CIFAR-100 zero-shot top-1 accuracy (10K test images)

### Key details
- LoRA: rank-4, applied to Q and V projections in all 24 attention layers (12 visual + 12 text)
- WiSE-FT: theta = 0.5 * theta_FT + 0.5 * theta_pretrained (after full FT completes)
- LP-FT: Phase 1 trains only projection layers, phase 2 resumes with all unfrozen
- CIFAR-100 prompts: 7-template ensemble from CLIP paper

## Results

| Method | CC3M i2t R@1 | CC3M t2i R@1 | CIFAR-100 top1 | CIFAR-100 top5 | Temp | Wall time |
|--------|-------------|-------------|----------------|----------------|------|-----------|
| Baseline | 0.626 | 0.609 | 0.623 | 0.872 | 14.3 | — |
| Full FT (A) | **0.760** | **0.755** | 0.630 | 0.877 | 14.6 | ~41 min |
| Freeze (B) | 0.626 | 0.630 | 0.596 (-2.7) | 0.854 | 14.6 | ~38 min |
| LP-FT (C) | — | — | — | — | — | CRASHED |
| LoRA (D) | 0.729 | 0.714 | 0.636 (+1.3) | 0.878 | **17.3** | ~40 min |
| WiSE-FT (E pre) | 0.768 | 0.745 | 0.636 | 0.880 | 14.6 | ~40 min |
| WiSE-FT (E post) | 0.734 | 0.729 | **0.663 (+4.1)** | **0.894** | — | — |

### Prediction scorecard
1. ❌ "Full FT loses most CIFAR-100" — **wrong!** Full FT barely changed CIFAR-100
   (0.623 → 0.630). No catastrophic forgetting detected on this benchmark.
2. ✅ "Frozen backbone barely improves CC3M" — **correct.** i2t R@1 stayed at 0.626.
   Surprising: it actually *hurt* CIFAR-100 by 2.7 pts.
3. ⚠️ "LP-FT offers best tradeoff" — **untested** (crashed on phase 2 resume).
   WiSE-FT confirmed as best tradeoff.
4. ✅ "LoRA preserves more CIFAR-100" — **correct.** LoRA: 0.636 vs Full FT: 0.630,
   though the gap is small. LoRA gets 96% of full FT's CC3M recall with 0.6% of params.

## Analysis

### Forgetting magnitude
**The big surprise: there's almost no forgetting on CIFAR-100.**

| Method | CIFAR-100 Δ vs baseline |
|--------|------------------------|
| Full FT (A) | +0.7 pts |
| Freeze (B) | **-2.7 pts** |
| LoRA (D) | +1.3 pts |
| WiSE-FT (E pre) | +1.4 pts |
| WiSE-FT (E post) | **+4.1 pts** |

Full fine-tuning on CC3M does NOT cause catastrophic forgetting on CIFAR-100.
In fact, most methods *improved* CIFAR-100 slightly. This likely means CC3M's
distribution overlaps enough with CIFAR-100's visual concepts that fine-tuning
is complementary, not destructive.

The only method that hurt CIFAR-100 was freezing the backbone — training only
the projection layers distorted the embedding alignment without the encoder
being able to compensate.

### Efficiency frontier

Pareto-optimal methods (best CC3M for a given CIFAR-100 level):

```
CIFAR-100 top1 ↑
  0.663 |                                    * WiSE-FT (post)    ← best OOD
  0.636 |                    * LoRA     * WiSE-FT (pre)
  0.630 |                                 * Full FT              ← best ID
  0.623 | * Baseline
  0.596 |   * Freeze                                             ← worst
        +----+-------+-------+-------+-------+-------→ CC3M i2t R@1
           0.62    0.66    0.70    0.74    0.78
```

**WiSE-FT post-interpolation is the clear Pareto winner**: best CIFAR-100
(0.663) while retaining strong CC3M recall (0.734). It "creates" OOD
performance that neither the pretrained nor fine-tuned model had alone.

**LoRA is the efficiency winner**: 0.6% of params, 96% of CC3M recall,
+1.3 pts on CIFAR-100. If compute or memory is constrained, LoRA dominates.

### Temperature dynamics
Most methods barely moved temperature (14.3 → 14.6), consistent with exp 007's
finding that pretrained calibration is already good.

**LoRA is the exception**: temperature climbed to 17.3. With only ~900K trainable
params (none in the encoders' main weights), the LoRA adapters seem to
compensate by pushing the model toward higher confidence. The temperature
effectively amplifies the small changes LoRA can make to the attention patterns.
This is a novel diagnostic signal: temperature climb may indicate that the
adapter capacity is becoming a bottleneck.

### LoRA observations
- Rank-4 provided enough capacity for 96% of full FT's CC3M gain
- 24 attention layers × 4 LoRA matrices each = 96 trainable param tensors
- All 96 had gradients flowing (verified in smoke test)
- The temperature spike (14.3 → 17.3) suggests the adapters are working hard —
  higher rank might close the remaining 4% gap with full FT
- GPU memory: 0.9GB vs 2.7GB for full FT — LoRA is ~3x more memory-efficient

### Frozen backbone failure mode
Freezing the backbone was the worst strategy — it barely helped CC3M and was
the only method to *hurt* CIFAR-100. This makes sense: the projection layers
are just linear transforms. Without the encoder adapting, the projections can
only rotate/scale the existing embedding space, not reshape it. And by
optimizing projections for CC3M alignment, we de-optimized them for the
general-purpose alignment that made CIFAR-100 zero-shot work.

### LP-FT crash (bug)
Phase 2 crashed with `ValueError: loaded state dict contains a parameter group
that doesn't match the size of optimizer's group`. The checkpoint from Phase 1
saved optimizer state for 655K params, but Phase 2 tried to load it with 151M
params. Fix: skip optimizer state loading when param count changes (use
`--resume` for weights only, rebuild optimizer fresh).

## Conclusions

1. **No catastrophic forgetting on CIFAR-100.** CC3M fine-tuning is complementary
   to CIFAR-100's visual concepts, not destructive. The forgetting problem may
   require a more distant OOD task (e.g., satellite imagery, medical, or truly
   out-of-domain classification) to manifest.

2. **WiSE-FT is free lunch.** Weight interpolation improved CIFAR-100 by +4.1 pts
   over baseline while only giving up 3.4 pts of CC3M recall. It beats both
   the pretrained and fine-tuned models on OOD — creating performance that
   neither model had alone.

3. **LoRA is the best parameter-efficient method.** 0.6% of params, 96% of
   full FT's CC3M gain, +1.3 pts CIFAR-100. Temperature spike is a useful
   diagnostic for adapter saturation.

4. **Freezing the backbone is counterproductive.** Don't bother with projection-only
   training for contrastive adaptation — the projections can't compensate for a
   frozen encoder.

5. **Temperature is a rich diagnostic.** Stable temp = pretrained calibration
   holds. Rising temp = model gaining confidence (or compensating for limited
   capacity). Falling temp = model struggling (from exp 003-005).

## Next steps
- Fix LP-FT resume bug (skip optimizer state when param count changes), re-run
- Try a more distant OOD benchmark (e.g., EuroSAT, DTD textures) where forgetting
  is more likely to appear
- Test LoRA rank=8 and rank=16 to see if the temperature spike subsides and
  CC3M recall catches up to full FT
- Try WiSE-FT with alpha sweep (0.3, 0.5, 0.7) to find optimal interpolation
- Proceed to exp 009 (Layer-wise LR Decay) per lesson plan
