# Experiment Log

Phase 1 (001–003): From-scratch training fundamentals. See `REFERENCE.md`.
Phase 2 (004+): Efficiency, fine-tuning, data-centric methods. See `LESSON_PLAN.md`.

Each experiment gets a numbered directory (`001_`, `002_`, ...) with a
standardized `notes.md` documenting the full arc from hypothesis to conclusions.

## Template

Every `notes.md` follows this structure:

```markdown
# Experiment NNN — Short Title

## Hypothesis
What do we expect to happen and why?

## Method
- Config changes (lr, batch size, steps, etc.)
- What's different from the previous run?
- Command used to launch

## Observations
- Loss curve behavior
- Accuracy progression
- Temperature dynamics
- GPU memory / throughput
- Anything unexpected

## Results
| Metric         | Value  |
|----------------|--------|
| Final loss     |        |
| i2t_recall@1   |        |
| t2i_recall@1   |        |
| i2t_recall@5   |        |
| t2i_recall@5   |        |
| Steps trained  |        |
| Wall time      |        |

## Conclusions
- What did we learn?
- What should we change for the next experiment?
```

## Experiment Index

| #   | Name | Status | Key Finding |
|-----|------|--------|-------------|
| 001 | Baseline — first training run | **done** | Loss 5.65→4.15, 10% in-batch acc, but recall still near random. Only 2.6GB of 24GB VRAM used — room to scale batch size. |
| 002 | Large batch B=512, 20K steps | **done** | Loss 6.30→2.55, 44% acc, recall@1=0.137. LR destabilized mid-training but cosine decay recovered. Temperature increased (model gained confidence). |
| 003 | Lower LR 3e-4, B=512, 20K steps | **done** | Loss 6.30→0.30, 90% acc, recall@1=0.276. No destabilization. 2x better than exp 002. Overfitting emerging — recall plateaued while train loss kept dropping. |
| 004 | Grad accum eff B=2048, 20K opt steps | **done** | Loss 0.09, 97% acc — but recall@1 DROPPED to 0.200. Grad accum ≠ true larger batch for contrastive learning. More data passes = more memorization, not better generalization. |
| 005 | SigLIP sigmoid loss, B=512, 20K steps | **done** | SigLIP ≈ InfoNCE at this scale. recall@1=0.259 vs 0.276 — within noise. Confirms the recall ceiling is a data/model limit, not a loss function artifact. |
