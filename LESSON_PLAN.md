# CLIP Learning Plan — Phase 2

Phase 1 (experiments 001–003) established from-scratch training fundamentals:
batch size effects, LR sensitivity, contrastive loss mechanics, and the
temperature diagnostic. Phase 2 shifts to efficiency techniques, fine-tuning
dynamics, and data-centric methods.

Reference article: pasted into conversation 2026-03-22.

---

## Key Finding from Phase 2 Tier 1

**The recall ceiling (~0.26-0.28 R@1) on 1M CC3M pairs is a data/model
limit, not an optimization problem.** Confirmed across three approaches:

| Experiment | Approach | recall@1 | Verdict |
|---|---|---|---|
| 003 | InfoNCE, B=512, lr=3e-4 | **0.276** | Best result |
| 004 | Grad accum eff B=2048 | 0.200 | Worse — more overfitting |
| 005 | SigLIP sigmoid loss | 0.259 | Same ceiling |

Loss function, gradient quality, and effective batch size are NOT the
bottleneck. To break through, we need either more/better data or a
pretrained starting point. **Tier 2 (fine-tuning) is now the priority.**

---

## Priority Tiers

### Tier 1 — From-scratch optimization (COMPLETED)

**Experiment 004: Gradient Accumulation** ✅ Done
- Result: recall@1 DROPPED to 0.200 — grad accum smooths gradients but
  doesn't add negatives. 4x more data passes = 4x more memorization.
- Lesson: grad accum ≠ true larger batch for contrastive learning.

**Experiment 005: SigLIP Sigmoid Loss** ✅ Done
- Result: recall@1=0.259, within noise of InfoNCE's 0.276.
- Lesson: at B=512 / 1M pairs, loss function is not the bottleneck.

**Experiment 006: CLIPA Token Reduction (3–4× speedup)** — optional
- Still valuable as a speedup experiment if we want faster iteration
- Train at reduced resolution (112×112) with fewer text tokens (8–16)
- Key question: can we match exp 003 quality in 40 min instead of 160 min?
- Lower priority now that we know the from-scratch ceiling

### Tier 2 — Fine-tuning dynamics

**Experiment 007: Fine-tune Pretrained CLIP (LR exploration)** ✅ Done
- Pretrained zero-shot: R@1=0.626 (2.3x best from-scratch)
- Fine-tuned at lr=1e-5: R@1=0.772 (2.8x from-scratch) in only 5K steps / 40 min
- lr=1e-5 beat lr=5e-6 — contrary to literature, possibly because CC3M is
  close to pretraining distribution or 5K steps too short for forgetting
- No catastrophic forgetting detected on CC3M eval, but need OOD measurement
- Temperature barely moved (14.3→14.6) — pretrained calibration already good
- Lesson: pretrained models exist in a different universe. 400M pairs of
  pretraining knowledge dwarfs anything achievable from scratch on 1M.

**Experiment 008: Adaptation Methods & Catastrophic Forgetting** — DONE
- Measured OOD forgetting via CIFAR-100 zero-shot classification (baseline: 62.3%)
- Compared: full FT, frozen backbone, LoRA rank=4, WiSE-FT (α=0.5), LP-FT
- Results: NO catastrophic forgetting on CIFAR-100 — all methods preserved or
  improved baseline accuracy. CC3M is close enough to CIFAR-100's visual concepts
  that fine-tuning is complementary, not destructive.
- WiSE-FT is the Pareto winner: CIFAR-100 66.3% (+4.1 pts), CC3M R@1=0.734
- LoRA is the efficiency winner: 0.6% of params, 96% of full FT's CC3M recall
- Frozen backbone was counterproductive: barely moved CC3M, hurt CIFAR-100 by 2.7 pts
- LoRA temperature spiked to 17.3 (vs 14.6 for full FT) — adapter capacity signal
- LP-FT crashed on resume (optimizer state mismatch), now fixed, needs re-run
- Lesson: forgetting depends on distributional distance. Need more distant OOD
  benchmarks (EuroSAT, DTD) to see real forgetting. Weight interpolation is free lunch.

**Experiment 009: Adopt CLIP Benchmark for Multi-Dataset Evaluation** — HIGH PRIORITY
- Our exp 008 forgetting evaluation used only CIFAR-100 (too close to CC3M)
- Adopt LAION's `clip-benchmark` for standardized eval across 38+ tasks:
  zero-shot classification (ImageNet, EuroSAT, DTD, FGVC-Aircraft, etc.),
  retrieval (Flickr30k, COCO), and distribution shift (ImageNet-V2/R/Sketch)
- Re-evaluate exp 008 checkpoints against the full benchmark suite
- This retroactively strengthens (or revises) all our forgetting claims
- Key question: does full FT degrade on *shift* tasks even if CIFAR-100 is fine?
- References: WiSE-FT evaluates shift robustness; DataComp uses 38-task suite

**Experiment 010: Layer-wise LR Decay (LLRD)**
- Assign progressively lower LR to earlier transformer layers
- Decay factor 0.60, base LR 6e-4
- Bottom layer gets ~2e-6, top layer gets full 6e-4
- FT-CLIP showed this pushes ViT-B/16 from 84.3% → 85.3% on ImageNet
- Key insight: early layers encode generic features, protect them

### Tier 3 — Data-centric methods (break the regime ceiling)

Peer review confirmed our "data ceiling" is regime-specific, not intrinsic.
Modern CLIP research treats data curation as a first-class intervention.
These experiments test whether better data (not more optimizer tricks) moves
the ceiling within our compute budget.

**Experiment 011: Caption Enrichment (BLIP/VeCLIP-style)**
- Pre-generate improved captions using a captioner or LLM rewriting
- BLIP bootstraps/cleans captions using a captioner+filter loop
- VeCLIP rewrites captions to be visually enriched (more descriptive)
- LaCLIP (NeurIPS 2023) showed gains from text diversity via LLM paraphrase
- CC3M captions are notoriously noisy and repetitive — this directly attacks
  the supervision quality bottleneck
- Key question: how much of the "data ceiling" is really a *caption quality* ceiling?
- Practical approach: use a local LLM to generate 2–5 rewrites per caption,
  randomly sample during training

**Experiment 012: Data Filtering with DFN**
- Use Apple's DFN filter model (`apple/DFN-public`, ViT-B/32)
- Score our 1M CC3M pairs by quality
- Train on top-50%, top-25%, top-10% and compare
- Counter-intuitive hypothesis: LESS data but higher quality may beat
  more data. MedCLIP showed 20K good pairs beat 200K noisy ones.
- Key insight: "what you train on matters more than architecture"
- DataComp institutionalized this: fixed model/code, compete on data curation

**Experiment 013: Scale to CC12M**
- CC12M provides ~12M image-text pairs (12x our current 1M CC3M subset)
- Explicitly introduced as a relaxed, larger pretraining dataset vs CC3M
- Tests whether the regime ceiling moves with data quantity alone
- Compare: same recipe (exp 003 settings) on 1M vs 5M vs 12M subsets
- Key question: is the ceiling quantity-limited or quality-limited?

**Experiment 014: Hard Negative Mining / Cross-Batch Memory**
- Pre-compute CLIP embeddings for all 1M pairs (~30 min on 4090)
- Build nearest-neighbor index with FAISS
- Construct batches with explicit hard negatives vs random batching
- Also explore Cross-Batch Memory (XBM): store embeddings from previous
  batches to mine harder negatives across iterations
- MoCo-style momentum queue is another option for increasing effective
  negatives without more VRAM
- Key question: can we break the single-GPU batch-size cap on negatives?

### Tier 4 — Advanced techniques (deeper understanding)

**Experiment 015: CLIP Distillation**
- Distill from ViT-L/14 teacher into ViT-B/32 student
- open_clip built-in: `--distill-model ViT-L-14 --distill-pretrained openai`
- Compare: student from scratch vs distilled vs fine-tuned from pretrained
- TinyCLIP: affinity mimicking + weight inheritance
- CLIP-KD: simple MSE on normalized features works surprisingly well

**Experiment 016: Prompt Tuning (CoOp → CoCoOp → MaPLe)**
- Freeze entire CLIP, learn only continuous prompt vectors
- CoOp: 16 context vectors, few thousand trainable params
- CoCoOp: input-conditional prompts, better novel class generalization
- MaPLe: prompt both vision and language at multiple layers
- All use <1GB memory, train in minutes
- Key question: where does adaptation capacity reside — weights or prompts?

---

## Memory Budget Reference (RTX 4090, 24GB)

| Model | Full FT batch (AMP) | + grad ckpt | LoRA (r=16) | Status |
|---|---|---|---|---|
| ViT-B/32 (151M) | 256–384 | 384–512 | 512+ | Workhorse |
| ViT-B/16 (150M) | 128–192 | 192–256 | 256–512 | Comfortable |
| ViT-L/14 (428M) | 8–16 | 32–64 | 128–256 | LoRA preferred |
| ViT-H/14 (756M) | OOM | 4–8 | 32–64 | LoRA only |

Always use `--precision amp` (bf16 preferred) and `--grad-checkpointing`.

---

## Key Models & Checkpoints to Know

| Model | What it is | open_clip name |
|---|---|---|
| OpenAI CLIP | Original, 400M WIT pairs | `--pretrained openai` |
| LAION-2B | Open reproduction on 2B pairs | `--pretrained laion2b_s34b_b79k` |
| DFN | Apple, data-filtered training | `apple/DFN-public` (filter model) |
| MetaCLIP | Meta, metadata-curated training | `--pretrained metaclip_400m` |
| SigLIP2 | Google, sigmoid + aux objectives | Available via open_clip/HF |
| PE-Core-bigG | SOTA 85.4% IN zero-shot | Largest open_clip model |

---

## Domain-Specific CLIP Models (Medical Imaging Examples)

| Model | Domain | Dataset | Base | Reproducible on 4090? |
|---|---|---|---|---|
| CheXzero | Chest X-ray | 370K pairs | ViT-B/32 | Yes |
| PLIP | Pathology | 208K pairs | ViT-B/32 | Yes |
| Quilt-1M | Histology | 1M pairs | ViT-B/32 | Yes |
| MedCLIP | General medical | ~200K pairs | Swin-T | Yes |
| PubMedCLIP | Radiology | 80K pairs | ViT-B/32 | Yes (but underperforms — forgetting) |
| BiomedCLIP | Biomedical | 15M pairs | Custom | No — needs cluster |

MedCLIP is notable: 20K well-engineered pairs beat 200K noisy ones by
decoupling images/texts and using medical knowledge for soft labels.
PubMedCLIP is a cautionary tale: small domain fine-tuning can cause
catastrophic forgetting that makes the model WORSE than vanilla CLIP.
