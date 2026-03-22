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

### Tier 2 — Fine-tuning dynamics (NEXT — the conceptual leap)

**Experiment 007: Fine-tune Pretrained CLIP (LR exploration)**
- Load pretrained ViT-B/32 (`--pretrained laion2b_s34b_b79k`)
- Fine-tune on our CC3M data at lr=1e-6, 5e-6, 1e-5, 5e-5
- Compare against our from-scratch exp 003 checkpoint
- Key insight: fine-tuning needs ~100× lower LR than from-scratch
- Watch for catastrophic forgetting — eval on held-out AND on zero-shot tasks

**Experiment 008: LP-FT vs WiSE-FT vs LoRA**
- Controlled comparison on a downstream task (CIFAR-100 or domain-specific):
  1. Linear probe: freeze CLIP, train linear head only
  2. Full fine-tune: lr=1e-5, train everything
  3. LP-FT: linear probe 50 epochs → fine-tune everything 10 epochs
  4. WiSE-FT: after full FT, interpolate weights with original (α=0.5)
  5. LoRA: rank-4 on all attention layers (~147K params vs 87M)
- Directly demonstrates catastrophic forgetting and its solutions
- Key question: where does adaptation capacity reside in CLIP?

**Experiment 009: Layer-wise LR Decay (LLRD)**
- Assign progressively lower LR to earlier transformer layers
- Decay factor 0.60, base LR 6e-4
- Bottom layer gets ~2e-6, top layer gets full 6e-4
- FT-CLIP showed this pushes ViT-B/16 from 84.3% → 85.3% on ImageNet
- Key insight: early layers encode generic features, protect them

### Tier 3 — Data-centric methods (make the most of 1M pairs)

We can't download more data, so these experiments focus on extracting
more value from what we have.

**Experiment 010: Hard Negative Mining with FAISS**
- Pre-compute CLIP embeddings for all 1M pairs (~30 min on 4090)
- Build nearest-neighbor index
- Construct batches with explicit hard negatives vs random batching
- Compare at same effective batch size
- Key question: does hard negative mining break the generalization plateau?

**Experiment 011: LLM Caption Augmentation (LaCLIP)**
- Pre-generate 2–5 caption paraphrases per sample using a local LLM
- Randomly sample during training (text varies across epochs)
- LaCLIP (NeurIPS 2023) showed significant improvement from text diversity
- Note: strong image augmentations typically DON'T help CLIP
- Key question: is our recall plateau a text diversity problem?
  CC3M captions are noisy and repetitive — LLM rewrites could help.

**Experiment 012: Data Filtering with DFN**
- Use Apple's DFN filter model (`apple/DFN-public`, ViT-B/32)
- Score our 1M CC3M pairs by quality
- Train on top-50%, top-25%, top-10% and compare
- Counter-intuitive hypothesis: LESS data but higher quality may beat
  more data. MedCLIP showed 20K good pairs beat 200K noisy ones.
- Key insight: "what you train on matters more than architecture"

### Tier 4 — Advanced techniques (deeper understanding)

**Experiment 013: CLIP Distillation**
- Distill from ViT-L/14 teacher into ViT-B/32 student
- open_clip built-in: `--distill-model ViT-L-14 --distill-pretrained openai`
- Compare: student from scratch vs distilled vs fine-tuned from pretrained
- TinyCLIP: affinity mimicking + weight inheritance
- CLIP-KD: simple MSE on normalized features works surprisingly well

**Experiment 014: Prompt Tuning (CoOp → CoCoOp → MaPLe)**
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
