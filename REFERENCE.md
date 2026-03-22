## Prompts for Your MLE Learning Program

Here's a set of prompts staged across the learning arc — from environment bootstrap through actual CLIP training. Designed to leverage Claude Code's agentic strengths (file editing, running commands, reading your codebase).

---

### 🏗️ Stage 1 — Environment Bootstrap

**Prompt 1: WSL2 ML Environment Setup**
```
I'm setting up a fresh ML training environment on WSL2 (Ubuntu 24.04) 
with an RTX 4090. Help me:
1. Verify CUDA is accessible (nvidia-smi, torch.cuda check)
2. Set up a uv-managed Python 3.11 environment
3. Install PyTorch with CUDA 12.x wheels
4. Install open_clip_torch and its dependencies
5. Write a smoke test script that confirms GPU training is functional

Create all config files and scripts needed. Run the smoke test at the end
and show me the output.
```

---

### 📦 Stage 2 — Dataset & DataLoader

**Prompt 2: CC3M DataLoader**
```
I want to train CLIP on the Conceptual Captions 3M dataset (CC3M).
Help me:
1. Write a PyTorch Dataset class for CC3M (TSV format with url + caption)
2. Add async image downloading with failure tolerance 
3. Add a CLIP-appropriate preprocessing pipeline (224px, normalize)
4. Write a DataLoader setup with proper num_workers for my 4090
5. Add a quick sanity check that shows sample images + captions

Use open_clip's tokenizer for captions. Target batch size of 256.
```

---

### 🔬 Stage 3 — Training Loop

**Prompt 3: CLIP Training Loop from Scratch**
```
I have a CC3M DataLoader ready. Now build me a minimal but correct 
CLIP training loop using open_clip. I want to understand the mechanics,
so keep it readable over optimized. Include:

1. Model init (ViT-B/32 from scratch, not pretrained)
2. Contrastive loss implementation with temperature parameter
3. Training loop with gradient scaler for mixed precision (BF16)
4. Basic logging: loss, image-text accuracy, GPU memory usage
5. Checkpoint saving every N steps

Add comments explaining WHY each step exists — I'm learning, not shipping.
```

---

### 📊 Stage 4 — Monitoring & Debugging

**Prompt 4: Training Dashboard**
```
My CLIP training loop is running. Add observability:
1. TensorBoard logging for loss curves and the temperature parameter
2. Periodic zero-shot evaluation on a small held-out set 
   (pick 100 image-caption pairs, measure recall@1 and recall@5)
3. A script that visualizes the embedding space using UMAP on 
   1000 random samples — plot image embeddings colored by a rough 
   semantic category if detectable from captions

Run the UMAP visualization and show me the output.
```

---

### 🔍 Stage 5 — Forensic / Analytical (plays to your Code Scene interests)

**Prompt 5: Training Postmortem Analysis**
```
Analyze my CLIP training run. I have TensorBoard logs at ./runs/ 
and checkpoints at ./checkpoints/.

I want you to:
1. Parse the tfevents files and find where loss plateaued or diverged
2. Check gradient norms across layers — identify any that vanished or exploded
3. Compare embedding alignment (cosine sim of matched pairs) at 
   checkpoint 1 vs latest
4. Write a short "incident report" style summary: what the model learned, 
   what broke, what I should change for the next run

Treat this like a code forensics exercise — find the story in the numbers.
```

---

### 🚀 Bonus — Zero-Shot Eval on Your Own Data

**Prompt 6: Domain-Transfer Zero-Shot Test**
```
I want to test my trained CLIP model on a domain-transfer challenge.
Find or generate 20 sample images from each of these categories:
[dog, cat, car, building, food].

Set up a zero-shot classification experiment:
- Text prompts: "a photo of a [category]" style
- Measure top-1 accuracy with my model vs openai/clip-vit-base-patch32
- Plot a confusion matrix for both
- Write a 1-paragraph interpretation: where does my from-scratch
  model fail that the pretrained one doesn't, and why?