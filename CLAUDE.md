# CLAUDE.md — CLIP Training / MLE Learning Project

## Who I Am

I'm an AI Architect with a background in ML infrastructure, model serving, and platform
architecture. This project is a self-directed MLE learning program — I'm here to build
deep intuition for training dynamics, not just ship working code. Prioritize explanatory
comments and observable behavior over cleverness or premature optimization.

I work forensically: I want to understand *why* things behave the way they do, not just
that they work. Treat analysis tasks like incident investigations — find the story in the
data.

---

## Environment

- **OS:** WSL2 (Ubuntu 24.04) on Windows 11
- **GPU:** NVIDIA RTX 4090 (24GB VRAM, Ada Lovelace, CUDA 12.x)
- **Python:** 3.12 via `uv`
- **Key libs:** PyTorch (CUDA 12.x wheels), `open_clip_torch`, `tensorboard`, `umap-learn`
- **Package manager:** `uv` — always use `uv run` or `uv pip install`, never bare `pip`
- **Shell:** Bash inside WSL2

Before running any training or GPU code, verify the environment is sane:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.memory_allocated())"
```

---

## Project Structure

```
clip-training/
├── CLAUDE.md               ← you are here
├── data/
│   ├── cc3m/               ← CC3M TSV files + downloaded images
│   └── eval/               ← held-out evaluation set (100–1000 pairs)
├── src/
│   ├── dataset.py          ← CC3M Dataset + DataLoader
│   ├── model.py            ← CLIP model init (open_clip wrapper)
│   ├── train.py            ← main training loop
│   ├── loss.py             ← contrastive loss, temperature param
│   └── eval.py             ← zero-shot eval, recall@K, UMAP viz
├── scripts/
│   ├── smoke_test.py       ← GPU + data pipeline sanity check
│   └── postmortem.py       ← training run analysis / forensics
├── checkpoints/            ← saved model states (every N steps)
├── runs/                   ← TensorBoard event files
└── pyproject.toml          ← uv-managed dependencies
```

---

## Training Configuration (Defaults)

| Parameter         | Value          | Notes                                      |
|-------------------|----------------|--------------------------------------------|
| Model             | ViT-B/32       | From scratch unless stated otherwise       |
| Batch size        | 256            | Image-text pairs                           |
| Precision         | BF16           | Mixed precision via `torch.amp`            |
| Temperature       | Learnable      | Log-parameterized, initialized at 0.07     |
| Optimizer         | AdamW          | lr=1e-3, weight_decay=0.1                  |
| Scheduler         | Cosine         | With warmup (first 10% of steps)           |
| Checkpoint freq   | Every 1000 steps |                                           |
| DataLoader workers| 8              | Tune down if WSL2 shared memory errors     |

---

## Code Style Expectations

- **Comments explain WHY, not WHAT.** I already know what `loss.backward()` does.
  Tell me why contrastive loss needs the temperature clamp, or why we zero gradients
  before the forward pass vs after.
- **Readable over clever.** One concept per function. No magic numbers without a name.
- **Fail loudly.** Assertions on tensor shapes at key boundaries. I'd rather crash
  early with a clear message than silently train on garbage.
- **Type hints on all function signatures.**
- **No silent GPU fallback to CPU.** If CUDA isn't available, raise an error.

---

## Observability Standards

Every training run should produce:
- TensorBoard logs: loss (train), recall@1 (eval), temperature value, grad norms
- Console output: step, loss, GPU memory used, time-per-step
- Checkpoints: model weights + optimizer state + step number + config snapshot

The `postmortem.py` script should be runnable after any training session to generate
a structured analysis report.

---

## Evaluation Protocol

**Zero-shot recall** (primary metric during learning):
- recall@1 and recall@5 on held-out image→text and text→image retrieval
- Evaluated every 500 steps on `data/eval/`

**Embedding space checks** (periodic, not every run):
- UMAP projection of 1000 image + text embeddings
- Cosine similarity distribution of matched vs. unmatched pairs
- These are diagnostic, not optimization targets

---

---

## What "Done" Looks Like for a Task

A task is complete when:
1. The code runs without error in this environment (WSL2, uv, CUDA 12.x)
2. Output is verified — show me actual terminal output, not just "this should work"
3. I understand what happened — key decisions are explained in comments or summary
4. The next logical step is obvious from the state of the repo

Don't hand off broken scaffolding. If something can't be verified, say so explicitly
and explain what's blocking it.
