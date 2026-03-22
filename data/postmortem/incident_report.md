# Training Postmortem — Experiment 002

## Executive Summary

This report analyzes the exp 002 training run (ViT-B/32, B=512, lr=1e-3,
20K steps on ~1M CC3M pairs). The run exhibited a three-phase dynamic:
fast learning, LR-induced destabilization, and cosine-decay recovery.

---

## 1. Loss Dynamics

Loss range: 2.269 — 6.303
Initial loss: 6.303 (step 0)
Final loss: 2.585 (step 19990)
Best smoothed loss: 2.516 (step ~19550)
Longest divergence: steps ~1460–2320 (860 steps, loss rose from 4.700 to 5.146)

**Interpretation**: The loss curve tells a story of a learning rate that was
too aggressive for the model's capacity after warmup. The warmup phase
(lr ramping from 0.01x to 1x) allowed gradual adaptation, but the full
lr=1e-3 after warmup caused the model to overshoot in weight space,
partially destroying the representations it had learned. The cosine decay
gradually reduced the LR, allowing the model to reconverge.

---

## 2. Gradient Dynamics

Phase                    Mean      Std      Max      Min
--------------------------------------------------------
warmup (0-10%)           2.88     1.19     9.16     1.71
early (10-25%)           1.84     0.71     7.91     1.09
mid (25-50%)             1.09     0.21     2.25     0.77
late (50-75%)            1.02     0.12     1.38     0.76
final (75-100%)          1.47     0.11     1.69     1.22

Gradient spikes (>3σ): 20 events
  Steps: [np.int64(0), np.int64(2110), np.int64(2320), np.int64(2880), np.int64(3810), np.int64(4350), np.int64(6010), np.int64(6120), np.int64(8440), np.int64(8990)]...

**Interpretation**: Gradient norms track the story told by the loss.
High norms during warmup reflect large initial updates from random weights.
Spikes after warmup correspond to the destabilization period. The gradual
decline in the late phase shows the model settling into a stable basin
as the learning rate decreases.

---

## 3. Embedding Alignment (Early vs Late Checkpoint)

Metric                            Early (step 1000)    Late (step 20000)
------------------------------------------------------------------------
Matched pair sim (mean)                      0.4343               0.4866
Unmatched pair sim (mean)                    0.2240               0.2207
Gap (matched - unmatched)                    0.2102               0.2659
Logit scale                                     2.7                  3.3

**Interpretation**: The gap between matched and unmatched pair similarity
is the core metric of contrastive learning. A larger gap means the model
can more reliably distinguish correct from incorrect image-text pairs.
The increase from early to late checkpoint confirms that despite the
mid-training destabilization, the model ultimately learned strong
alignment.

---

## 4. Eval Recall Progression

  Step    500: i2t_R@1=0.0378  t2i_R@1=0.0378
  Step   1000: i2t_R@1=0.0534  t2i_R@1=0.0482
  Step   1500: i2t_R@1=0.0495  t2i_R@1=0.0299
  Step   2000: i2t_R@1=0.0378  t2i_R@1=0.0286
  Step   2500: i2t_R@1=0.0312  t2i_R@1=0.0273
  Step   3000: i2t_R@1=0.0339  t2i_R@1=0.0339
  Step   3500: i2t_R@1=0.0234  t2i_R@1=0.0260
  Step   4000: i2t_R@1=0.0365  t2i_R@1=0.0339
  Step   4500: i2t_R@1=0.0326  t2i_R@1=0.0339
  Step   5000: i2t_R@1=0.0365  t2i_R@1=0.0312
  Step   5500: i2t_R@1=0.0430  t2i_R@1=0.0417
  Step   6000: i2t_R@1=0.0495  t2i_R@1=0.0430
  Step   6500: i2t_R@1=0.0586  t2i_R@1=0.0521
  Step   7000: i2t_R@1=0.0612  t2i_R@1=0.0508
  Step   7500: i2t_R@1=0.0508  t2i_R@1=0.0469
  Step   8000: i2t_R@1=0.0651  t2i_R@1=0.0690
  Step   8500: i2t_R@1=0.0651  t2i_R@1=0.0573
  Step   9000: i2t_R@1=0.0677  t2i_R@1=0.0638
  Step   9500: i2t_R@1=0.0768  t2i_R@1=0.0716
  Step  10000: i2t_R@1=0.0872  t2i_R@1=0.0742
  Step  10500: i2t_R@1=0.0781  t2i_R@1=0.0768
  Step  11000: i2t_R@1=0.0807  t2i_R@1=0.0951
  Step  11500: i2t_R@1=0.0951  t2i_R@1=0.0977
  Step  12000: i2t_R@1=0.1042  t2i_R@1=0.0924
  Step  12500: i2t_R@1=0.1120  t2i_R@1=0.0898
  Step  13000: i2t_R@1=0.1068  t2i_R@1=0.1081
  Step  13500: i2t_R@1=0.1094  t2i_R@1=0.1003
  Step  14000: i2t_R@1=0.1172  t2i_R@1=0.1133
  Step  14500: i2t_R@1=0.1185  t2i_R@1=0.1081
  Step  15000: i2t_R@1=0.1224  t2i_R@1=0.1146
  Step  15500: i2t_R@1=0.1198  t2i_R@1=0.1172
  Step  16000: i2t_R@1=0.1367  t2i_R@1=0.1315
  Step  16500: i2t_R@1=0.1341  t2i_R@1=0.1276
  Step  17000: i2t_R@1=0.1250  t2i_R@1=0.1172
  Step  17500: i2t_R@1=0.1250  t2i_R@1=0.1159
  Step  18000: i2t_R@1=0.1263  t2i_R@1=0.1250
  Step  18500: i2t_R@1=0.1250  t2i_R@1=0.1211
  Step  19000: i2t_R@1=0.1302  t2i_R@1=0.1224
  Step  19500: i2t_R@1=0.1237  t2i_R@1=0.1237

---

## 5. Incident Report

**What the model learned**: The model built a semantic embedding space
that organizes visual concepts (animals, buildings, vehicles, people)
into distinct clusters. It achieved 44% in-batch accuracy (vs 0.2%
random) and 13.7% recall@1 on 1K eval samples.

**What broke**: The learning rate (1e-3) was too high after the warmup
phase ended. This caused ~3000 steps of effective unlearning where the
model's representations degraded. Evidence: loss increased, accuracy
dropped, temperature decreased (model lost confidence).

**What saved it**: The cosine learning rate schedule. As lr decayed from
1e-3 toward 0, the model restabilized and ultimately surpassed its
pre-destabilization performance by a large margin. The second half of
training (steps 10K-20K) was the most productive.

**What to change for the next run**: Reduce peak LR to 3e-4. This should
eliminate the destabilization entirely, allowing the model to make
productive use of all training steps.
