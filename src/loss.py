"""Contrastive loss functions for CLIP training.

Two implementations:
  - CLIPLoss (InfoNCE): the original CLIP loss — symmetric softmax over
    the full similarity matrix. Each sample competes against all others.
  - SigLIPLoss (Sigmoid): from Zhai et al. 2023 — each image-text pair
    is classified independently as matching/non-matching via sigmoid.
    Outperforms InfoNCE at small batch sizes (<16K).

Both share the same interface: (image_features, text_features) → (loss, metrics).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """Symmetric cross-entropy contrastive loss with learnable temperature.

    Why symmetric? A single image→text CE loss would learn to retrieve text
    given an image, but not vice versa. Averaging both directions forces the
    model to build an embedding space that works for retrieval in either
    direction — which is what makes CLIP useful for zero-shot classification
    (where you go from text prompts → image matching).

    Why learnable temperature? Temperature controls how "peaked" the softmax
    distribution is. Too low → all similarities look the same, no gradient
    signal. Too high → only the hardest negatives matter, training is noisy.
    Letting the model learn it means it can sharpen the distribution as
    the embeddings improve, automatically annealing from exploration to
    exploitation.
    """

    def __init__(self, init_temperature: float = 0.07) -> None:
        super().__init__()
        # We store log(1/temperature) — called "logit_scale" following OpenAI's
        # implementation. Why log-parameterized?
        #   1. Temperature must be positive. exp() guarantees this without
        #      needing constrained optimization.
        #   2. The optimizer sees a smoother landscape in log-space. A step
        #      of 0.1 in log-space is a 10% change regardless of scale,
        #      whereas in linear space it could be negligible or catastrophic.
        #   3. init_temperature=0.07 → logit_scale=ln(1/0.07)≈2.66
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / init_temperature).log()
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute symmetric contrastive loss.

        Args:
            image_features: L2-normalized image embeddings [B, D].
            text_features: L2-normalized text embeddings [B, D].

        Returns:
            (loss, metrics) where metrics contains accuracy and temperature
            for logging without recomputing.
        """
        assert image_features.shape == text_features.shape, (
            f"Shape mismatch: images {image_features.shape} vs text {text_features.shape}"
        )

        # Clamp logit_scale to prevent training instability.
        # exp(4.6) ≈ 100, which is the upper bound from the CLIP paper.
        # Without this clamp, the scale can blow up early in training when
        # cosine similarities are near-zero, causing the model to overfit
        # to noise in the similarity matrix.
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # Cosine similarity matrix: [B, B] where entry (i,j) is the scaled
        # cosine similarity between image_i and text_j.
        # Because features are L2-normalized, @ (matmul) = cosine similarity.
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Labels: the diagonal is the matching pair.
        # In a batch of 256, image_0 matches text_0, image_1 matches text_1, etc.
        # This is why drop_last=True in the DataLoader matters — every batch
        # must have the same structure for these labels to be correct.
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # In-batch accuracy: what fraction of the time does the model rank
        # the correct match as #1? This is more interpretable than loss
        # during training. Random chance = 1/batch_size ≈ 0.4% at B=256.
        with torch.no_grad():
            i2t_acc = (logits_per_image.argmax(dim=1) == labels).float().mean().item()
            t2i_acc = (logits_per_text.argmax(dim=1) == labels).float().mean().item()

        metrics = {
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
            "temperature": (1.0 / logit_scale).item(),
            "logit_scale": logit_scale.item(),
        }

        return (loss_i2t + loss_t2i) / 2.0, metrics


class SigLIPLoss(nn.Module):
    """Sigmoid pairwise contrastive loss (SigLIP).

    Instead of InfoNCE's "which of B texts matches this image?" (global
    softmax ranking), SigLIP asks for each pair independently: "does this
    image match this text? yes/no." (binary sigmoid classification).

    Why this helps at small batch sizes:
      - InfoNCE's softmax denominator sums over B terms. The gradient for
        each pair is entangled with every other pair in the batch. Noisy
        negatives dilute the signal.
      - SigLIP evaluates each pair independently. The gradient for
        (image_0, text_0) doesn't depend on what (image_5, text_73) looks
        like. Each update is more targeted.

    Extra degree of freedom:
      - CLIPLoss has one learnable scalar: logit_scale (temperature⁻¹)
      - SigLIPLoss has two: logit_scale AND logit_bias
      - The bias shifts the sigmoid decision boundary. This lets the model
        separately control "how sharp" (temperature) and "where is the
        threshold between match and non-match" (bias).

    Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training"
    (ICCV 2023).
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        init_bias: float = -10.0,
    ) -> None:
        super().__init__()
        # Same log-parameterized temperature as CLIPLoss
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / init_temperature).log()
        )
        # Bias is initialized negative so that most pairs start as
        # "non-matching" (sigmoid(negative) < 0.5). This is correct —
        # in a batch of B, there are B matches and B²-B non-matches,
        # so the prior should be "probably not a match."
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute sigmoid pairwise contrastive loss.

        Args:
            image_features: L2-normalized image embeddings [B, D].
            text_features: L2-normalized text embeddings [B, D].

        Returns:
            (loss, metrics) — same interface as CLIPLoss.
        """
        assert image_features.shape == text_features.shape

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # [B, B] similarity matrix, same as CLIPLoss
        logits = logit_scale * image_features @ text_features.t() + self.logit_bias

        # Target matrix: +1 on diagonal (match), -1 off-diagonal (non-match)
        batch_size = image_features.shape[0]
        labels = 2 * torch.eye(batch_size, device=image_features.device) - 1

        # Sigmoid loss: -log(sigmoid(label * logit))
        # For matches (label=+1): pushes logit positive → sigmoid→1
        # For non-matches (label=-1): pushes logit negative → sigmoid→1
        # This is equivalent to F.binary_cross_entropy_with_logits
        # with targets=(labels+1)/2, but the formulation below is
        # numerically stable and matches the SigLIP paper.
        loss = -F.logsigmoid(labels * logits).mean()

        # In-batch accuracy for comparability with CLIPLoss
        with torch.no_grad():
            # For each image, which text has highest similarity?
            i2t_acc = (logits.argmax(dim=1) == torch.arange(batch_size, device=logits.device)).float().mean().item()
            t2i_acc = (logits.argmax(dim=0) == torch.arange(batch_size, device=logits.device)).float().mean().item()

        metrics = {
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
            "temperature": (1.0 / logit_scale).item(),
            "logit_scale": logit_scale.item(),
            "logit_bias": self.logit_bias.item(),
        }

        return loss, metrics
