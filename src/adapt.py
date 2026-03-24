"""Adaptation methods for pretrained CLIP — exp 008.

Three strategies for adapting a pretrained model while controlling forgetting:

1. freeze_backbone: Only train projection layers (contrastive "linear probe")
2. apply_lora: Inject low-rank adapters into attention layers (Hu et al., 2021)
3. wise_ft_interpolate: Interpolate fine-tuned weights with pretrained (Wortsman et al., 2022)

The key insight: full fine-tuning is a blunt instrument. These methods trade off
in-distribution performance against preservation of pretrained knowledge.
"""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def freeze_backbone(model: nn.Module) -> dict[str, int]:
    """Freeze everything except final projection layers.

    In CLIP, the projection layers map encoder outputs into the shared
    embedding space. By only training these, we adapt the "alignment"
    between modalities without disturbing the learned representations.

    For ViT-B-32: visual.proj [768, 512] + text_projection [512, 512]
    = 655K trainable params vs ~87M total.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze projection layers — these are the "last mile" mapping
    # from encoder space to shared embedding space
    trainable_names = []
    for name, param in model.named_parameters():
        if name in ("visual.proj", "text_projection"):
            param.requires_grad = True
            trainable_names.append(f"{name} {list(param.shape)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    logger.info(f"Freeze backbone: {trainable:,} trainable, {frozen:,} frozen")
    for name in trainable_names:
        logger.info(f"  trainable: {name}")

    return {"trainable": trainable, "frozen": frozen}


def unfreeze_all(model: nn.Module) -> dict[str, int]:
    """Unfreeze all model parameters (for LP-FT phase 2)."""
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Unfroze all: {total:,} trainable params")
    return {"trainable": total, "frozen": 0}


class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with LoRA on Q and V.

    LoRA (Low-Rank Adaptation) adds a small trainable "delta" to the frozen
    attention projections: W' = W + B @ A, where A is [rank, d] and B is [d, rank].

    Why Q and V but not K? Hu et al. (2021) found that adapting Q and V gives
    nearly the same performance as adapting all projections, with fewer params.
    K carries positional/structural info that transfers well without modification.

    The scaling factor (1/rank) prevents the LoRA delta from dominating the
    original weights at initialization. Combined with zero-init on B, the model
    starts exactly where pretrained left off.
    """

    def __init__(self, original: nn.MultiheadAttention, rank: int = 4):
        super().__init__()
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = 1.0 / rank

        # Keep original weights frozen — we reference them directly
        self.in_proj_weight = original.in_proj_weight
        self.in_proj_bias = original.in_proj_bias
        self.out_proj = original.out_proj

        for p in [self.in_proj_weight, self.in_proj_bias]:
            p.requires_grad = False
        for p in self.out_proj.parameters():
            p.requires_grad = False

        # LoRA adapters for Q and V
        # A: random init (kaiming), B: zero init → LoRA delta starts at zero
        # so the model begins exactly at pretrained quality.
        # Create on same device as original weights to avoid device mismatch.
        device = original.in_proj_weight.device
        dtype = original.in_proj_weight.dtype

        self.lora_q_A = nn.Parameter(torch.empty(rank, self.embed_dim, device=device, dtype=dtype))
        self.lora_q_B = nn.Parameter(torch.zeros(self.embed_dim, rank, device=device, dtype=dtype))
        self.lora_v_A = nn.Parameter(torch.empty(rank, self.embed_dim, device=device, dtype=dtype))
        self.lora_v_B = nn.Parameter(torch.zeros(self.embed_dim, rank, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_q_A)
        nn.init.kaiming_uniform_(self.lora_v_A)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Multi-head attention with LoRA deltas on Q and V projections.

        Instead of using PyTorch's opaque MHA forward, we manually decompose
        the computation. This makes the LoRA injection transparent:
          Q = W_q @ x + (B_q @ A_q @ x) * scaling
          K = W_k @ x
          V = W_v @ x + (B_v @ A_v @ x) * scaling
        """
        d = self.embed_dim

        # Split packed in_proj_weight [3d, d] into Q, K, V portions
        W_q = self.in_proj_weight[:d]
        W_k = self.in_proj_weight[d : 2 * d]
        W_v = self.in_proj_weight[2 * d :]
        b_q = self.in_proj_bias[:d]
        b_k = self.in_proj_bias[d : 2 * d]
        b_v = self.in_proj_bias[2 * d :]

        # Project with LoRA delta on Q and V
        q = F.linear(query, W_q, b_q) + F.linear(query, self.lora_q_B @ self.lora_q_A) * self.scaling
        k = F.linear(key, W_k, b_k)
        v = F.linear(value, W_v, b_v) + F.linear(value, self.lora_v_B @ self.lora_v_A) * self.scaling

        # Reshape: [B, S, D] → [B, H, S, D/H] for multi-head attention
        B, S_q, _ = q.shape
        S_kv = k.shape[1]
        q = q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled_dot_product_attention handles the causal mask from the
        # text transformer — it accepts both float masks (additive, with -inf)
        # and boolean masks
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Reshape back: [B, H, S, D/H] → [B, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, d)

        # Output projection (frozen, no LoRA here)
        attn_output = self.out_proj(attn_output)

        return attn_output, None


def apply_lora(model: nn.Module, rank: int = 4) -> dict[str, int]:
    """Replace all attention layers with LoRA-augmented versions.

    Walks the model tree, finds every nn.MultiheadAttention, and swaps it
    with a LoRAMultiheadAttention that keeps the original weights frozen
    and only trains the low-rank adapters.

    For ViT-B-32 with rank=4:
      - 24 attention layers (12 visual + 12 text)
      - 4 LoRA matrices per layer: q_A, q_B, v_A, v_B
      - Visual: 4 * (4*768 + 768*4) = 24,576 params per layer
      - Text: 4 * (4*512 + 512*4) = 16,384 params per layer
      - Total: 12*24,576 + 12*16,384 = 491,520 trainable params
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in model.named_modules():
        # Find parent modules that contain an 'attn' child
        if hasattr(module, "attn") and isinstance(module.attn, nn.MultiheadAttention):
            lora_attn = LoRAMultiheadAttention(module.attn, rank=rank)
            module.attn = lora_attn
            replaced += 1

    # Also unfreeze projection layers — they're small and help alignment
    for name, param in model.named_parameters():
        if name in ("visual.proj", "text_projection"):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    logger.info(
        f"LoRA rank={rank}: replaced {replaced} attention layers, "
        f"{trainable:,} trainable, {frozen:,} frozen"
    )

    return {"trainable": trainable, "frozen": frozen, "replaced": replaced}


def wise_ft_interpolate(
    model: nn.Module,
    pretrained_state_dict: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> None:
    """WiSE-FT: interpolate fine-tuned weights with pretrained (Wortsman et al., 2022).

    theta_wise = alpha * theta_ft + (1 - alpha) * theta_pretrained

    The intuition: fine-tuning moves weights toward in-distribution performance
    but away from the pretrained "general knowledge" basin. Interpolating pulls
    them back partway, recovering OOD performance at a small ID cost.

    Alpha=0.5 is the default from the paper. Higher alpha = more fine-tuned
    (better ID, worse OOD). Lower alpha = more pretrained (better OOD, worse ID).

    Modifies model in-place.
    """
    ft_state = model.state_dict()
    interpolated = {}

    for key in ft_state:
        if key in pretrained_state_dict:
            interpolated[key] = (
                alpha * ft_state[key] + (1 - alpha) * pretrained_state_dict[key]
            )
        else:
            # Keys that only exist in fine-tuned model (e.g., LoRA params)
            # keep their fine-tuned values
            interpolated[key] = ft_state[key]

    model.load_state_dict(interpolated)
    logger.info(f"WiSE-FT: interpolated with alpha={alpha}")


def save_pretrained_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot the pretrained weights before any fine-tuning.

    Call this right after model creation to capture the reference point
    for WiSE-FT interpolation later.
    """
    return copy.deepcopy(model.state_dict())
