"""Logit masking operations for constrained decoding."""

from __future__ import annotations

import torch


def apply_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = -float("inf"),
) -> torch.Tensor:
    """Apply a boolean valid-token mask to raw vocabulary logits.

    Args:
        logits: Tensor of shape `(vocab_size,)` or `(batch_size, vocab_size)`.
        mask: Boolean tensor where True indicates a valid next token.
              Shape may be `(vocab_size,)` or broadcastable to logits shape.
        fill_value: Value used to replace invalid logit positions (default -inf).

    Returns:
        New tensor with invalid positions set to fill_value.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    elif mask.device != logits.device:
        mask = mask.to(logits.device)

    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Invert mask: True where invalid (to be filled)
    invalid_mask = ~mask
    return logits.masked_fill(invalid_mask, fill_value)


def apply_batched_masks(
    logits: torch.Tensor,
    masks: list[torch.Tensor],
    fill_value: float = -float("inf"),
) -> torch.Tensor:
    """Apply a distinct per-sequence valid token mask to batched logits.

    Args:
        logits: Tensor of shape `(batch_size, vocab_size)`.
        masks: List of boolean 1D tensors of length `batch_size`.
        fill_value: Value used to replace invalid logit positions.

    Returns:
        New tensor of shape `(batch_size, vocab_size)` with masked positions filled.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits tensor, got shape {logits.shape}")
    if len(masks) != logits.shape[0]:
        raise ValueError(
            f"Batch dimension mismatch: logits has {logits.shape[0]} items, "
            f"masks list has {len(masks)}"
        )

    stacked_mask = torch.stack(
        [m.to(device=logits.device, dtype=torch.bool) for m in masks]
    )
    return logits.masked_fill(~stacked_mask, fill_value)
