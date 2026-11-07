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
