import torch

from decodeforge.mask import apply_batched_masks, apply_mask


def test_apply_mask_1d():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])

    masked = apply_mask(logits, mask)
    assert masked[0] == 1.0
    assert masked[1] == -float("inf")
    assert masked[2] == 3.0
    assert masked[3] == -float("inf")


def test_apply_mask_2d_batch():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    mask = torch.tensor([False, True, False])

    masked = apply_mask(logits, mask)
    assert masked[0, 1] == 2.0
    assert masked[1, 1] == 5.0
    assert masked[0, 0] == -float("inf")
    assert masked[1, 2] == -float("inf")


def test_custom_fill_value():
    logits = torch.tensor([10.0, 20.0])
    mask = torch.tensor([True, False])

    masked = apply_mask(logits, mask, fill_value=-9999.0)
    assert masked[0] == 10.0
    assert masked[1] == -9999.0


def test_apply_batched_masks():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    mask1 = torch.tensor([True, False, False])
    mask2 = torch.tensor([False, False, True])

    batched = apply_batched_masks(logits, [mask1, mask2])
    assert batched[0, 0] == 1.0
    assert batched[0, 1] == -float("inf")
    assert batched[0, 2] == -float("inf")
    assert batched[1, 0] == -float("inf")
    assert batched[1, 1] == -float("inf")
    assert batched[1, 2] == 6.0
