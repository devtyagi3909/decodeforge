import pytest
import torch
from pydantic import BaseModel

from decodeforge.engine import GrammarEngine
from decodeforge.mask import apply_mask


class UserAction(BaseModel):
    action: str
    code: int


def test_engine_from_regex():
    # Simple regex grammar: "yes" or "no"
    vocab = {
        0: "y",
        1: "es",
        2: "n",
        3: "o",
        4: "maybe",
        5: "<eos>",
    }
    engine = GrammarEngine.from_regex("yes|no", vocab=vocab, eos_token_id=5)

    # Initial state: only 'y' (0) or 'n' (2) are allowed
    mask = engine.get_token_mask()
    assert mask[0].item() is True
    assert mask[2].item() is True
    assert mask[1].item() is False  # 'es'
    assert mask[3].item() is False  # 'o'
    assert mask[4].item() is False  # 'maybe'
    assert mask[5].item() is False  # '<eos>' not allowed yet

    # Advance with 'y' (0)
    engine.advance(0)
    mask = engine.get_token_mask()
    assert mask[1].item() is True  # 'es' allowed
    assert mask[0].item() is False
    assert mask[2].item() is False

    # Advance with 'es' (1) -> reached "yes"
    engine.advance(1)
    assert engine.is_terminated() is True
    mask = engine.get_token_mask()
    assert mask[5].item() is True  # '<eos>' now allowed!

    # Advancing with EOS should succeed
    engine.advance(5)


def test_engine_rejection_error():
    vocab = {0: "a", 1: "b"}
    engine = GrammarEngine.from_regex("a", vocab=vocab)
    with pytest.raises(ValueError, match="violates grammar"):
        engine.advance(1)  # 'b' violates grammar


def test_engine_from_pydantic():
    vocab = {
        0: '{"action":',
        1: '"click"',
        2: ',"code":',
        3: "10",
        4: "}",
        5: "invalid",
        6: "<eos>",
    }
    engine = GrammarEngine.from_pydantic(UserAction, vocab=vocab, eos_token_id=6)

    # First token must be '{"action":'
    mask = engine.get_token_mask()
    assert mask[0].item() is True
    assert mask[5].item() is False

    # Advance through json sequence
    engine.advance(0)  # {"action":
    mask = engine.get_token_mask()
    assert mask[1].item() is True

    engine.advance(1)  # "click"
    mask = engine.get_token_mask()
    assert mask[2].item() is True

    engine.advance(2)  # ,"code":
    mask = engine.get_token_mask()
    assert mask[3].item() is True

    engine.advance(3)  # 10
    mask = engine.get_token_mask()
    assert mask[4].item() is True

    engine.advance(4)  # }
    assert engine.is_terminated() is True

    mask = engine.get_token_mask()
    assert mask[6].item() is True  # <eos> allowed


def test_logits_sampling_loop_simulation():
    # Simulate an actual logits argmax decode loop with masking
    vocab = {0: "foo", 1: "bar", 2: "<eos>"}
    engine = GrammarEngine.from_regex("foo", vocab=vocab, eos_token_id=2)

    # Mock model outputs logits where 'bar' (1) has highest unconstrained logit
    raw_logits = torch.tensor([5.0, 10.0, -1.0])
    mask = engine.get_token_mask()
    masked = apply_mask(raw_logits, mask)

    # Argmax on masked logits must pick 'foo' (0) because 'bar' (1) is -inf
    selected = torch.argmax(masked).item()
    assert selected == 0

    engine.advance(selected)
    assert engine.is_terminated() is True
