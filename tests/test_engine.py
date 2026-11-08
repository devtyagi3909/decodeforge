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


def test_engine_state_analytics():
    vocab = {0: "a", 1: "b", 2: "c", 3: "<eos>"}
    engine = GrammarEngine.from_regex("a|b", vocab=vocab, eos_token_id=3)

    assert engine.num_states > 0
    assert engine.num_transitions > 0
    # From start: only 'a' (0) and 'b' (1) are valid -> 2 tokens
    assert engine.active_token_count() == 2
    assert engine.vocab_coverage_ratio() == 0.5


def test_get_batch_masks_shape_and_correctness():
    # Three states: start, after 'a', after 'ab' — different valid continuations.
    vocab = {0: "a", 1: "b", 2: "c", 3: "<eos>"}
    engine = GrammarEngine.from_regex("a(b|c)", vocab=vocab, eos_token_id=3)

    # Collect the first few reachable states for the batch query.
    s0 = engine.dfa.start_state
    s1 = engine.dfa.walk_string(s0, "a")
    assert s1 is not None, "DFA must have a state after 'a'"

    batch = engine.get_batch_masks([s0, s1])
    assert batch.shape == (2, 4), f"expected (2, 4), got {batch.shape}"
    assert batch.dtype == torch.bool

    # Row 0 (start): only 'a' (id 0) is valid.
    assert batch[0, 0].item() is True   # 'a' valid
    assert batch[0, 1].item() is False  # 'b' invalid from start
    assert batch[0, 3].item() is False  # EOS not valid before accepting

    # Row 1 (after 'a'): 'b' and 'c' are valid, 'a' is not.
    assert batch[1, 1].item() is True   # 'b' valid
    assert batch[1, 2].item() is True   # 'c' valid
    assert batch[1, 0].item() is False  # 'a' invalid here


def test_get_batch_masks_caches_consistently():
    vocab = {0: "x", 1: "y", 2: "<eos>"}
    engine = GrammarEngine.from_regex("x+", vocab=vocab, eos_token_id=2)
    s0 = engine.dfa.start_state

    # Warm cache via get_token_mask first.
    individual = engine.get_token_mask(s0)
    batch = engine.get_batch_masks([s0])

    assert torch.equal(individual, batch[0]), (
        "batch mask row must be identical to individually computed mask"
    )
