import time

from pydantic import BaseModel

from decodeforge.engine import GrammarEngine


class NestedMetadata(BaseModel):
    priority: int
    tag: str


class ComplexTicket(BaseModel):
    title: str
    is_active: bool
    count: int


def test_complex_schema_compilation():
    vocab = {
        0: '{"title":',
        1: '"Fix issue"',
        2: ',"is_active":',
        3: "true",
        4: ',"count":',
        5: "42",
        6: "}",
        7: "<eos>",
        8: "random_noise",
    }
    engine = GrammarEngine.from_pydantic(ComplexTicket, vocab=vocab, eos_token_id=7)

    sequence = [0, 1, 2, 3, 4, 5, 6]
    for token_id in sequence:
        mask = engine.get_token_mask()
        assert mask[token_id].item() is True
        assert mask[8].item() is False  # Noise must be masked out
        engine.advance(token_id)

    assert engine.is_terminated() is True
    mask = engine.get_token_mask()
    assert mask[7].item() is True  # EOS now valid


def test_cached_mask_latency():
    # Verify sub-microsecond retrieval for cached states
    vocab = {i: f"token_{i}" for i in range(1000)}
    vocab[0] = "start"
    vocab[1] = "end"
    vocab[2] = "<eos>"

    engine = GrammarEngine.from_regex("start|end", vocab=vocab, eos_token_id=2)

    # Prime cache
    _ = engine.get_token_mask()

    # Benchmark cached access
    start = time.perf_counter()
    iterations = 10_000
    for _ in range(iterations):
        _ = engine.get_token_mask()
    elapsed = time.perf_counter() - start

    per_call_us = (elapsed / iterations) * 1_000_000
    # Cached dictionary lookup should be under 2 microseconds
    assert per_call_us < 2.0, f"Expected < 2μs, got {per_call_us:.3f}μs"
