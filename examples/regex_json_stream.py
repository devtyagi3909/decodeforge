"""Example: Constrained regex decoding for structured boolean flags."""

from __future__ import annotations

from decodeforge.engine import GrammarEngine


def main() -> None:
    # Grammar requiring exact status format: "status:(READY|PENDING|FAILED)"
    vocab = {
        0: "status:",
        1: "READY",
        2: "PENDING",
        3: "FAILED",
        4: "UNKNOWN",
        5: "<eos>",
    }

    pattern = r"status:(?:READY|PENDING|FAILED)"
    engine = GrammarEngine.from_regex(pattern, vocab=vocab, eos_token_id=5)

    print("Initial mask:")
    mask = engine.get_token_mask()
    valid_initial = [vocab[i] for i, valid in enumerate(mask) if valid.item()]
    print(f"Permitted start tokens: {valid_initial}")
    assert valid_initial == ["status:"]

    # Step 1: Advance status:
    engine.advance(0)
    mask = engine.get_token_mask()
    valid_next = [vocab[i] for i, valid in enumerate(mask) if valid.item()]
    print(f"Permitted value tokens: {valid_next}")
    assert set(valid_next) == {"READY", "PENDING", "FAILED"}

    # Step 2: Sample READY
    engine.advance(1)
    assert engine.is_terminated() is True
    mask = engine.get_token_mask()
    assert mask[5].item() is True  # EOS permitted
    print("Terminated successfully.")


if __name__ == "__main__":
    main()
