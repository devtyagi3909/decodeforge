"""Example: Constrained Pydantic tool call generation simulation."""

from __future__ import annotations

import torch
from pydantic import BaseModel

from decodeforge.engine import GrammarEngine
from decodeforge.mask import apply_mask


class DatabaseQuery(BaseModel):
    table: str
    limit: int
    read_only: bool


def main() -> None:
    # Mock vocabulary
    vocab = {
        0: '{"table":',
        1: '"users"',
        2: ',"limit":',
        3: "100",
        4: ',"read_only":',
        5: "true",
        6: "}",
        7: "<eos>",
        8: "DROP TABLE users;",
    }

    engine = GrammarEngine.from_pydantic(DatabaseQuery, vocab=vocab, eos_token_id=7)
    print("Grammar compiled successfully.")
    print(
        f"Total states: {engine.num_states}, total transitions: {engine.num_transitions}"
    )

    # Simulated generation steps
    steps = [0, 1, 2, 3, 4, 5, 6, 7]
    for step_idx, expected_token in enumerate(steps):
        mask = engine.get_token_mask()
        # Mock logits where malicious SQL token (8) has highest unmasked logit
        raw_logits = torch.zeros(len(vocab))
        raw_logits[8] = 50.0  # Would be selected without grammar constraint
        raw_logits[expected_token] = 10.0

        masked_logits = apply_mask(raw_logits, mask)
        chosen_token = torch.argmax(masked_logits).item()

        assert chosen_token == expected_token, (
            f"Step {step_idx}: unexpected token chosen"
        )
        print(
            f"Step {step_idx}: token {chosen_token} ({vocab[chosen_token]!r}) sampled."
        )
        engine.advance(chosen_token)

    print("Generation completed strictly matching DatabaseQuery Pydantic schema!")


if __name__ == "__main__":
    main()
