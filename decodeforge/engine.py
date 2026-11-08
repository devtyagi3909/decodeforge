"""GrammarEngine: compiles schemas and regexes into high-throughput token bitmasks."""

from __future__ import annotations

from typing import Any

import torch

from decodeforge.dfa import RegexDFA
from decodeforge.schema import json_schema_to_regex, pydantic_to_regex


class GrammarEngine:
    """High-throughput constrained grammar decoding engine.

    Pre-compiles regular expressions and schemas into a deterministic finite
    automaton (DFA) and generates O(1) cached vocabulary bitmasks for rejection
    sampling in LLM inference loops.
    """

    def __init__(
        self,
        dfa: RegexDFA,
        vocab: dict[int, str] | list[str] | Any,
        eos_token_id: int | None = None,
    ) -> None:
        self.dfa = dfa
        self.vocab: dict[int, str] = self._normalize_vocab(vocab)
        self.vocab_size: int = max(self.vocab.keys()) + 1 if self.vocab else 0
        self.eos_token_id: int | None = eos_token_id
        self.current_state: int = dfa.start_state
        self._mask_cache: dict[int, torch.Tensor] = {}
        self._next_state_cache: dict[tuple[int, int], int | None] = {}

    @property
    def state(self) -> int:
        return self.current_state

    @property
    def num_states(self) -> int:
        """Total number of states in the underlying DFA."""
        return len(self.dfa.states)

    @property
    def num_transitions(self) -> int:
        """Total number of transitions in the underlying DFA."""
        return len(self.dfa.transitions)

    def active_token_count(self, state: int | None = None) -> int:
        """Return the number of valid tokens from the given or current state."""
        mask = self.get_token_mask(state)
        return int(mask.sum().item())

    def vocab_coverage_ratio(self, state: int | None = None) -> float:
        """Return the fraction of vocabulary tokens permitted from the given or current state."""
        if self.vocab_size == 0:
            return 0.0
        return self.active_token_count(state) / float(self.vocab_size)

    def reset(self) -> None:
        """Reset the engine to the initial grammar state."""
        self.current_state = self.dfa.start_state

    def is_terminated(self) -> bool:
        """Return True if the current state satisfies the grammar (EOS allowed)."""
        return self.dfa.is_accepting(self.current_state)

    def get_token_mask(self, state: int | None = None) -> torch.Tensor:
        """Return a boolean tensor mask of shape `(vocab_size,)` for valid next tokens.

        True indicates the token is permissible; False indicates it violates the grammar.
        """
        curr = self.current_state if state is None else state
        if curr in self._mask_cache:
            return self._mask_cache[curr]

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        for token_id, token_str in self.vocab.items():
            nxt_state = self.dfa.walk_string(curr, token_str)
            if nxt_state is not None:
                mask[token_id] = True
                self._next_state_cache[(curr, token_id)] = nxt_state
            else:
                self._next_state_cache[(curr, token_id)] = None

        if self.eos_token_id is not None and self.dfa.is_accepting(curr):
            mask[self.eos_token_id] = True

        self._mask_cache[curr] = mask
        return mask

    def get_batch_masks(self, states: list[int]) -> torch.Tensor:
        """Return a stacked boolean mask tensor of shape ``(len(states), vocab_size)``
        for a batch of grammar states.

        Useful for parallel beam search or speculative decoding where multiple
        hypothesis states must be masked simultaneously.

        Args:
            states: List of DFA state IDs to generate masks for.

        Returns:
            Boolean tensor of shape ``(len(states), vocab_size)``.  Row ``i``
            is the token mask for ``states[i]``.
        """
        return torch.stack([self.get_token_mask(s) for s in states])

    def advance(self, token_id: int) -> int:
        """Advance the grammar state machine with the sampled token.

        Args:
            token_id: Token ID selected in the inference loop.

        Returns:
            New state ID.

        Raises:
            ValueError: If the token violates the grammar.
        """
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            if not self.is_terminated():
                raise ValueError(
                    f"EOS token {token_id} emitted before grammar reached accepting state "
                    f"(current state: {self.current_state})"
                )
            return self.current_state

        cache_key = (self.current_state, token_id)
        if cache_key in self._next_state_cache:
            nxt = self._next_state_cache[cache_key]
        else:
            token_str = self.vocab.get(token_id)
            if token_str is None:
                nxt = None
            else:
                nxt = self.dfa.walk_string(self.current_state, token_str)
            self._next_state_cache[cache_key] = nxt

        if nxt is None:
            token_repr = self.vocab.get(token_id, f"ID={token_id}")
            raise ValueError(
                f"Token {token_repr!r} (id {token_id}) violates grammar "
                f"from state {self.current_state}"
            )

        self.current_state = nxt
        return self.current_state

    @classmethod
    def from_regex(
        cls,
        pattern: str,
        tokenizer: Any = None,
        *,
        vocab: Any = None,
        eos_token_id: int | None = None,
    ) -> GrammarEngine:
        """Compile a regular expression pattern into a GrammarEngine.

        Args:
            pattern: Regular expression string.
            tokenizer: Tokenizer instance or vocabulary dictionary.
            vocab: Alternative alias for tokenizer.
            eos_token_id: Optional EOS token ID override.
        """
        tok = tokenizer if tokenizer is not None else vocab
        if tok is None:
            raise ValueError("Either tokenizer or vocab must be provided.")
        dfa = RegexDFA.from_pattern(pattern)
        vocab_dict, detected_eos = cls._extract_vocab_and_eos(tok)
        return cls(dfa, vocab_dict, eos_token_id=eos_token_id or detected_eos)

    @classmethod
    def from_pydantic(
        cls,
        model: type[Any],
        tokenizer: Any = None,
        *,
        vocab: Any = None,
        eos_token_id: int | None = None,
    ) -> GrammarEngine:
        """Compile a Pydantic model into a GrammarEngine."""
        regex_pattern = pydantic_to_regex(model)
        return cls.from_regex(
            regex_pattern,
            tokenizer=tokenizer,
            vocab=vocab,
            eos_token_id=eos_token_id,
        )

    @classmethod
    def from_json_schema(
        cls,
        schema: dict[str, Any],
        tokenizer: Any = None,
        *,
        vocab: Any = None,
        eos_token_id: int | None = None,
    ) -> GrammarEngine:
        """Compile a JSON schema dict into a GrammarEngine."""
        regex_pattern = json_schema_to_regex(schema)
        return cls.from_regex(
            regex_pattern,
            tokenizer=tokenizer,
            vocab=vocab,
            eos_token_id=eos_token_id,
        )

    @staticmethod
    def _extract_vocab_and_eos(tokenizer: Any) -> tuple[dict[int, str], int | None]:
        """Extract vocabulary mapping and eos_token_id from various tokenizer formats."""
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        if isinstance(tokenizer, dict):
            # Already a mapping of {token_id: token_str} or {token_str: token_id}
            sample_k, sample_v = (
                next(iter(tokenizer.items())) if tokenizer else (None, None)
            )
            if isinstance(sample_k, int) and isinstance(sample_v, str):
                return dict(tokenizer), eos_token_id
            elif isinstance(sample_k, str) and isinstance(sample_v, int):
                return {v: k for k, v in tokenizer.items()}, eos_token_id
            return dict(tokenizer), eos_token_id

        if isinstance(tokenizer, list):
            return {i: t for i, t in enumerate(tokenizer)}, eos_token_id

        # HuggingFace Tokenizer / PreTrainedTokenizer
        if hasattr(tokenizer, "get_vocab"):
            vocab_dict = tokenizer.get_vocab()
            # In transformers, get_vocab() returns {token_str: token_id}
            inverted = {idx: token for token, idx in vocab_dict.items()}
            return inverted, eos_token_id

        if hasattr(tokenizer, "vocab"):
            vocab_dict = tokenizer.vocab
            inverted = {idx: token for token, idx in vocab_dict.items()}
            return inverted, eos_token_id

        raise TypeError(
            f"Unable to extract vocabulary from tokenizer of type {type(tokenizer)}. "
            "Pass a dict of {token_id: token_str} or a HuggingFace tokenizer."
        )

    @staticmethod
    def _normalize_vocab(vocab: dict[int, str] | list[str] | Any) -> dict[int, str]:
        if isinstance(vocab, dict):
            return {int(k): str(v) for k, v in vocab.items()}
        elif isinstance(vocab, list):
            return {i: str(t) for i, t in enumerate(vocab)}
        return dict(vocab)
