"""Sub-token trie for vocabulary prefix indexing and fast character-level filtering."""

from __future__ import annotations

from collections.abc import Iterable


class TrieNode:
    """Node within the sub-token prefix trie."""

    __slots__ = ("children", "is_leaf", "token_ids")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.token_ids: list[int] = []
        self.is_leaf: bool = False


class SubTokenTrie:
    """Trie structure indexing token strings to their corresponding vocabulary IDs.

    Enables prefix-based token search and incremental string-matching against
    grammar states.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self._vocab_size: int = 0

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def insert(self, token_str: str, token_id: int) -> None:
        """Insert a token string and its vocabulary ID into the trie."""
        node = self.root
        for char in token_str:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_leaf = True
        node.token_ids.append(token_id)
        if token_id >= self._vocab_size:
            self._vocab_size = token_id + 1

    @classmethod
    def from_vocab(
        cls, vocab: Iterable[tuple[int, str]] | dict[int, str] | list[str]
    ) -> SubTokenTrie:
        """Construct a SubTokenTrie from a vocabulary collection.

        Args:
            vocab: Mapping or iterable of (token_id, token_str), or a list where
                   index corresponds to token_id.
        """
        trie = cls()
        if isinstance(vocab, dict):
            for token_id, token_str in vocab.items():
                trie.insert(token_str, token_id)
        elif isinstance(vocab, list):
            for token_id, token_str in enumerate(vocab):
                trie.insert(token_str, token_id)
        else:
            for token_id, token_str in vocab:
                trie.insert(token_str, token_id)
        return trie

    def get_tokens_with_prefix(self, prefix: str) -> list[int]:
        """Return all token IDs that start with the specified prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        tokens: list[int] = []
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr.is_leaf:
                tokens.extend(curr.token_ids)
            stack.extend(curr.children.values())
        return tokens

    def match_prefix(self, text: str) -> TrieNode | None:
        """Navigate the trie along the given text prefix.

        Returns the reached TrieNode, or None if the path diverges.
        """
        node = self.root
        for char in text:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
