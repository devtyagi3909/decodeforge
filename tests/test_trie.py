from decodeforge.trie import SubTokenTrie


def test_trie_insert_and_prefix_match():
    vocab = {0: "cat", 1: "cater", 2: "dog", 3: "car", 4: "cart"}
    trie = SubTokenTrie.from_vocab(vocab)

    assert trie.vocab_size == 5
    ca_tokens = set(trie.get_tokens_with_prefix("ca"))
    assert ca_tokens == {0, 1, 3, 4}

    cat_tokens = set(trie.get_tokens_with_prefix("cat"))
    assert cat_tokens == {0, 1}

    dog_tokens = set(trie.get_tokens_with_prefix("do"))
    assert dog_tokens == {2}

    none_tokens = trie.get_tokens_with_prefix("z")
    assert none_tokens == []


def test_trie_walk():
    vocab = ["a", "ab", "abc"]
    trie = SubTokenTrie.from_vocab(vocab)

    node = trie.match_prefix("ab")
    assert node is not None
    assert node.is_leaf is True
    assert 1 in node.token_ids

    node_missing = trie.match_prefix("xyz")
    assert node_missing is None
