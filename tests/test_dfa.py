from decodeforge.dfa import RegexDFA


def test_dfa_literal():
    dfa = RegexDFA.from_pattern("hello")
    assert dfa.walk_string(dfa.start_state, "hello") in dfa.accept_states
    assert dfa.walk_string(dfa.start_state, "hell") not in dfa.accept_states
    assert dfa.walk_string(dfa.start_state, "world") is None


def test_dfa_alternation():
    dfa = RegexDFA.from_pattern("cat|dog")
    cat_end = dfa.walk_string(dfa.start_state, "cat")
    dog_end = dfa.walk_string(dfa.start_state, "dog")
    assert cat_end in dfa.accept_states
    assert dog_end in dfa.accept_states
    assert dfa.walk_string(dfa.start_state, "cow") is None


def test_dfa_quantifiers():
    # Digits +
    dfa = RegexDFA.from_pattern(r"\d+")
    assert dfa.walk_string(dfa.start_state, "12345") in dfa.accept_states
    assert dfa.walk_string(dfa.start_state, "a") is None


def test_dfa_char_class():
    dfa = RegexDFA.from_pattern(r"[a-c]+")
    assert dfa.walk_string(dfa.start_state, "abac") in dfa.accept_states
    assert dfa.walk_string(dfa.start_state, "d") is None
