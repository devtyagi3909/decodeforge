import tempfile
from pathlib import Path

from decodeforge.dfa import RegexDFA
from decodeforge.export import (
    dfa_from_dict,
    dfa_to_dict,
    export_dfa_json,
    load_dfa_json,
)


def test_dfa_dict_roundtrip():
    dfa = RegexDFA.from_pattern("test|prod")
    data = dfa_to_dict(dfa)

    restored = dfa_from_dict(data)
    assert restored.states == dfa.states
    assert restored.start_state == dfa.start_state
    assert restored.accept_states == dfa.accept_states
    assert restored.walk_string(restored.start_state, "test") in restored.accept_states
    assert restored.walk_string(restored.start_state, "prod") in restored.accept_states
    assert restored.walk_string(restored.start_state, "dev") is None


def test_dfa_file_roundtrip():
    dfa = RegexDFA.from_pattern(r"[0-9]+")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "dfa.json")
        export_dfa_json(dfa, filepath)
        restored = load_dfa_json(filepath)

        assert (
            restored.walk_string(restored.start_state, "12345")
            in restored.accept_states
        )
        assert restored.walk_string(restored.start_state, "abc") is None
