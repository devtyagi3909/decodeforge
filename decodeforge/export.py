"""Serialization utilities for compiled DFA state machines."""

from __future__ import annotations

import json
from typing import Any

from decodeforge.dfa import RegexDFA


def dfa_to_dict(dfa: RegexDFA) -> dict[str, Any]:
    """Serialize a RegexDFA instance into a JSON-compatible dictionary."""
    serialized_transitions = [
        {"from": src, "char": ch, "to": dst}
        for (src, ch), dst in dfa.transitions.items()
    ]
    return {
        "states": sorted(dfa.states),
        "start_state": dfa.start_state,
        "accept_states": sorted(dfa.accept_states),
        "transitions": serialized_transitions,
    }


def dfa_from_dict(data: dict[str, Any]) -> RegexDFA:
    """Reconstruct a RegexDFA from a serialized dictionary."""
    states = set(data["states"])
    start_state = int(data["start_state"])
    accept_states = set(data["accept_states"])
    transitions = {(t["from"], t["char"]): t["to"] for t in data["transitions"]}
    return RegexDFA(states, transitions, start_state, accept_states)


def export_dfa_json(dfa: RegexDFA, filepath: str) -> None:
    """Save a RegexDFA to a JSON file."""
    payload = dfa_to_dict(dfa)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_dfa_json(filepath: str) -> RegexDFA:
    """Load a RegexDFA from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dfa_from_dict(data)
