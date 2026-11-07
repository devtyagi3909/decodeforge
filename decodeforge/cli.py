"""Command-line interface for inspecting and testing grammars."""

from __future__ import annotations

import argparse
import json
import sys

from decodeforge.dfa import RegexDFA
from decodeforge.schema import json_schema_to_regex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decodeforge",
        description="DecodeForge grammar compiler and state machine inspection CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect states and transitions of a regex or schema"
    )
    inspect_parser.add_argument(
        "--pattern", type=str, help="Regular expression pattern"
    )
    inspect_parser.add_argument("--schema", type=str, help="Path to JSON schema file")
    inspect_parser.add_argument(
        "--test",
        type=str,
        help="Optional candidate string to validate against the grammar",
    )

    args = parser.parse_args(argv)

    if args.command == "inspect":
        if not args.pattern and not args.schema:
            print("Error: Must provide either --pattern or --schema", file=sys.stderr)
            return 1

        pattern = args.pattern
        if args.schema:
            with open(args.schema, "r", encoding="utf-8") as f:
                schema_dict = json.load(f)
            pattern = json_schema_to_regex(schema_dict)

        dfa = RegexDFA.from_pattern(pattern)
        print("=== DecodeForge Grammar Inspection ===")
        print(f"Pattern: {pattern}")
        print(f"States: {len(dfa.states)}")
        print(f"Transitions: {len(dfa.transitions)}")
        print(f"Accepting states: {len(dfa.accept_states)}")

        if args.test is not None:
            end_state = dfa.walk_string(dfa.start_state, args.test)
            if end_state is None:
                print(
                    f"Test string {args.test!r}: REJECTED (invalid prefix / character)"
                )
            elif end_state in dfa.accept_states:
                print(f"Test string {args.test!r}: ACCEPTED (terminal match)")
            else:
                print(f"Test string {args.test!r}: VALID PREFIX (incomplete match)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
