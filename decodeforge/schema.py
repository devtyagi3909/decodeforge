"""Schema to regular expression compiler for JSON and Pydantic models."""

from __future__ import annotations

import json
from typing import Any


def pydantic_to_regex(model: type[Any]) -> str:
    """Compile a Pydantic BaseModel class into a strict JSON-matching regular expression."""
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    elif hasattr(model, "schema"):
        schema = model.schema()
    else:
        raise TypeError(f"Expected a Pydantic model class, got {model}")
    return json_schema_to_regex(schema)


def json_schema_to_regex(schema: dict[str, Any]) -> str:
    """Compile a JSON Schema dictionary into a regular expression pattern."""
    defs = schema.get("$defs") or schema.get("definitions") or {}
    return _compile_schema_node(schema, defs)


def _compile_schema_node(node: dict[str, Any], defs: dict[str, Any]) -> str:
    """Recursively convert a schema node to regex."""
    if "$ref" in node:
        ref_path = node["$ref"]
        ref_name = ref_path.split("/")[-1]
        if ref_name in defs:
            return _compile_schema_node(defs[ref_name], defs)
        return r".*"

    node_type = node.get("type")

    # Handle anyOf / oneOf
    if "anyOf" in node or "oneOf" in node:
        branches = node.get("anyOf") or node.get("oneOf") or []
        branch_regexes = [_compile_schema_node(b, defs) for b in branches]
        return r"(?:" + "|".join(branch_regexes) + r")"

    if "enum" in node:
        escaped_choices = [re_escape_json_val(val) for val in node["enum"]]
        return r"(?:" + "|".join(escaped_choices) + r")"

    if node_type == "string":
        return r'"([^"\\]|\\.)*"'
    elif node_type == "integer":
        return r"-?(?:0|[1-9]\d*)"
    elif node_type == "number":
        return r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    elif node_type == "boolean":
        return r"(?:true|false)"
    elif node_type == "null":
        return r"null"
    elif node_type == "array":
        items_schema = node.get("items", {})
        item_regex = _compile_schema_node(items_schema, defs) if items_schema else r".*"
        # Matches [] or [ item, item, ... ]
        return r"\[\s*(?:" + item_regex + r"(?:\s*,\s*" + item_regex + r")*)?\s*\]"
    elif node_type == "object" or "properties" in node:
        props = node.get("properties", {})
        set(node.get("required", []))
        if not props:
            # Generic dictionary/object: {"key": value}
            val_schema = node.get("additionalProperties", {})
            val_regex = (
                _compile_schema_node(val_schema, defs)
                if isinstance(val_schema, dict) and val_schema
                else r'"([^"\\]|\\.)*"'
            )
            return (
                r"\{\s*(?:\"([^\"\\]|\\.)*\"\s*:\s*"
                + val_regex
                + r"(?:\s*,\s*\"([^\"\\]|\\.)*\"\s*:\s*"
                + val_regex
                + r")*)?\s*\}"
            )

        prop_patterns = []
        for prop_name, prop_node in props.items():
            val_reg = _compile_schema_node(prop_node, defs)
            field_pattern = r'"' + json.dumps(prop_name)[1:-1] + r'"\s*:\s*' + val_reg
            prop_patterns.append(field_pattern)

        joined_props = r"\s*,\s*".join(prop_patterns)
        return r"\{\s*" + joined_props + r"\s*\}"

    # Default fallback
    return r".*"


def re_escape_json_val(val: Any) -> str:
    """Format and escape a Python literal value as a JSON regex string."""
    encoded = json.dumps(val)
    # Escape special regex characters except for quotes and basic json structures
    escaped = ""
    for char in encoded:
        if char in r"\.^$*+?{}[]()|":
            escaped += "\\" + char
        else:
            escaped += char
    return escaped
