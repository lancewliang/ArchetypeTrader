"""Small HTML template renderer for static report files.

The report templates intentionally use a tiny subset of Jinja-like syntax so
Phase I reporting can keep HTML outside Python code without adding a runtime
dependency:

    - ``{{ value.path }}`` for escaped variable interpolation;
    - ``{% if value.path %}`` / ``{% else %}`` / ``{% endif %}``;
    - ``{% for item in values %}`` / ``{% endfor %}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_TOKEN_PATTERN = re.compile(r"({{.*?}}|{%.*?%})", re.DOTALL)
_FOR_PATTERN = re.compile(r"^for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)$")


class TemplateSyntaxError(ValueError):
    """Raised when a report template contains unsupported syntax."""


@dataclass(frozen=True)
class _TextNode:
    value: str


@dataclass(frozen=True)
class _VariableNode:
    expression: str


@dataclass(frozen=True)
class _IfNode:
    expression: str
    true_nodes: tuple["_Node", ...]
    false_nodes: tuple["_Node", ...]


@dataclass(frozen=True)
class _ForNode:
    variable_name: str
    expression: str
    body_nodes: tuple["_Node", ...]


_Node = _TextNode | _VariableNode | _IfNode | _ForNode


def render_template_file(path: str | Path, context: Mapping[str, Any]) -> str:
    """Render a UTF-8 template file with escaped variables."""

    template_path = Path(path)
    return render_template(template_path.read_text(encoding="utf-8"), context)


def render_template(template: str, context: Mapping[str, Any]) -> str:
    """Render a template string with the minimal report template syntax."""

    tokens = _tokenize(template)
    nodes, index, end_tag = _parse_nodes(tokens)
    if end_tag is not None:
        raise TemplateSyntaxError(f"unexpected template tag: {end_tag}")
    if index != len(tokens):
        raise TemplateSyntaxError("template parser stopped before the end")
    return _render_nodes(nodes, [context])


def _tokenize(template: str) -> tuple[str, ...]:
    return tuple(part for part in _TOKEN_PATTERN.split(template) if part)


def _parse_nodes(
    tokens: Sequence[str],
    index: int = 0,
    end_tags: frozenset[str] = frozenset(),
) -> tuple[tuple[_Node, ...], int, str | None]:
    nodes: list[_Node] = []
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("{{") and token.endswith("}}"):
            nodes.append(_VariableNode(token[2:-2].strip()))
            index += 1
            continue
        if not (token.startswith("{%") and token.endswith("%}")):
            nodes.append(_TextNode(token))
            index += 1
            continue

        tag = token[2:-2].strip()
        if tag in end_tags:
            return tuple(nodes), index, tag
        if tag.startswith("if "):
            true_nodes, index, found_tag = _parse_nodes(
                tokens,
                index + 1,
                frozenset({"else", "endif"}),
            )
            false_nodes: tuple[_Node, ...] = ()
            if found_tag == "else":
                false_nodes, index, found_tag = _parse_nodes(
                    tokens,
                    index + 1,
                    frozenset({"endif"}),
                )
            if found_tag != "endif":
                raise TemplateSyntaxError(f"unclosed if block: {tag}")
            nodes.append(_IfNode(tag[3:].strip(), true_nodes, false_nodes))
            index += 1
            continue
        match = _FOR_PATTERN.match(tag)
        if match:
            body_nodes, index, found_tag = _parse_nodes(
                tokens,
                index + 1,
                frozenset({"endfor"}),
            )
            if found_tag != "endfor":
                raise TemplateSyntaxError(f"unclosed for block: {tag}")
            nodes.append(_ForNode(match.group(1), match.group(2).strip(), body_nodes))
            index += 1
            continue
        if tag in {"else", "endif", "endfor"}:
            raise TemplateSyntaxError(f"unexpected template tag: {tag}")
        raise TemplateSyntaxError(f"unsupported template tag: {tag}")
    return tuple(nodes), index, None


def _render_nodes(nodes: Sequence[_Node], scopes: list[Mapping[str, Any]]) -> str:
    chunks: list[str] = []
    for node in nodes:
        if isinstance(node, _TextNode):
            chunks.append(node.value)
        elif isinstance(node, _VariableNode):
            chunks.append(escape(str(_resolve(node.expression, scopes))))
        elif isinstance(node, _IfNode):
            branch = (
                node.true_nodes
                if _resolve(node.expression, scopes)
                else node.false_nodes
            )
            chunks.append(_render_nodes(branch, scopes))
        elif isinstance(node, _ForNode):
            values = _resolve(node.expression, scopes)
            if values is None:
                continue
            for value in values:
                scopes.append({node.variable_name: value})
                chunks.append(_render_nodes(node.body_nodes, scopes))
                scopes.pop()
    return "".join(chunks)


def _resolve(expression: str, scopes: Sequence[Mapping[str, Any]]) -> Any:
    current: Any = _resolve_name(expression.split(".", maxsplit=1)[0], scopes)
    for part in expression.split(".")[1:]:
        if isinstance(current, Mapping):
            current = current.get(part, "")
        else:
            current = getattr(current, part, "")
    return "" if current is None else current


def _resolve_name(name: str, scopes: Sequence[Mapping[str, Any]]) -> Any:
    for scope in reversed(scopes):
        if name in scope:
            return scope[name]
    return ""


__all__ = ["TemplateSyntaxError", "render_template", "render_template_file"]
