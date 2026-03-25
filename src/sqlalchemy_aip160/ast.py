"""AST node types for AIP-160 filter expressions.

Provides a structured representation of parsed AIP-160 filters that can be
inspected, manipulated, serialized back to strings, and combined.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Union


class Operator(Enum):
    """AIP-160 comparison operators."""

    EQ = "="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    HAS = ":"


@dataclass
class StringValue:
    """A quoted string value in a filter expression."""

    value: str
    quote_char: str = '"'


@dataclass
class NumberValue:
    """A numeric value (int or float) in a filter expression."""

    value: int | float


@dataclass
class WildcardValue:
    """A bare wildcard (*) value, used in expressions like ``field:*``."""

    pass


# Union of all possible value types in a comparison.
Value = Union[StringValue, NumberValue, WildcardValue]


@dataclass
class Comparison:
    """A single field-operator-value comparison (e.g. ``status = "active"``)."""

    field: str
    operator: Operator
    value: Value


@dataclass
class AndExpression:
    """Logical AND of two or more sub-expressions."""

    children: list[FilterNode]

    def __post_init__(self) -> None:
        flattened: list[FilterNode] = []
        for child in self.children:
            if isinstance(child, AndExpression):
                flattened.extend(child.children)
            else:
                flattened.append(child)
        self.children = flattened


@dataclass
class OrExpression:
    """Logical OR of two or more sub-expressions."""

    children: list[FilterNode]

    def __post_init__(self) -> None:
        flattened: list[FilterNode] = []
        for child in self.children:
            if isinstance(child, OrExpression):
                flattened.extend(child.children)
            else:
                flattened.append(child)
        self.children = flattened


@dataclass
class NotExpression:
    """Logical NOT of a sub-expression."""

    child: FilterNode


# Union of all AST node types.
FilterNode = Union[Comparison, AndExpression, OrExpression, NotExpression]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_value(value: Value) -> str:
    if isinstance(value, StringValue):
        q = value.quote_char
        escaped = value.value.replace("\\", "\\\\").replace(q, f"\\{q}")
        return f"{q}{escaped}{q}"
    if isinstance(value, NumberValue):
        return str(value.value)
    if isinstance(value, WildcardValue):
        return "*"
    raise TypeError(f"Unknown value type: {type(value)}")  # pragma: no cover


def _serialize_node(node: FilterNode, parent_type: type | None = None) -> str:
    if isinstance(node, Comparison):
        # Special case: field:* (has-wildcard)
        if node.operator == Operator.HAS and isinstance(node.value, WildcardValue):
            return f"{node.field}:*"
        return f"{node.field} {node.operator.value} {_serialize_value(node.value)}"

    if isinstance(node, AndExpression):
        parts = [_serialize_node(c, AndExpression) for c in node.children]
        result = " AND ".join(parts)
        # AND inside OR needs parentheses (AND is lower precedence in AIP-160)
        if parent_type is OrExpression:
            return f"({result})"
        return result

    if isinstance(node, OrExpression):
        parts = [_serialize_node(c, OrExpression) for c in node.children]
        result = " OR ".join(parts)
        # OR inside AND does NOT need parens (OR is higher precedence)
        return result

    if isinstance(node, NotExpression):
        inner = _serialize_node(node.child, NotExpression)
        if isinstance(node.child, (AndExpression, OrExpression)):
            return f"NOT ({inner})"
        return f"NOT {inner}"

    raise TypeError(f"Unknown node type: {type(node)}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Tree-walking helpers
# ---------------------------------------------------------------------------


def _collect_fields(node: FilterNode) -> set[str]:
    if isinstance(node, Comparison):
        return {node.field}
    if isinstance(node, (AndExpression, OrExpression)):
        result: set[str] = set()
        for child in node.children:
            result |= _collect_fields(child)
        return result
    if isinstance(node, NotExpression):
        return _collect_fields(node.child)
    return set()  # pragma: no cover


def _field_matches(node_field: str, target: str) -> bool:
    """Check if *node_field* matches *target* exactly or as a dotted prefix."""
    return node_field == target or node_field.startswith(target + ".")


def _rename_field_in_node(node: FilterNode, old: str, new: str) -> None:
    if isinstance(node, Comparison):
        if node.field == old:
            node.field = new
        elif node.field.startswith(old + "."):
            node.field = new + node.field[len(old) :]
    elif isinstance(node, (AndExpression, OrExpression)):
        for child in node.children:
            _rename_field_in_node(child, old, new)
    elif isinstance(node, NotExpression):
        _rename_field_in_node(node.child, old, new)


def _remove_field_from_node(
    node: FilterNode,
    field: str,
    extracted: list[Comparison] | None = None,
) -> FilterNode | None:
    """Remove comparisons matching *field*. Returns pruned subtree or ``None``."""
    if isinstance(node, Comparison):
        if _field_matches(node.field, field):
            if extracted is not None:
                extracted.append(node)
            return None
        return node

    if isinstance(node, (AndExpression, OrExpression)):
        new_children: list[FilterNode] = []
        for child in node.children:
            result = _remove_field_from_node(child, field, extracted)
            if result is not None:
                new_children.append(result)
        if len(new_children) == 0:
            return None
        if len(new_children) == 1:
            return new_children[0]
        if isinstance(node, AndExpression):
            return AndExpression(children=new_children)
        return OrExpression(children=new_children)

    if isinstance(node, NotExpression):
        result = _remove_field_from_node(node.child, field, extracted)
        if result is None:
            return None
        return NotExpression(child=result)

    return node  # pragma: no cover


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


@dataclass
class FilterExpression:
    """A structured, manipulable AIP-160 filter expression.

    This is the primary public type returned by :func:`parse_filter` and
    accepted by :func:`apply_filter`.
    """

    root: FilterNode | None = None

    # -- serialization -------------------------------------------------------

    def __str__(self) -> str:
        if self.root is None:
            return ""
        return _serialize_node(self.root)

    def __repr__(self) -> str:
        return f"FilterExpression({self.root!r})"

    def __bool__(self) -> bool:
        return self.root is not None

    # -- inspection ----------------------------------------------------------

    def get_fields(self) -> set[str]:
        """Return the set of all field names referenced in the expression."""
        if self.root is None:
            return set()
        return _collect_fields(self.root)

    # -- mutation ------------------------------------------------------------

    def rename_field(self, old: str, new: str) -> None:
        """Rename a field in-place.  Only touches field positions, not values."""
        if self.root is not None:
            _rename_field_in_node(self.root, old, new)

    def remove(self, field: str) -> None:
        """Remove all clauses referencing *field*.

        Logical connectors are cleaned up automatically.
        """
        if self.root is not None:
            self.root = _remove_field_from_node(self.root, field)

    def extract(self, field: str) -> list[Comparison]:
        """Remove and return all clauses matching *field*.

        Works like :meth:`remove` but also returns the removed
        :class:`Comparison` nodes.
        """
        if self.root is None:
            return []
        extracted: list[Comparison] = []
        self.root = _remove_field_from_node(self.root, field, extracted)
        return extracted

    # -- combining -----------------------------------------------------------

    def __and__(self, other: FilterExpression) -> FilterExpression:
        if self.root is None:
            return FilterExpression(root=copy.deepcopy(other.root))
        if other.root is None:
            return FilterExpression(root=copy.deepcopy(self.root))
        return FilterExpression(
            root=AndExpression(
                children=[copy.deepcopy(self.root), copy.deepcopy(other.root)]
            )
        )

    def __or__(self, other: FilterExpression) -> FilterExpression:
        if self.root is None or other.root is None:
            # anything OR match-all = match-all
            return FilterExpression(root=None)
        return FilterExpression(
            root=OrExpression(
                children=[copy.deepcopy(self.root), copy.deepcopy(other.root)]
            )
        )

    def __invert__(self) -> FilterExpression:
        if self.root is None:
            raise ValueError("Cannot negate an empty (match-all) FilterExpression")
        return FilterExpression(root=NotExpression(child=copy.deepcopy(self.root)))
