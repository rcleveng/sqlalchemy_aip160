"""Programmatic builder for AIP-160 filter expressions."""

from __future__ import annotations

from .ast import (
    AndExpression,
    Comparison,
    FilterExpression,
    NumberValue,
    Operator,
    StringValue,
    Value,
    WildcardValue,
)


def _coerce_to_value(value: str | int | float | bool | Value) -> Value:
    """Convert a Python literal to the corresponding AST value node."""
    if isinstance(value, (StringValue, NumberValue, WildcardValue)):
        return value
    if isinstance(value, bool):
        # bool before int — bool is a subclass of int in Python
        return StringValue(value="true" if value else "false")
    if isinstance(value, int):
        return NumberValue(value=value)
    if isinstance(value, float):
        return NumberValue(value=value)
    if isinstance(value, str):
        return StringValue(value=value)
    raise TypeError(f"Unsupported value type: {type(value)}")


class FilterBuilder:
    """Fluent builder for constructing AIP-160 filter expressions.

    Example::

        f = FilterBuilder()
        f.add("kind", "=", "issue").add("priority", ">", 3)
        str(f)  # 'kind = "issue" AND priority > 3'
    """

    def __init__(self) -> None:
        self._comparisons: list[Comparison] = []

    def add(
        self,
        field: str,
        op: str | Operator,
        value: str | int | float | bool | Value,
    ) -> FilterBuilder:
        """Add a comparison clause.  Returns *self* for chaining."""
        if isinstance(op, str):
            op = Operator(op)
        self._comparisons.append(
            Comparison(field=field, operator=op, value=_coerce_to_value(value))
        )
        return self

    def build(self) -> FilterExpression:
        """Build and return a :class:`FilterExpression`."""
        if not self._comparisons:
            return FilterExpression(root=None)
        if len(self._comparisons) == 1:
            return FilterExpression(root=self._comparisons[0])
        return FilterExpression(
            root=AndExpression(children=list(self._comparisons))
        )

    def __str__(self) -> str:
        return str(self.build())
