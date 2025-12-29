"""AIP-160 filter implementation for SQLAlchemy.

This module implements Google's AIP-160 filtering specification for use with
SQLAlchemy models. It parses filter strings and converts them to SQLAlchemy
filter expressions.

Usage:
    from sqlalchemy_aip160 import apply_filter

    query = session.query(MyModel)
    filtered_query = apply_filter(query, MyModel, 'status = "active" AND priority > 5')

Supported operations:
    - Comparisons: =, !=, <, >, <=, >=
    - Logical operators: AND, OR, NOT (or -)
    - Has operator (:) for checking membership or field presence
    - Nested field access: field.subfield
    - Functions: Currently limited (extensible)
    - Wildcards: field = "*.foo" for pattern matching
    - SQLAlchemy synonyms: filter by synonym names as well as column names

Reference: https://google.aip.dev/160
"""

import logging
from datetime import datetime
from functools import lru_cache
from typing import Any, TypeVar
from uuid import UUID

from lark import Lark, Transformer
from lark.exceptions import LarkError, VisitError
from sqlalchemy import Select, and_, inspect, not_, or_
from sqlalchemy.orm import InstrumentedAttribute

logger = logging.getLogger(__name__)

T = TypeVar("T")

# AIP-160 Grammar based on https://google.aip.dev/assets/misc/ebnf-filtering.txt
# Modified to handle whitespace implicitly for easier parsing
AIP160_GRAMMAR = r"""
// AIP-160 Filter Grammar
// Reference: https://google.aip.dev/160

filter: expression?

// AND has lower precedence than OR (per AIP-160)
expression: sequence (AND sequence)*

// Implicit AND between adjacent terms (fuzzy AND)
sequence: factor+

// OR has higher precedence than AND (per AIP-160)
factor: term (OR term)*

// NOT/- prefix
term: NOT simple -> not_term
    | MINUS simple -> not_term
    | simple

simple: restriction
      | composite

// field op value, or just field (existence)
restriction: comparable comparator arg -> comparison
           | comparable

comparable: function
          | member

// Field access with optional nested fields
member: NAME ("." NAME)*

// Function call
function: NAME ("." NAME)* "(" [arglist] ")"

// Comparators - order matters, longest first
comparator: LESS_EQUALS  -> op_le
          | GREATER_EQUALS -> op_ge
          | NOT_EQUALS -> op_ne
          | LESS_THAN -> op_lt
          | GREATER_THAN -> op_gt
          | EQUALS -> op_eq
          | HAS -> op_has

composite: "(" expression ")"

arglist: arg ("," arg)*

// arg can be a literal, comparable, composite, or wildcard
arg: STRING
   | NUMBER
   | STAR
   | comparable
   | composite

// Terminals - order matters for matching
NOT: "NOT"i
AND: "AND"i
OR: "OR"i
MINUS: "-"
STAR: "*"

LESS_EQUALS: "<="
GREATER_EQUALS: ">="
NOT_EQUALS: "!="
LESS_THAN: "<"
GREATER_THAN: ">"
EQUALS: "="
HAS: ":"

// NAME cannot be a keyword
NAME: /(?!(NOT|AND|OR)\b)[a-zA-Z_][a-zA-Z0-9_]*/i

// Literals
NUMBER: /[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?/
STRING: /"[^"\\]*(?:\\.[^"\\]*)*"/ | /'[^'\\]*(?:\\.[^'\\]*)*'/

%import common.WS
%ignore WS
"""


class FilterError(Exception):
    """Exception raised for filter parsing or application errors."""

    pass


class InvalidFieldError(FilterError):
    """Exception raised when a filter references an invalid field."""

    pass


class InvalidOperatorError(FilterError):
    """Exception raised when an invalid operator is used."""

    pass


@lru_cache(maxsize=1)
def _get_parser() -> Lark:
    """Get or create the cached Lark parser."""
    return Lark(AIP160_GRAMMAR, start="filter", parser="lalr")


def _coerce_value(column: InstrumentedAttribute, value: Any) -> Any:
    """Coerce a parsed value to match the column's Python type."""
    if value is None:
        return None

    python_type = None
    try:
        python_type = column.type.python_type
    except NotImplementedError:
        return value

    if python_type is None:
        return value

    # Handle string values that need conversion
    if isinstance(value, str):
        if python_type is bool:
            return value.lower() in ("true", "1", "yes")
        if python_type is int:
            return int(value)
        if python_type is float:
            return float(value)
        if python_type == UUID:
            return UUID(value)
        if python_type == datetime:
            # Support ISO format datetime strings
            return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return value


def _wildcard_to_like(pattern: str) -> str:
    """Convert AIP-160 wildcard pattern to SQL LIKE pattern."""
    # Escape SQL LIKE special characters except our wildcard
    result = pattern.replace("%", r"\%").replace("_", r"\_")
    # Convert * to %
    result = result.replace("*", "%")
    return result


class SQLAlchemyTransformer(Transformer):
    """Transforms a parsed AIP-160 filter tree into SQLAlchemy expressions."""

    def __init__(
        self,
        model_class: type,
        allowed_fields: set[str] | None = None,
        field_aliases: dict[str, str] | None = None,
    ):
        """Initialize the transformer.

        Args:
            model_class: The SQLAlchemy model class to filter on.
            allowed_fields: Optional set of field names that are allowed in filters.
                           If None, all model columns are allowed.
            field_aliases: Optional mapping of alias names to actual field paths.
                          E.g., {"category": "category.name"} allows users to filter
                          with `category = "x"` which resolves to `category.name = "x"`.
        """
        super().__init__()
        self.model_class = model_class
        self._allowed_fields = allowed_fields
        self._field_aliases = field_aliases
        self._column_map = self._build_column_map()
        self._pending_joins: dict[
            str, InstrumentedAttribute
        ] = {}  # path -> relationship attr

    def _build_column_map(self) -> dict[str, InstrumentedAttribute]:
        """Build a mapping of field names to SQLAlchemy columns, including synonyms."""
        columns = {}

        # Add regular columns
        for name in dir(self.model_class):
            attr = getattr(self.model_class, name)
            if isinstance(attr, InstrumentedAttribute):
                if self._allowed_fields is None or name in self._allowed_fields:
                    columns[name] = attr

        # Add synonyms - resolve to target column
        try:
            mapper = inspect(self.model_class)
            for syn_name, syn_prop in mapper.synonyms.items():
                if self._allowed_fields is None or syn_name in self._allowed_fields:
                    # Get the target column's InstrumentedAttribute
                    target_attr = getattr(self.model_class, syn_prop.name)
                    if isinstance(target_attr, InstrumentedAttribute):
                        columns[syn_name] = target_attr
        except (AttributeError, TypeError) as e:
            # Inspection may fail for unmapped classes or unusual configurations
            logger.debug(
                "Failed to inspect model %s for synonyms: %s", self.model_class, e
            )

        return columns

    def _resolve_alias(self, field_name: str) -> str:
        """Resolve field aliases to actual paths.

        Args:
            field_name: The field name as specified in the filter.

        Returns:
            The resolved field path (may be the same if no alias exists).
        """
        if self._field_aliases and field_name in self._field_aliases:
            return self._field_aliases[field_name]
        return field_name

    def _get_column(self, field_path: str) -> tuple[InstrumentedAttribute, str | None]:
        """Get a column from a potentially nested field path.

        Supports relationship traversal (e.g., "category.name") and tracks
        required joins for later application to the query.

        Args:
            field_path: The field path, optionally with dots for relationships.

        Returns:
            Tuple of (column, join_path) where join_path is None for direct
            columns or the relationship path for nested access.

        Raises:
            InvalidFieldError: If the field or relationship doesn't exist.
        """
        parts = field_path.split(".")

        # Validate against allowed_fields BEFORE alias resolution
        # This ensures users can only use fields/aliases they're allowed to
        original_field = parts[0]
        if (
            self._allowed_fields is not None
            and original_field not in self._allowed_fields
        ):
            available = ", ".join(sorted(self._allowed_fields))
            raise InvalidFieldError(
                f"Field '{original_field}' is not allowed. Allowed fields: {available}"
            )

        # Resolve aliases (e.g., "category" -> "category.name" or "title" -> "name")
        resolved_path = self._resolve_alias(field_path)

        # Re-split after alias resolution (alias may change the path)
        parts = resolved_path.split(".")

        # Simple case - direct column access
        if len(parts) == 1:
            field_name = parts[0]
            if field_name not in self._column_map:
                available = ", ".join(sorted(self._column_map.keys()))
                raise InvalidFieldError(
                    f"Unknown field '{field_name}'. Available fields: {available}"
                )
            return self._column_map[field_name], None

        # Relationship traversal (e.g., "category.name" or "category.parent.name")
        current_model = self.model_class
        join_path_parts: list[str] = []

        # Walk through relationship chain (all parts except the last)
        for part in parts[:-1]:
            rel_attr = getattr(current_model, part, None)
            if rel_attr is None:
                raise InvalidFieldError(
                    f"Unknown field or relationship '{part}' on {current_model.__name__}"
                )

            # Check if it's a relationship property
            if not hasattr(rel_attr, "property") or not hasattr(
                rel_attr.property, "mapper"
            ):
                raise InvalidFieldError(
                    f"'{part}' on {current_model.__name__} is not a relationship"
                )

            join_path_parts.append(part)
            join_path = ".".join(join_path_parts)

            # Track this join (deduplication by path)
            if join_path not in self._pending_joins:
                self._pending_joins[join_path] = rel_attr

            # Move to the related model
            current_model = rel_attr.property.mapper.class_

        # Get the final column from the related model
        final_field = parts[-1]
        final_column = getattr(current_model, final_field, None)

        if final_column is None or not isinstance(final_column, InstrumentedAttribute):
            raise InvalidFieldError(
                f"Unknown field '{final_field}' on {current_model.__name__}"
            )

        return final_column, ".".join(parts[:-1])

    def _parse_value(self, value: Any) -> Any:
        """Parse a value token into its Python representation."""
        if isinstance(value, str):
            # Handle quoted strings - strip quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
                # Handle escape sequences
                value = (
                    value.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
                )
        return value

    def filter(self, items):
        """Handle the top-level filter rule."""
        if not items:
            return True  # Empty filter matches everything
        return items[0]

    def expression(self, items):
        """Handle AND expressions between sequences."""
        # Filter out AND tokens, keep only the actual expressions
        exprs = [item for item in items if not isinstance(item, str)]
        if len(exprs) == 1:
            return exprs[0]
        return and_(*exprs)

    def sequence(self, items):
        """Handle implicit AND between adjacent terms."""
        if len(items) == 1:
            return items[0]
        return and_(*items)

    def factor(self, items):
        """Handle OR expressions."""
        # Filter out OR tokens, keep only the actual expressions
        exprs = [item for item in items if not isinstance(item, str)]
        if len(exprs) == 1:
            return exprs[0]
        return or_(*exprs)

    def not_term(self, items):
        """Handle NOT/- prefix."""
        # items[0] is the NOT or MINUS token, items[1] is the expression
        return not_(items[-1])

    def term(self, items):
        """Handle a term without negation."""
        return items[0]

    def simple(self, items):
        """Handle simple restriction or composite."""
        return items[0]

    def restriction(self, items):
        """Handle bare field (existence check)."""
        field_path = items[0]
        if isinstance(field_path, str):
            raise FilterError(
                f"Bare value '{field_path}' not supported. "
                "Use explicit field comparisons like 'field = value'."
            )
        return field_path

    def comparison(self, items):
        """Handle field comparison: field op value."""
        field_path, op, value = items[0], items[1], items[2]

        column, _join_path = self._get_column(field_path)
        value = self._parse_value(value)
        value = _coerce_value(column, value)

        if op == "=":
            # Check for wildcard pattern
            if isinstance(value, str) and "*" in value:
                pattern = _wildcard_to_like(value)
                return column.like(pattern)
            return column == value
        elif op == "!=":
            return column != value
        elif op == "<":
            return column < value
        elif op == ">":
            return column > value
        elif op == "<=":
            return column <= value
        elif op == ">=":
            return column >= value
        elif op == ":":
            # Has operator - check for presence or containment
            if value == "*":
                return column.isnot(None)
            # For array/JSON columns, would need special handling
            # For now, treat as equality
            return column == value
        else:
            raise InvalidOperatorError(f"Unknown operator: {op}")

    def comparable(self, items):
        """Handle comparable (member or function)."""
        return items[0]

    def member(self, items):
        """Handle member access (field.subfield)."""
        return ".".join(str(item) for item in items)

    def function(self, items):
        """Handle function calls."""
        # Extract function name from the items
        func_parts = [str(item) for item in items if isinstance(item, str)]
        func_name = ".".join(func_parts) if func_parts else "unknown"
        raise FilterError(
            f"Function '{func_name}' is not supported. "
            "Use standard comparison operators."
        )

    def arglist(self, items):
        """Handle function argument list."""
        return list(items)

    def arg(self, items):
        """Handle a single argument - can be literal, comparable, or composite."""
        item = items[0]
        # If it's already been transformed (comparable/composite), return as-is
        # Otherwise it's a STRING or NUMBER token that we need to handle
        return item

    def composite(self, items):
        """Handle parenthesized expressions."""
        return items[0]

    # Operator handlers - return the operator string
    def op_eq(self, items):
        return "="

    def op_ne(self, items):
        return "!="

    def op_lt(self, items):
        return "<"

    def op_gt(self, items):
        return ">"

    def op_le(self, items):
        return "<="

    def op_ge(self, items):
        return ">="

    def op_has(self, items):
        return ":"

    # Terminal handlers
    def NAME(self, token):
        """Handle NAME tokens."""
        return str(token)

    def STRING(self, token):
        """Handle STRING tokens."""
        return str(token)

    def NUMBER(self, token):
        """Handle NUMBER tokens."""
        s = str(token)
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)

    def NOT(self, token):
        return "NOT"

    def AND(self, token):
        return "AND"

    def OR(self, token):
        return "OR"

    def MINUS(self, token):
        return "-"

    def STAR(self, token):
        return "*"


def parse_filter(filter_string: str | None) -> Any:
    """Parse an AIP-160 filter string into a parse tree.

    Args:
        filter_string: The filter string to parse.

    Returns:
        The parsed tree.

    Raises:
        FilterError: If the filter string is invalid.
    """
    if not filter_string or not filter_string.strip():
        return None

    parser = _get_parser()
    try:
        return parser.parse(filter_string)
    except LarkError as e:
        raise FilterError(f"Invalid filter syntax: {e}") from e


def build_filter_expression(
    model_class: type[T],
    filter_string: str,
    allowed_fields: set[str] | None = None,
    field_aliases: dict[str, str] | None = None,
) -> tuple[Any, list[InstrumentedAttribute]]:
    """Build a SQLAlchemy filter expression from an AIP-160 filter string.

    Args:
        model_class: The SQLAlchemy model class to filter on.
        filter_string: The AIP-160 filter string.
        allowed_fields: Optional set of field names that are allowed in filters.
                       If None, all model columns are allowed.
        field_aliases: Optional mapping of alias names to actual field paths.
                      E.g., {"category": "category.name"} allows users to filter
                      with `category = "x"` which resolves to `category.name = "x"`.

    Returns:
        A tuple of (expression, joins) where:
        - expression: A SQLAlchemy filter expression for .where() or .filter()
        - joins: A list of relationship attributes that need to be joined

    Raises:
        FilterError: If the filter string is invalid.
        InvalidFieldError: If a filter references an invalid field.
    """
    tree = parse_filter(filter_string)
    if tree is None:
        return True, []  # Empty filter matches everything

    transformer = SQLAlchemyTransformer(model_class, allowed_fields, field_aliases)
    try:
        expr = transformer.transform(tree)
        joins = list(transformer._pending_joins.values())
        return expr, joins
    except VisitError as e:
        # Extract the original exception from Lark's VisitError wrapper
        if e.orig_exc is not None:
            raise e.orig_exc from None
        raise FilterError(str(e)) from e


def apply_filter(
    query: Select[tuple[T]],
    model_class: type[T],
    filter_string: str | None,
    allowed_fields: set[str] | None = None,
    field_aliases: dict[str, str] | None = None,
) -> Select[tuple[T]]:
    """Apply an AIP-160 filter to a SQLAlchemy query.

    Args:
        query: The SQLAlchemy query to filter.
        model_class: The SQLAlchemy model class being queried.
        filter_string: The AIP-160 filter string. If None or empty, returns
                      the query unchanged.
        allowed_fields: Optional set of field names that are allowed in filters.
                       If None, all model columns are allowed.
        field_aliases: Optional mapping of alias names to actual field paths.
                      E.g., {"category": "category.name"} allows users to filter
                      with `category = "x"` which resolves to `category.name = "x"`.

    Returns:
        The filtered query.

    Raises:
        FilterError: If the filter string is invalid.
        InvalidFieldError: If a filter references an invalid field.

    Example:
        >>> from sqlalchemy import select
        >>> query = select(User)
        >>> filtered = apply_filter(
        ...     query, User,
        ...     'status = "active" AND created_at > "2024-01-01"'
        ... )

        # With relationship filtering:
        >>> filtered = apply_filter(
        ...     query, User,
        ...     'department.name = "Engineering"'
        ... )

        # With field aliases:
        >>> filtered = apply_filter(
        ...     query, User,
        ...     'department = "Engineering"',
        ...     field_aliases={"department": "department.name"}
        ... )
    """
    if not filter_string or not filter_string.strip():
        return query

    expr, joins = build_filter_expression(
        model_class, filter_string, allowed_fields, field_aliases
    )
    if expr is True:
        return query

    # Apply joins first (for relationship traversal)
    for join_attr in joins:
        query = query.join(join_attr)

    return query.where(expr)
