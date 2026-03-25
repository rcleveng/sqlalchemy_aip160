"""Tests for the FilterExpression AST: parsing, serialization, inspection, and manipulation."""

import pytest

from sqlalchemy_aip160 import (
    AndExpression,
    Comparison,
    FilterBuilder,
    FilterExpression,
    NotExpression,
    NumberValue,
    Operator,
    OrExpression,
    StringValue,
    WildcardValue,
    parse_filter,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseFilter:
    """Tests for parse_filter returning FilterExpression."""

    def test_empty_string(self):
        expr = parse_filter("")
        assert expr.root is None
        assert str(expr) == ""

    def test_none(self):
        expr = parse_filter(None)
        assert expr.root is None

    def test_whitespace(self):
        expr = parse_filter("   ")
        assert expr.root is None

    def test_simple_equality(self):
        expr = parse_filter('status = "active"')
        assert isinstance(expr.root, Comparison)
        assert expr.root.field == "status"
        assert expr.root.operator == Operator.EQ
        assert isinstance(expr.root.value, StringValue)
        assert expr.root.value.value == "active"

    def test_integer_comparison(self):
        expr = parse_filter("priority > 3")
        assert isinstance(expr.root, Comparison)
        assert expr.root.field == "priority"
        assert expr.root.operator == Operator.GT
        assert isinstance(expr.root.value, NumberValue)
        assert expr.root.value.value == 3

    def test_float_comparison(self):
        expr = parse_filter("score >= 4.5")
        assert isinstance(expr.root, Comparison)
        assert isinstance(expr.root.value, NumberValue)
        assert expr.root.value.value == 4.5

    def test_all_operators(self):
        ops = {
            "=": Operator.EQ,
            "!=": Operator.NE,
            "<": Operator.LT,
            ">": Operator.GT,
            "<=": Operator.LE,
            ">=": Operator.GE,
            ":": Operator.HAS,
        }
        for sym, op_enum in ops.items():
            expr = parse_filter(f"field {sym} 5")
            assert expr.root.operator == op_enum, f"Failed for {sym}"

    def test_and_expression(self):
        expr = parse_filter('status = "active" AND priority > 3')
        assert isinstance(expr.root, AndExpression)
        assert len(expr.root.children) == 2

    def test_or_expression(self):
        expr = parse_filter('status = "active" OR status = "pending"')
        assert isinstance(expr.root, OrExpression)
        assert len(expr.root.children) == 2

    def test_not_expression(self):
        expr = parse_filter('NOT status = "active"')
        assert isinstance(expr.root, NotExpression)
        assert isinstance(expr.root.child, Comparison)

    def test_minus_not(self):
        expr = parse_filter('-status = "active"')
        assert isinstance(expr.root, NotExpression)

    def test_has_wildcard(self):
        expr = parse_filter("category:*")
        assert isinstance(expr.root, Comparison)
        assert expr.root.operator == Operator.HAS
        assert isinstance(expr.root.value, WildcardValue)

    def test_nested_member(self):
        expr = parse_filter('category.name = "electronics"')
        assert isinstance(expr.root, Comparison)
        assert expr.root.field == "category.name"

    def test_parenthesized(self):
        expr = parse_filter('(status = "active" OR status = "pending") AND priority > 2')
        assert isinstance(expr.root, AndExpression)
        assert isinstance(expr.root.children[0], OrExpression)

    def test_implicit_and(self):
        expr = parse_filter('status = "active" priority > 2')
        assert isinstance(expr.root, AndExpression)

    def test_single_quotes(self):
        expr = parse_filter("status = 'active'")
        assert isinstance(expr.root, Comparison)
        assert expr.root.value.value == "active"
        assert expr.root.value.quote_char == "'"

    def test_complex_expression(self):
        expr = parse_filter(
            '(status = "active" OR status = "pending") AND priority <= 3 AND score >= 4.0'
        )
        assert isinstance(expr.root, AndExpression)
        assert len(expr.root.children) == 3

    def test_invalid_syntax(self):
        from sqlalchemy_aip160 import FilterError

        with pytest.raises(FilterError):
            parse_filter("status ==== active")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for FilterExpression.__str__."""

    def test_simple_equality(self):
        expr = parse_filter('status = "active"')
        assert str(expr) == 'status = "active"'

    def test_integer(self):
        expr = parse_filter("priority > 3")
        assert str(expr) == "priority > 3"

    def test_float(self):
        expr = parse_filter("score >= 4.5")
        assert str(expr) == "score >= 4.5"

    def test_has_wildcard(self):
        expr = parse_filter("category:*")
        assert str(expr) == "category:*"

    def test_and(self):
        expr = parse_filter('status = "active" AND priority > 3')
        assert str(expr) == 'status = "active" AND priority > 3'

    def test_or(self):
        expr = parse_filter('status = "active" OR status = "pending"')
        assert str(expr) == 'status = "active" OR status = "pending"'

    def test_not(self):
        expr = parse_filter('NOT status = "active"')
        assert str(expr) == 'NOT status = "active"'

    def test_nested_member(self):
        expr = parse_filter('category.name = "electronics"')
        assert str(expr) == 'category.name = "electronics"'

    def test_parenthesized_or_inside_and(self):
        """OR inside AND doesn't need parens in AIP-160 (OR has higher prec)."""
        expr = parse_filter('status = "active" AND a = "1" OR b = "2"')
        # OR has higher precedence, so the parse tree is:
        # AND(status="active", OR(a="1", b="2"))
        serialized = str(expr)
        assert 'status = "active" AND a = "1" OR b = "2"' == serialized

    def test_parenthesized_and_inside_or(self):
        """AND inside OR needs parens to preserve semantics."""
        # Build manually: OR(AND(a, b), c)
        a = Comparison(field="a", operator=Operator.EQ, value=StringValue("1"))
        b = Comparison(field="b", operator=Operator.EQ, value=StringValue("2"))
        c = Comparison(field="c", operator=Operator.EQ, value=StringValue("3"))
        expr = FilterExpression(root=OrExpression(children=[AndExpression(children=[a, b]), c]))
        assert str(expr) == '(a = "1" AND b = "2") OR c = "3"'

    def test_not_composite(self):
        """NOT of composite expression should parenthesize."""
        a = Comparison(field="a", operator=Operator.EQ, value=StringValue("1"))
        b = Comparison(field="b", operator=Operator.EQ, value=StringValue("2"))
        expr = FilterExpression(root=NotExpression(child=AndExpression(children=[a, b])))
        assert str(expr) == 'NOT (a = "1" AND b = "2")'

    def test_round_trip_complex(self):
        original = '(status = "active" OR status = "pending") AND priority > 2'
        expr = parse_filter(original)
        # Re-parse the serialized form and compare structure
        reparsed = parse_filter(str(expr))
        assert str(expr) == str(reparsed)

    def test_single_quote_preserved(self):
        expr = parse_filter("status = 'active'")
        assert str(expr) == "status = 'active'"

    def test_escaping_in_value(self):
        expr = FilterExpression(
            root=Comparison(
                field="name",
                operator=Operator.EQ,
                value=StringValue(value='say "hello"'),
            )
        )
        assert str(expr) == 'name = "say \\"hello\\""'

    def test_empty(self):
        assert str(FilterExpression(root=None)) == ""


# ---------------------------------------------------------------------------
# get_fields
# ---------------------------------------------------------------------------


class TestGetFields:

    def test_empty(self):
        assert parse_filter("").get_fields() == set()

    def test_single(self):
        assert parse_filter('status = "active"').get_fields() == {"status"}

    def test_and(self):
        assert parse_filter('status = "active" AND priority > 3').get_fields() == {
            "status",
            "priority",
        }

    def test_or(self):
        assert parse_filter('a = "1" OR b = "2"').get_fields() == {"a", "b"}

    def test_not(self):
        assert parse_filter('NOT status = "active"').get_fields() == {"status"}

    def test_dotted(self):
        expr = parse_filter('category.name = "electronics" AND priority > 3')
        assert expr.get_fields() == {"category.name", "priority"}

    def test_duplicates(self):
        expr = parse_filter('status = "active" AND status != "deleted"')
        assert expr.get_fields() == {"status"}


# ---------------------------------------------------------------------------
# rename_field
# ---------------------------------------------------------------------------


class TestRenameField:

    def test_simple(self):
        expr = parse_filter('kind = "issue"')
        expr.rename_field("kind", "kind_str")
        assert str(expr) == 'kind_str = "issue"'

    def test_value_untouched(self):
        expr = parse_filter('kind = "mankind"')
        expr.rename_field("kind", "kind_str")
        assert str(expr) == 'kind_str = "mankind"'
        assert expr.root.value.value == "mankind"

    def test_dotted_prefix(self):
        expr = parse_filter('category.name = "electronics"')
        expr.rename_field("category", "cat")
        assert str(expr) == 'cat.name = "electronics"'

    def test_in_and(self):
        expr = parse_filter('kind = "issue" AND source_id = "abc"')
        expr.rename_field("kind", "kind_str")
        assert str(expr) == 'kind_str = "issue" AND source_id = "abc"'

    def test_no_match(self):
        expr = parse_filter('status = "active"')
        expr.rename_field("kind", "kind_str")
        assert str(expr) == 'status = "active"'

    def test_empty(self):
        expr = parse_filter("")
        expr.rename_field("kind", "kind_str")  # should not raise
        assert str(expr) == ""

    def test_multiple_occurrences(self):
        expr = parse_filter('kind = "a" AND kind != "b"')
        expr.rename_field("kind", "type")
        assert expr.get_fields() == {"type"}


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:

    def test_single_clause(self):
        expr = parse_filter('status = "active"')
        expr.remove("status")
        assert expr.root is None
        assert str(expr) == ""

    def test_remove_from_and_middle(self):
        expr = parse_filter('a = "1" AND b = "2" AND c = "3"')
        expr.remove("b")
        assert str(expr) == 'a = "1" AND c = "3"'

    def test_remove_from_and_first(self):
        expr = parse_filter('a = "1" AND b = "2" AND c = "3"')
        expr.remove("a")
        assert str(expr) == 'b = "2" AND c = "3"'

    def test_remove_from_and_last(self):
        expr = parse_filter('a = "1" AND b = "2" AND c = "3"')
        expr.remove("c")
        assert str(expr) == 'a = "1" AND b = "2"'

    def test_remove_all_from_and(self):
        expr = parse_filter('a = "1" AND a = "2"')
        expr.remove("a")
        assert expr.root is None

    def test_remove_leaves_single_child_unwrapped(self):
        expr = parse_filter('a = "1" AND b = "2"')
        expr.remove("a")
        assert isinstance(expr.root, Comparison)
        assert str(expr) == 'b = "2"'

    def test_remove_from_or(self):
        expr = parse_filter('a = "1" OR b = "2" OR c = "3"')
        expr.remove("b")
        assert str(expr) == 'a = "1" OR c = "3"'

    def test_remove_from_not(self):
        expr = parse_filter('NOT a = "1"')
        expr.remove("a")
        assert expr.root is None

    def test_remove_nested(self):
        expr = parse_filter('(a = "1" AND b = "2") AND c = "3"')
        expr.remove("a")
        assert str(expr) == 'b = "2" AND c = "3"'

    def test_remove_no_match(self):
        expr = parse_filter('status = "active"')
        expr.remove("priority")
        assert str(expr) == 'status = "active"'

    def test_remove_empty(self):
        expr = parse_filter("")
        expr.remove("anything")
        assert expr.root is None

    def test_remove_dotted_field(self):
        expr = parse_filter('category.name = "electronics" AND priority > 3')
        expr.remove("category")
        assert str(expr) == "priority > 3"

    def test_remove_priority_from_edges(self):
        """Example from issue: remove from start/end of expression."""
        expr = parse_filter('kind = "acc:issue" AND source_id = "abc" AND priority > 3')
        expr.remove("priority")
        assert str(expr) == 'kind = "acc:issue" AND source_id = "abc"'

        expr2 = parse_filter('priority > 3 AND kind = "acc:issue" AND source_id = "abc"')
        expr2.remove("priority")
        assert str(expr2) == 'kind = "acc:issue" AND source_id = "abc"'


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class TestExtract:

    def test_extract_single(self):
        expr = parse_filter('label = "safety" AND status = "active"')
        extracted = expr.extract("label")
        assert len(extracted) == 1
        assert extracted[0].field == "label"
        assert extracted[0].value.value == "safety"
        assert str(expr) == 'status = "active"'

    def test_extract_multiple(self):
        expr = parse_filter('label = "safety" AND status = "active" AND label = "cost"')
        extracted = expr.extract("label")
        assert len(extracted) == 2
        assert [c.value.value for c in extracted] == ["safety", "cost"]
        assert str(expr) == 'status = "active"'

    def test_extract_all(self):
        expr = parse_filter('label = "a" AND label = "b"')
        extracted = expr.extract("label")
        assert len(extracted) == 2
        assert expr.root is None

    def test_extract_none(self):
        expr = parse_filter('status = "active"')
        extracted = expr.extract("label")
        assert len(extracted) == 0
        assert str(expr) == 'status = "active"'

    def test_extract_empty(self):
        expr = parse_filter("")
        assert expr.extract("label") == []

    def test_extract_from_or(self):
        expr = parse_filter('a = "1" OR b = "2" OR c = "3"')
        extracted = expr.extract("b")
        assert len(extracted) == 1
        assert str(expr) == 'a = "1" OR c = "3"'

    def test_extract_from_not(self):
        expr = parse_filter('NOT label = "bad"')
        extracted = expr.extract("label")
        assert len(extracted) == 1
        assert expr.root is None

    def test_extract_and_rebuild(self):
        """Example from issue: extract kind, change value, re-combine."""
        expr = parse_filter('kind = "acc:issue" AND source_id = "abc"')
        kind_clauses = expr.extract("kind")
        assert kind_clauses[0].value.value == "acc:issue"

        replacement = FilterBuilder().add("kind", "=", "acc:meeting").build()
        combined = expr & replacement
        assert 'source_id = "abc"' in str(combined)
        assert 'kind = "acc:meeting"' in str(combined)


# ---------------------------------------------------------------------------
# Combining (__and__, __or__, __invert__)
# ---------------------------------------------------------------------------


class TestCombining:

    def test_and(self):
        a = parse_filter('status = "active"')
        b = parse_filter("priority > 3")
        combined = a & b
        assert isinstance(combined.root, AndExpression)
        assert str(combined) == 'status = "active" AND priority > 3'

    def test_and_with_empty_left(self):
        a = parse_filter("")
        b = parse_filter('status = "active"')
        combined = a & b
        assert str(combined) == 'status = "active"'

    def test_and_with_empty_right(self):
        a = parse_filter('status = "active"')
        b = parse_filter("")
        combined = a & b
        assert str(combined) == 'status = "active"'

    def test_or(self):
        a = parse_filter('status = "active"')
        b = parse_filter('status = "pending"')
        combined = a | b
        assert isinstance(combined.root, OrExpression)
        assert str(combined) == 'status = "active" OR status = "pending"'

    def test_or_with_empty_is_match_all(self):
        a = parse_filter('status = "active"')
        b = parse_filter("")
        assert (a | b).root is None
        assert (b | a).root is None

    def test_invert(self):
        a = parse_filter('status = "active"')
        inverted = ~a
        assert isinstance(inverted.root, NotExpression)
        assert str(inverted) == 'NOT status = "active"'

    def test_invert_empty(self):
        a = parse_filter("")
        assert (~a).root is None

    def test_bool(self):
        assert bool(parse_filter('status = "active"')) is True
        assert bool(parse_filter("")) is False

    def test_chained_and_flattens(self):
        a = parse_filter('a = "1"')
        b = parse_filter('b = "2"')
        c = parse_filter('c = "3"')
        combined = a & b & c
        assert isinstance(combined.root, AndExpression)
        assert len(combined.root.children) == 3

    def test_and_nodes_are_independent_copies(self):
        a = parse_filter('status = "active"')
        b = parse_filter('priority > 3')
        combined = a & b
        # Mutating original expression should not affect combined
        a.rename_field("status", "state")
        assert str(combined) == 'status = "active" AND priority > 3'

    def test_or_nodes_are_independent_copies(self):
        a = parse_filter('status = "active"')
        b = parse_filter('status = "pending"')
        combined = a | b
        # Mutating original expression should not affect combined
        a.rename_field("status", "state")
        assert str(combined) == 'status = "active" OR status = "pending"'

    def test_invert_node_is_independent_copy(self):
        a = parse_filter('status = "active"')
        inverted = ~a
        # Mutating original expression should not affect inverted
        a.rename_field("status", "state")
        assert str(inverted) == 'NOT status = "active"'

    def test_and_with_empty_left_is_independent_copy(self):
        a = parse_filter('')
        b = parse_filter('status = "active"')
        combined = a & b
        b.rename_field("status", "state")
        assert str(combined) == 'status = "active"'

    def test_and_with_empty_right_is_independent_copy(self):
        a = parse_filter('status = "active"')
        b = parse_filter('')
        combined = a & b
        a.rename_field("status", "state")
        assert str(combined) == 'status = "active"'


# ---------------------------------------------------------------------------
# FilterBuilder
# ---------------------------------------------------------------------------


class TestFilterBuilder:

    def test_single_comparison(self):
        f = FilterBuilder().add("kind", "=", "issue")
        assert str(f) == 'kind = "issue"'

    def test_multiple_comparisons(self):
        f = FilterBuilder()
        f.add("kind", "=", "issue")
        f.add("source_id", "=", "abc")
        assert str(f) == 'kind = "issue" AND source_id = "abc"'

    def test_fluent(self):
        result = str(
            FilterBuilder()
            .add("a", "=", "1")
            .add("b", ">", 3)
            .add("c", "<=", 5.5)
        )
        assert result == 'a = "1" AND b > 3 AND c <= 5.5'

    def test_empty(self):
        assert str(FilterBuilder()) == ""

    def test_build_returns_filter_expression(self):
        f = FilterBuilder().add("a", "=", "1")
        expr = f.build()
        assert isinstance(expr, FilterExpression)

    def test_boolean_value(self):
        f = FilterBuilder().add("active", "=", True)
        assert str(f) == 'active = "true"'

    def test_operator_enum(self):
        f = FilterBuilder().add("priority", Operator.GT, 5)
        assert str(f) == "priority > 5"

    def test_invalid_operator(self):
        with pytest.raises(ValueError):
            FilterBuilder().add("a", "~", "b")

    def test_build_and_combine(self):
        user = parse_filter('status = "active"')
        server = FilterBuilder().add("org_id", "=", "org-123").build()
        combined = user & server
        assert 'status = "active"' in str(combined)
        assert 'org_id = "org-123"' in str(combined)

    def test_build_is_independent_copy(self):
        """Build returns independent copies so mutating doesn't corrupt the builder."""
        builder = FilterBuilder().add("a", "=", "x")
        expr1 = builder.build()
        expr1.rename_field("a", "b")
        expr2 = builder.build()
        assert str(expr2) == 'a = "x"', "Builder state should not be affected by mutations"


# ---------------------------------------------------------------------------
# Integration: parse → manipulate → serialize (no DB needed)
# ---------------------------------------------------------------------------


class TestIntegrationNoDB:

    def test_extract_labels_example(self):
        """The motivating example from the issue."""
        expr = parse_filter('label = "safety" AND status = "active" AND label = "cost"')
        labels = expr.extract("label")
        assert [c.value.value for c in labels] == ["safety", "cost"]
        assert str(expr) == 'status = "active"'

    def test_rename_kind_example(self):
        """Field renaming example from the issue."""
        expr = parse_filter('kind = "mankind" AND source_id = "abc"')
        expr.rename_field("kind", "kind_str")
        assert str(expr) == 'kind_str = "mankind" AND source_id = "abc"'

    def test_change_value_example(self):
        """Changing a value by extract + rebuild."""
        expr = parse_filter('kind = "acc:issue" AND source_id = "abc"')
        expr.extract("kind")
        replacement = FilterBuilder().add("kind", "=", "acc:meeting").build()
        combined = expr & replacement
        assert 'source_id = "abc"' in str(combined)
        assert 'kind = "acc:meeting"' in str(combined)

    def test_combine_user_and_server(self):
        user = parse_filter('status = "active" AND priority > 3')
        server = parse_filter('org_id = "org-123"')
        combined = user & server
        result = str(combined)
        assert "status" in result
        assert "priority" in result
        assert "org_id" in result

    def test_full_round_trip(self):
        """Parse → extract → rename → combine → serialize → re-parse."""
        expr = parse_filter('label = "x" AND kind = "issue" AND priority > 3')
        labels = expr.extract("label")
        assert len(labels) == 1
        expr.rename_field("kind", "kind_str")
        server = FilterBuilder().add("org", "=", "123").build()
        final = expr & server
        serialized = str(final)
        reparsed = parse_filter(serialized)
        assert reparsed.get_fields() == {"kind_str", "priority", "org"}
