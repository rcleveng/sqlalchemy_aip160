"""Integration tests: parse → manipulate → apply_filter with real DB queries."""

import pytest
from sqlalchemy import Integer, String, Float, Boolean, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from sqlalchemy_aip160 import (
    FilterBuilder,
    apply_filter,
    parse_filter,
)


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(50))
    kind_str: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50))
    priority: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean)
    label: Mapped[str | None] = mapped_column(String(50), nullable=True)


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    s = Session()
    s.add_all(
        [
            Item(
                id=1,
                name="A",
                kind="issue",
                kind_str="issue",
                status="active",
                priority=1,
                score=4.5,
                is_active=True,
                label="safety",
            ),
            Item(
                id=2,
                name="B",
                kind="meeting",
                kind_str="meeting",
                status="active",
                priority=3,
                score=3.0,
                is_active=True,
                label="cost",
            ),
            Item(
                id=3,
                name="C",
                kind="issue",
                kind_str="issue",
                status="inactive",
                priority=5,
                score=5.0,
                is_active=False,
                label=None,
            ),
            Item(
                id=4,
                name="D",
                kind="meeting",
                kind_str="meeting",
                status="pending",
                priority=2,
                score=2.5,
                is_active=True,
                label="safety",
            ),
        ]
    )
    s.commit()
    s.close()
    return eng


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


class TestApplyFilterWithExpression:
    def test_pass_filter_expression_directly(self, session):
        expr = parse_filter('status = "active"')
        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 2
        assert all(r.status == "active" for r in results)

    def test_rename_field_then_apply(self, session):
        expr = parse_filter('kind = "issue"')
        expr.rename_field("kind", "kind_str")
        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 2
        assert all(r.kind_str == "issue" for r in results)

    def test_remove_clause_then_apply(self, session):
        expr = parse_filter('status = "active" AND priority > 2')
        expr.remove("priority")
        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 2
        assert all(r.status == "active" for r in results)

    def test_extract_and_apply_remainder(self, session):
        expr = parse_filter('label = "safety" AND status = "active"')
        labels = expr.extract("label")
        assert labels[0].value.value == "safety"
        # Remaining filter: status = "active"
        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 2

    def test_builder_then_apply(self, session):
        f = FilterBuilder()
        f.add("kind", "=", "issue").add("is_active", "=", True)
        query = apply_filter(select(Item), Item, f.build())
        results = session.execute(query).scalars().all()
        assert len(results) == 1
        assert results[0].name == "A"

    def test_combine_then_apply(self, session):
        user = parse_filter('kind = "issue"')
        server = parse_filter("is_active = true")
        combined = user & server
        query = apply_filter(select(Item), Item, combined)
        results = session.execute(query).scalars().all()
        assert len(results) == 1
        assert results[0].name == "A"

    def test_empty_expression_returns_all(self, session):
        expr = parse_filter("")
        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 4

    def test_remove_all_clauses_returns_all(self, session):
        expr = parse_filter('status = "active"')
        expr.remove("status")
        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 4

    def test_change_value_workflow(self, session):
        """Extract kind="issue", replace with kind="meeting", apply."""
        expr = parse_filter('kind = "issue" AND priority > 1')
        expr.extract("kind")
        replacement = FilterBuilder().add("kind", "=", "meeting").build()
        final = expr & replacement
        query = apply_filter(select(Item), Item, final)
        results = session.execute(query).scalars().all()
        assert len(results) == 2
        assert all(r.kind == "meeting" for r in results)
        assert all(r.priority > 1 for r in results)


# ---------------------------------------------------------------------------
# Real-world pattern: extract pseudo-field, use its value, apply remainder
# ---------------------------------------------------------------------------


def _extract_starred_filter(
    filter_expr: str | None,
) -> tuple[bool | None, str | None]:
    """Extract ``starred=true/false`` from an AIP-160 filter string.

    This mirrors the real-world helper that replaces fragile regex extraction.

    Returns:
        (starred_value, remaining_filter)
    """
    if not filter_expr:
        return None, filter_expr

    expr = parse_filter(filter_expr)
    clauses = expr.extract("starred")

    if not clauses:
        return None, filter_expr

    starred_value = clauses[0].value.value.lower() == "true"
    remaining = str(expr) or None

    return starred_value, remaining


class TestExtractPseudoFieldPattern:
    """Tests that mirror real usage: extract a pseudo-field boolean,
    then apply the remaining filter to a DB query."""

    def test_starred_true_with_other_clauses(self, session):
        starred, remaining = _extract_starred_filter(
            'starred = "true" AND status = "active"'
        )
        assert starred is True
        assert remaining == 'status = "active"'
        query = apply_filter(select(Item), Item, remaining)
        results = session.execute(query).scalars().all()
        assert len(results) == 2
        assert all(r.status == "active" for r in results)

    def test_starred_false_with_other_clauses(self, session):
        starred, remaining = _extract_starred_filter(
            'status = "active" AND starred = "false" AND priority > 1'
        )
        assert starred is False
        assert remaining == 'status = "active" AND priority > 1'
        query = apply_filter(select(Item), Item, remaining)
        results = session.execute(query).scalars().all()
        assert len(results) == 1
        assert results[0].priority > 1

    def test_starred_unquoted_true(self, session):
        starred, remaining = _extract_starred_filter(
            'starred = true AND kind = "issue"'
        )
        assert starred is True
        assert remaining == 'kind = "issue"'
        query = apply_filter(select(Item), Item, remaining)
        results = session.execute(query).scalars().all()
        assert all(r.kind == "issue" for r in results)

    def test_starred_only(self, session):
        starred, remaining = _extract_starred_filter('starred = "true"')
        assert starred is True
        assert remaining is None
        # None remaining → return all rows
        query = apply_filter(select(Item), Item, remaining)
        results = session.execute(query).scalars().all()
        assert len(results) == 4

    def test_no_starred(self, session):
        starred, remaining = _extract_starred_filter('status = "active"')
        assert starred is None
        assert remaining == 'status = "active"'
        query = apply_filter(select(Item), Item, remaining)
        results = session.execute(query).scalars().all()
        assert len(results) == 2

    def test_none_input(self):
        starred, remaining = _extract_starred_filter(None)
        assert starred is None
        assert remaining is None

    def test_empty_input(self):
        starred, remaining = _extract_starred_filter("")
        assert starred is None
        assert remaining == ""

    def test_starred_at_start(self, session):
        starred, remaining = _extract_starred_filter(
            'starred = "false" AND kind = "meeting" AND priority > 1'
        )
        assert starred is False
        assert remaining == 'kind = "meeting" AND priority > 1'
        query = apply_filter(select(Item), Item, remaining)
        results = session.execute(query).scalars().all()
        assert all(r.kind == "meeting" and r.priority > 1 for r in results)

    def test_starred_at_end(self, session):
        starred, remaining = _extract_starred_filter(
            'kind = "issue" AND priority > 1 AND starred = "true"'
        )
        assert starred is True
        assert remaining == 'kind = "issue" AND priority > 1'

    def test_multiple_pseudo_fields(self, session):
        """Extract label AND starred from same filter, apply remainder."""
        filter_str = 'label = "safety" AND starred = "true" AND status = "active"'
        expr = parse_filter(filter_str)

        labels = expr.extract("label")
        starred_clauses = expr.extract("starred")

        assert [c.value.value for c in labels] == ["safety"]
        assert starred_clauses[0].value.value.lower() == "true"
        assert str(expr) == 'status = "active"'

        query = apply_filter(select(Item), Item, expr)
        results = session.execute(query).scalars().all()
        assert len(results) == 2
        assert all(r.status == "active" for r in results)
