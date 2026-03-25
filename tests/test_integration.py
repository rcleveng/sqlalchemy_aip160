"""Integration tests: parse → manipulate → apply_filter with real DB queries."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import Integer, String, Float, Boolean, DateTime, create_engine, select
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
    s.add_all([
        Item(id=1, name="A", kind="issue", kind_str="issue", status="active",
             priority=1, score=4.5, is_active=True, label="safety"),
        Item(id=2, name="B", kind="meeting", kind_str="meeting", status="active",
             priority=3, score=3.0, is_active=True, label="cost"),
        Item(id=3, name="C", kind="issue", kind_str="issue", status="inactive",
             priority=5, score=5.0, is_active=False, label=None),
        Item(id=4, name="D", kind="meeting", kind_str="meeting", status="pending",
             priority=2, score=2.5, is_active=True, label="safety"),
    ])
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
        server = parse_filter('is_active = true')
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
