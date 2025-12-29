"""Tests for AIP-160 filter implementation."""

import pytest
from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Integer, Float, Boolean, DateTime, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, synonym
from sqlalchemy.pool import StaticPool

from sqlalchemy_aip160.aip160_filter import (
    apply_filter,
    build_filter_expression,
    parse_filter,
    FilterError,
    InvalidFieldError,
)


# Test model setup - use non-Test prefix to avoid pytest collection warnings
class FilterTestBase(DeclarativeBase):
    pass


class SampleModel(FilterTestBase):
    """Simple test model for filter testing."""

    __tablename__ = "test_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50))
    priority: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)


class SampleModelWithSynonym(FilterTestBase):
    """Test model with column synonym for alias testing."""

    __tablename__ = "test_items_with_synonym"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_name: Mapped[str] = mapped_column("category", String(50))
    category = synonym("category_name")  # Alias


@pytest.fixture(scope="module")
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    FilterTestBase.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="module")
def sample_data(engine):
    """Insert sample data for testing."""
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    items = [
        SampleModel(
            id=1,
            name="Widget A",
            status="active",
            priority=1,
            score=4.5,
            is_active=True,
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            category="electronics",
        ),
        SampleModel(
            id=2,
            name="Widget B",
            status="inactive",
            priority=2,
            score=3.0,
            is_active=False,
            created_at=datetime(2024, 2, 20, tzinfo=timezone.utc),
            category="electronics",
        ),
        SampleModel(
            id=3,
            name="Gadget X",
            status="active",
            priority=3,
            score=4.8,
            is_active=True,
            created_at=datetime(2024, 3, 10, tzinfo=timezone.utc),
            category="hardware",
        ),
        SampleModel(
            id=4,
            name="Gadget Y",
            status="pending",
            priority=1,
            score=2.5,
            is_active=True,
            created_at=datetime(2024, 4, 5, tzinfo=timezone.utc),
            category=None,
        ),
        SampleModel(
            id=5,
            name="Tool Z",
            status="active",
            priority=5,
            score=5.0,
            is_active=True,
            created_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
            category="hardware",
        ),
    ]

    session.add_all(items)
    session.commit()
    session.close()

    return items


@pytest.fixture
def session(engine, sample_data):
    """Create a session for each test."""
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


class TestParseFilter:
    """Tests for filter parsing."""

    def test_parse_empty_filter(self):
        """Empty filter should parse to None."""
        assert parse_filter("") is None
        assert parse_filter("   ") is None
        assert parse_filter(None) is None

    def test_parse_simple_equality(self):
        """Simple equality should parse."""
        tree = parse_filter('status = "active"')
        assert tree is not None

    def test_parse_comparison_operators(self):
        """All comparison operators should parse."""
        operators = ["=", "!=", "<", ">", "<=", ">=", ":"]
        for op in operators:
            tree = parse_filter(f"priority {op} 5")
            assert tree is not None, f"Operator {op} failed to parse"

    def test_parse_logical_operators(self):
        """Logical operators should parse."""
        tree = parse_filter('status = "active" AND priority > 3')
        assert tree is not None

        tree = parse_filter('status = "active" OR priority > 3')
        assert tree is not None

    def test_parse_not_operator(self):
        """NOT operator should parse."""
        tree = parse_filter('NOT status = "active"')
        assert tree is not None

        tree = parse_filter('-status = "active"')
        assert tree is not None

    def test_parse_parentheses(self):
        """Parenthesized expressions should parse."""
        tree = parse_filter(
            '(status = "active" OR status = "pending") AND priority > 2'
        )
        assert tree is not None

    def test_parse_invalid_syntax(self):
        """Invalid syntax should raise FilterError."""
        with pytest.raises(FilterError):
            parse_filter("status ==== active")

        with pytest.raises(FilterError):
            parse_filter("(status = active")


class TestBuildFilterExpression:
    """Tests for building SQLAlchemy expressions."""

    def test_empty_filter_returns_true(self):
        """Empty filter should return True (match everything)."""
        expr = build_filter_expression(SampleModel, "")
        assert expr is True

    def test_equality_filter(self):
        """Equality filter should build correct expression."""
        expr = build_filter_expression(SampleModel, 'status = "active"')
        assert expr is not None
        # Just verify it builds without error

    def test_invalid_field_raises_error(self):
        """Invalid field should raise InvalidFieldError."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(SampleModel, 'nonexistent = "value"')
        assert "nonexistent" in str(exc_info.value)

    def test_allowed_fields_restriction(self):
        """Allowed fields should restrict which fields can be filtered."""
        # Should succeed with allowed field
        expr = build_filter_expression(
            SampleModel, 'status = "active"', allowed_fields={"status", "priority"}
        )
        assert expr is not None

        # Should fail with non-allowed field
        with pytest.raises(InvalidFieldError):
            build_filter_expression(
                SampleModel, 'name = "Widget"', allowed_fields={"status", "priority"}
            )


class TestApplyFilter:
    """Tests for applying filters to queries."""

    def test_empty_filter_returns_all(self, session):
        """Empty filter should return all items."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 5

    def test_none_filter_returns_all(self, session):
        """None filter should return all items."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, None)
        results = session.execute(filtered).scalars().all()
        assert len(results) == 5

    def test_equality_string_filter(self, session):
        """String equality filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, 'status = "active"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 3
        assert all(r.status == "active" for r in results)

    def test_equality_integer_filter(self, session):
        """Integer equality filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "priority = 1")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.priority == 1 for r in results)

    def test_not_equals_filter(self, session):
        """Not equals filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, 'status != "active"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.status != "active" for r in results)

    def test_less_than_filter(self, session):
        """Less than filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "priority < 3")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 3
        assert all(r.priority < 3 for r in results)

    def test_greater_than_filter(self, session):
        """Greater than filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "priority > 2")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.priority > 2 for r in results)

    def test_less_than_equals_filter(self, session):
        """Less than or equals filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "priority <= 2")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 3
        assert all(r.priority <= 2 for r in results)

    def test_greater_than_equals_filter(self, session):
        """Greater than or equals filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "priority >= 3")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.priority >= 3 for r in results)

    def test_float_comparison(self, session):
        """Float comparison should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "score >= 4.5")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 3
        assert all(r.score >= 4.5 for r in results)

    def test_and_filter(self, session):
        """AND filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(
            query, SampleModel, 'status = "active" AND priority > 2'
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.status == "active" and r.priority > 2 for r in results)

    def test_or_filter(self, session):
        """OR filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(
            query, SampleModel, 'status = "inactive" OR status = "pending"'
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        statuses = {r.status for r in results}
        assert statuses == {"inactive", "pending"}

    def test_not_filter(self, session):
        """NOT filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, 'NOT status = "active"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.status != "active" for r in results)

    def test_minus_not_filter(self, session):
        """Minus (-) as NOT should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, '-status = "active"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.status != "active" for r in results)

    def test_parentheses_precedence(self, session):
        """Parentheses should control precedence."""
        query = select(SampleModel)
        # Without parentheses: status = "active" AND (priority = 1 OR priority = 5)
        # because OR has higher precedence in AIP-160
        filtered = apply_filter(
            query, SampleModel, 'status = "active" AND priority = 1 OR priority = 5'
        )
        results = session.execute(filtered).scalars().all()
        # This matches: active with priority 1 OR priority 5
        assert len(results) >= 1

        # With parentheses
        filtered = apply_filter(
            query, SampleModel, '(status = "active" AND priority = 1) OR priority = 5'
        )
        results2 = session.execute(filtered).scalars().all()
        assert len(results2) >= 1

    def test_wildcard_filter(self, session):
        """Wildcard pattern matching should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, 'name = "Widget*"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.name.startswith("Widget") for r in results)

    def test_wildcard_suffix(self, session):
        """Wildcard at start should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, 'name = "*A"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0].name == "Widget A"

    def test_has_operator_presence(self, session):
        """Has operator for presence check should work."""
        query = select(SampleModel)
        # Check for non-null category
        filtered = apply_filter(query, SampleModel, "category:*")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 4
        assert all(r.category is not None for r in results)

    def test_complex_filter(self, session):
        """Complex nested filter should work."""
        query = select(SampleModel)
        filtered = apply_filter(
            query,
            SampleModel,
            '(status = "active" OR status = "pending") AND priority <= 3 AND score >= 4.0',
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) == 2
        for r in results:
            assert r.status in ("active", "pending")
            assert r.priority <= 3
            assert r.score >= 4.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_quotes_string(self, session):
        """Single-quoted strings should work."""
        query = select(SampleModel)
        filtered = apply_filter(query, SampleModel, "status = 'active'")
        results = session.execute(filtered).scalars().all()
        assert len(results) == 3

    def test_implicit_and(self, session):
        """Adjacent terms should be implicitly ANDed."""
        query = select(SampleModel)
        # "status = active priority > 2" is equivalent to "status = active AND priority > 2"
        filtered = apply_filter(query, SampleModel, 'status = "active" priority > 2')
        results = session.execute(filtered).scalars().all()
        assert all(r.status == "active" and r.priority > 2 for r in results)

    def test_boolean_field(self, session):
        """Boolean field filtering should work."""
        query = select(SampleModel)
        # SQLite stores booleans as 0/1, so we test with the actual value
        filtered = apply_filter(query, SampleModel, "is_active = true")
        results = session.execute(filtered).scalars().all()
        assert all(r.is_active for r in results)

    def test_function_not_supported(self):
        """Function calls should raise appropriate error."""
        with pytest.raises(FilterError) as exc_info:
            build_filter_expression(SampleModel, "startsWith(name, 'Widget')")
        assert "not supported" in str(exc_info.value).lower()

    def test_nested_field_not_supported(self):
        """Nested field access should raise appropriate error."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(SampleModel, 'user.name = "test"')
        assert "not yet supported" in str(exc_info.value).lower()

    def test_bare_value_not_supported(self):
        """Bare values without field should raise appropriate error."""
        with pytest.raises(FilterError) as exc_info:
            build_filter_expression(SampleModel, "active")
        assert "bare value" in str(exc_info.value).lower()


class TestAllowedFields:
    """Tests for allowed_fields functionality."""

    def test_allowed_fields_whitelist(self, session):
        """Only whitelisted fields should be filterable."""
        allowed = {"status", "priority"}

        # These should work
        query = select(SampleModel)
        filtered = apply_filter(
            query, SampleModel, 'status = "active"', allowed_fields=allowed
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) > 0

        filtered = apply_filter(
            query, SampleModel, "priority > 2", allowed_fields=allowed
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) > 0

        # This should fail
        with pytest.raises(InvalidFieldError):
            apply_filter(query, SampleModel, 'name = "Widget"', allowed_fields=allowed)

    def test_allowed_fields_empty_allows_nothing(self, session):
        """Empty allowed_fields set should allow no fields."""
        with pytest.raises(InvalidFieldError):
            apply_filter(
                select(SampleModel),
                SampleModel,
                'status = "active"',
                allowed_fields=set(),
            )


class TestSynonymSupport:
    """Tests for SQLAlchemy synonym support."""

    def test_synonym_in_column_map(self):
        """Synonym should be included in column map."""
        from sqlalchemy_aip160.aip160_filter import SQLAlchemyTransformer

        transformer = SQLAlchemyTransformer(SampleModelWithSynonym)
        assert "category" in transformer._column_map
        assert "category_name" in transformer._column_map

    def test_filter_by_synonym(self, session):
        """Should be able to filter using synonym name."""
        # Insert test data
        item = SampleModelWithSynonym(id=100, category_name="test_cat")
        session.add(item)
        session.commit()

        query = select(SampleModelWithSynonym)
        filtered = apply_filter(query, SampleModelWithSynonym, 'category = "test_cat"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0].category == "test_cat"

    def test_filter_by_target_column(self, session):
        """Should also be able to filter using target column name."""
        query = select(SampleModelWithSynonym)
        filtered = apply_filter(
            query, SampleModelWithSynonym, 'category_name = "test_cat"'
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0].category_name == "test_cat"

    def test_synonym_respects_allowed_fields(self):
        """Synonym should respect allowed_fields whitelist."""
        # category allowed, should work
        expr = build_filter_expression(
            SampleModelWithSynonym, 'category = "foo"', allowed_fields={"category"}
        )
        assert expr is not None

        # category not allowed, should fail
        with pytest.raises(InvalidFieldError):
            build_filter_expression(
                SampleModelWithSynonym, 'category = "foo"', allowed_fields={"id"}
            )
