"""Tests for AIP-160 filter implementation."""

import pytest
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    select,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, synonym, relationship
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
    """Test model with column synonym for alias testing.

    Uses `_internal_category` as the Python attribute name and `category` as
    the public synonym that users filter by.
    """

    __tablename__ = "test_items_with_synonym"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    _internal_category: Mapped[str] = mapped_column("category", String(50))
    category = synonym("_internal_category")  # Public alias for filtering


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
        """Empty filter should return True (match everything) with empty joins."""
        expr, joins = build_filter_expression(SampleModel, "")
        assert expr is True
        assert joins == []

    def test_equality_filter(self):
        """Equality filter should build correct expression."""
        expr, joins = build_filter_expression(SampleModel, 'status = "active"')
        assert expr is not None
        assert joins == []  # No joins for simple column access

    def test_invalid_field_raises_error(self):
        """Invalid field should raise InvalidFieldError."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(SampleModel, 'nonexistent = "value"')
        assert "nonexistent" in str(exc_info.value)

    def test_allowed_fields_restriction(self):
        """Allowed fields should restrict which fields can be filtered."""
        # Should succeed with allowed field
        expr, joins = build_filter_expression(
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

    def test_nested_field_non_relationship_fails(self):
        """Nested field access on non-relationship should raise error."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(SampleModel, 'status.name = "test"')
        # 'status' is a string column, not a relationship
        assert "not a relationship" in str(exc_info.value).lower()

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
        assert "category" in transformer._column_map  # The synonym
        assert "_internal_category" in transformer._column_map  # The actual attribute

    def test_filter_by_synonym(self, session):
        """Should be able to filter using synonym name."""
        # Insert test data
        item = SampleModelWithSynonym(id=100, _internal_category="test_cat")
        session.add(item)
        session.commit()

        query = select(SampleModelWithSynonym)
        filtered = apply_filter(query, SampleModelWithSynonym, 'category = "test_cat"')
        results = session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0].category == "test_cat"

    def test_filter_by_target_column(self, session):
        """Should also be able to filter using the underlying attribute name."""
        query = select(SampleModelWithSynonym)
        filtered = apply_filter(
            query, SampleModelWithSynonym, '_internal_category = "test_cat"'
        )
        results = session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0]._internal_category == "test_cat"

    def test_synonym_respects_allowed_fields(self):
        """Synonym should respect allowed_fields whitelist."""
        # category allowed, should work
        expr, joins = build_filter_expression(
            SampleModelWithSynonym, 'category = "foo"', allowed_fields={"category"}
        )
        assert expr is not None
        assert joins == []

        # category not allowed, should fail
        with pytest.raises(InvalidFieldError):
            build_filter_expression(
                SampleModelWithSynonym, 'category = "foo"', allowed_fields={"id"}
            )

    def test_multiple_synonyms_same_column(self):
        """Multiple synonyms can point to the same underlying column."""
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, synonym

        class TempBase(DeclarativeBase):
            pass

        class ModelWithMultipleSynonyms(TempBase):
            __tablename__ = "multi_syn"
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            _status: Mapped[str] = mapped_column("status", String(50))
            status = synonym("_status")
            state = synonym("_status")  # Another alias for the same column

        from sqlalchemy_aip160.aip160_filter import SQLAlchemyTransformer

        transformer = SQLAlchemyTransformer(ModelWithMultipleSynonyms)
        # All three names should be in the column map
        assert "_status" in transformer._column_map
        assert "status" in transformer._column_map
        assert "state" in transformer._column_map
        # Both synonyms should resolve to the same underlying column
        assert transformer._column_map["status"] is transformer._column_map["state"]

    def test_synonym_with_allowed_fields_excludes_target(self):
        """Synonym can be allowed even if the underlying attribute is not."""
        # Only allow the synonym, not the internal attribute
        expr, joins = build_filter_expression(
            SampleModelWithSynonym, 'category = "foo"', allowed_fields={"category"}
        )
        assert expr is not None
        assert joins == []

        # The internal attribute should not be accessible
        with pytest.raises(InvalidFieldError):
            build_filter_expression(
                SampleModelWithSynonym,
                '_internal_category = "foo"',
                allowed_fields={"category"},
            )


# Models for relationship testing
class RelationshipTestBase(DeclarativeBase):
    pass


class Store(RelationshipTestBase):
    """Store model for multi-level relationship testing."""

    __tablename__ = "rel_stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    region: Mapped[str] = mapped_column(String(50))


class Category(RelationshipTestBase):
    """Category model for relationship testing."""

    __tablename__ = "rel_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    store_id: Mapped[int | None] = mapped_column(
        ForeignKey("rel_stores.id"), nullable=True
    )
    store: Mapped["Store | None"] = relationship()


class Item(RelationshipTestBase):
    """Item model with relationship for testing."""

    __tablename__ = "rel_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    category_id: Mapped[int] = mapped_column(ForeignKey("rel_categories.id"))
    category: Mapped[Category] = relationship()


@pytest.fixture(scope="module")
def rel_engine():
    """Create an in-memory SQLite engine for relationship testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    RelationshipTestBase.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="module")
def rel_sample_data(rel_engine):
    """Insert sample data for relationship testing."""
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=rel_engine)
    session = SessionLocal()

    # Create stores first
    main_store = Store(id=1, name="Main Store", region="west")
    outlet = Store(id=2, name="Outlet", region="east")
    session.add_all([main_store, outlet])
    session.flush()

    # Create categories with stores
    electronics = Category(id=1, name="electronics", is_active=True, store_id=1)
    hardware = Category(id=2, name="hardware", is_active=True, store_id=1)
    deprecated = Category(id=3, name="deprecated", is_active=False, store_id=None)
    session.add_all([electronics, hardware, deprecated])

    # Create items with categories
    items = [
        Item(id=1, name="Widget A", category_id=1),  # electronics -> Main Store
        Item(id=2, name="Widget B", category_id=1),  # electronics -> Main Store
        Item(id=3, name="Gadget X", category_id=2),  # hardware -> Main Store
        Item(id=4, name="Legacy Y", category_id=3),  # deprecated (no store)
    ]
    session.add_all(items)
    session.commit()
    session.close()


@pytest.fixture
def rel_session(rel_engine, rel_sample_data):
    """Create a session for each relationship test."""
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=rel_engine)
    session = SessionLocal()
    yield session
    session.close()


class TestRelationshipFiltering:
    """Tests for relationship traversal filtering."""

    def test_relationship_traversal_basic(self, rel_session):
        """Basic relationship traversal should work."""
        query = select(Item)
        filtered = apply_filter(query, Item, 'category.name = "electronics"')
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.category.name == "electronics" for r in results)

    def test_relationship_traversal_multiple_fields(self, rel_session):
        """Filtering by multiple fields on same relationship should work."""
        query = select(Item)
        filtered = apply_filter(
            query, Item, 'category.name = "electronics" AND category.is_active = true'
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(
            r.category.name == "electronics" and r.category.is_active for r in results
        )

    def test_relationship_join_deduplication(self):
        """Same relationship filtered twice should only produce one join."""
        expr, joins = build_filter_expression(
            Item, 'category.name = "electronics" AND category.is_active = true'
        )
        # Should only have one join even though we filter on two fields
        assert len(joins) == 1

    def test_relationship_with_other_filters(self, rel_session):
        """Relationship filter combined with direct field filter should work."""
        query = select(Item)
        filtered = apply_filter(
            query, Item, 'name = "Widget*" AND category.name = "electronics"'
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.name.startswith("Widget") for r in results)
        assert all(r.category.name == "electronics" for r in results)

    def test_invalid_relationship_field(self):
        """Invalid field on relationship should raise error."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(Item, 'category.nonexistent = "foo"')
        assert "nonexistent" in str(exc_info.value).lower()
        assert "Category" in str(exc_info.value)

    def test_non_relationship_field_with_dot(self):
        """Dotted access on non-relationship should raise error."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(Item, 'name.invalid = "foo"')
        assert "not a relationship" in str(exc_info.value).lower()

    def test_unknown_relationship(self):
        """Unknown relationship should raise error."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(Item, 'nonexistent.field = "foo"')
        assert "nonexistent" in str(exc_info.value).lower()

    def test_relationship_filter_inactive_category(self, rel_session):
        """Should be able to filter by inactive category."""
        query = select(Item)
        filtered = apply_filter(query, Item, "category.is_active = false")
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0].name == "Legacy Y"
        assert not results[0].category.is_active

    def test_multi_level_relationship_traversal(self, rel_session):
        """Deep relationship chains (e.g., category.store.name) should work."""
        query = select(Item)
        filtered = apply_filter(query, Item, 'category.store.name = "Main Store"')
        results = rel_session.execute(filtered).scalars().all()
        # Items 1, 2, 3 have categories with store "Main Store"; item 4 has no store
        assert len(results) == 3
        assert all(r.category.store is not None for r in results)
        assert all(r.category.store.name == "Main Store" for r in results)

    def test_multi_level_relationship_by_region(self, rel_session):
        """Can filter by deeply nested field (category.store.region)."""
        query = select(Item)
        filtered = apply_filter(query, Item, 'category.store.region = "west"')
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 3
        assert all(r.category.store.region == "west" for r in results)

    def test_multi_level_relationship_join_count(self):
        """Multi-level relationship should produce correct number of joins."""
        expr, joins = build_filter_expression(
            Item, 'category.store.name = "Main Store"'
        )
        # Should have two joins: Item->Category and Category->Store
        assert len(joins) == 2

    def test_multi_level_with_single_level_combined(self, rel_session):
        """Can combine multi-level and single-level relationship filters."""
        query = select(Item)
        filtered = apply_filter(
            query,
            Item,
            'category.name = "electronics" AND category.store.name = "Main Store"',
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.category.name == "electronics" for r in results)
        assert all(r.category.store.name == "Main Store" for r in results)

    def test_multi_level_invalid_intermediate_relationship(self):
        """Invalid intermediate relationship should raise error."""
        with pytest.raises(InvalidFieldError) as exc_info:
            build_filter_expression(Item, 'category.nonexistent.name = "foo"')
        assert "nonexistent" in str(exc_info.value).lower()


class TestFieldAliases:
    """Tests for field alias functionality."""

    def test_field_alias_basic(self, rel_session):
        """Basic field alias should resolve to relationship path."""
        query = select(Item)
        filtered = apply_filter(
            query,
            Item,
            'cat = "electronics"',
            field_aliases={"cat": "category.name"},
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.category.name == "electronics" for r in results)

    def test_field_alias_triggers_join(self):
        """Field alias to relationship path should trigger join."""
        expr, joins = build_filter_expression(
            Item,
            'cat = "electronics"',
            field_aliases={"cat": "category.name"},
        )
        assert len(joins) == 1

    def test_unaliased_field_still_works(self, rel_session):
        """Non-aliased fields should still work alongside aliases."""
        query = select(Item)
        filtered = apply_filter(
            query,
            Item,
            'name = "Widget*" AND cat = "electronics"',
            field_aliases={"cat": "category.name"},
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2
        assert all(r.name.startswith("Widget") for r in results)

    def test_alias_to_direct_column(self, rel_session):
        """Alias can map to a direct column name too."""
        query = select(Item)
        filtered = apply_filter(
            query,
            Item,
            'title = "Widget A"',
            field_aliases={"title": "name"},
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 1
        assert results[0].name == "Widget A"

    def test_multiple_aliases(self, rel_session):
        """Multiple aliases should work together."""
        query = select(Item)
        filtered = apply_filter(
            query,
            Item,
            'cat = "electronics" AND active = true',
            field_aliases={"cat": "category.name", "active": "category.is_active"},
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2

    def test_alias_with_allowed_fields(self, rel_session):
        """Alias name should be checked against allowed_fields."""
        query = select(Item)
        # "cat" is allowed, so this should work
        filtered = apply_filter(
            query,
            Item,
            'cat = "electronics"',
            allowed_fields={"cat", "name"},
            field_aliases={"cat": "category.name"},
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 2

    def test_alias_not_in_allowed_fields_fails(self):
        """Alias not in allowed_fields should fail."""
        with pytest.raises(InvalidFieldError):
            build_filter_expression(
                Item,
                'cat = "electronics"',
                allowed_fields={"name"},  # "cat" not allowed
                field_aliases={"cat": "category.name"},
            )

    def test_empty_filter_with_aliases(self, rel_session):
        """Empty filter should work with aliases defined."""
        query = select(Item)
        filtered = apply_filter(
            query,
            Item,
            "",
            field_aliases={"cat": "category.name"},
        )
        results = rel_session.execute(filtered).scalars().all()
        assert len(results) == 4  # All items
