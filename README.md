# AIP-160 Filter Helper for SQLAlchemy

## Installation

```bash
pip install sqlalchemy-aip160
```

## Usage

```python
from sqlalchemy import select
from sqlalchemy_aip160 import apply_filter, parse_filter, FilterBuilder
```

### Basic filtering

```python
query = select(MyModel)
filtered = apply_filter(query, MyModel, 'status = "active"')
```

### Multiple conditions

```python
filtered = apply_filter(query, MyModel, 'status = "active" AND priority > 3')
```

### OR conditions

```python
filtered = apply_filter(query, MyModel, 'status = "active" OR status = "pending"')
```

### NOT operator

```python
filtered = apply_filter(query, MyModel, 'NOT status = "inactive"')
```

### Wildcard pattern matching

```python
filtered = apply_filter(query, MyModel, 'name = "Widget*"')
```

### Presence check (field is not null)

```python
filtered = apply_filter(query, MyModel, 'category:*')
```

### Complex nested expressions

```python
filtered = apply_filter(
    query, MyModel,
    '(status = "active" OR status = "pending") AND priority >= 3'
)
```

### Restrict filterable fields for security

```python
filtered = apply_filter(
    query, MyModel, 'status = "active"',
    allowed_fields={"status", "priority"}  # Only these fields can be filtered
)
```

### Field aliases

```python
filtered = apply_filter(
    query, MyModel,
    'department = "Engineering"',
    field_aliases={"department": "department.name"}
)
```

## Filter Inspection and Manipulation

`parse_filter` returns a structured `FilterExpression` that can be inspected,
manipulated, serialized back to a string, or passed directly to `apply_filter`.

### Inspect fields

```python
expr = parse_filter('status = "active" AND priority > 3')
expr.get_fields()  # {'status', 'priority'}
```

### Rename fields

```python
expr = parse_filter('kind = "issue" AND source_id = "abc"')
expr.rename_field("kind", "kind_str")
str(expr)  # 'kind_str = "issue" AND source_id = "abc"'
```

### Remove clauses

```python
expr = parse_filter('status = "active" AND priority > 3')
expr.remove("priority")
str(expr)  # 'status = "active"'
```

### Extract and replace clauses

```python
expr = parse_filter('kind = "acc:issue" AND source_id = "abc"')
kind_clauses = expr.extract("kind")  # Removes and returns kind comparisons
replacement = FilterBuilder().add("kind", "=", "acc:meeting").build()
combined = expr & replacement
str(combined)  # 'source_id = "abc" AND kind = "acc:meeting"'
```

### Combine expressions

```python
user_filter = parse_filter('status = "active"')
server_filter = parse_filter('org_id = "org-123"')
combined = user_filter & server_filter  # AND
either = user_filter | server_filter    # OR
negated = ~user_filter                  # NOT
```

### Build filters programmatically

```python
f = FilterBuilder()
f.add("kind", "=", "issue").add("priority", ">", 3)
str(f)  # 'kind = "issue" AND priority > 3'

# Pass directly to apply_filter
query = apply_filter(select(MyModel), MyModel, f.build())
```

### Pass FilterExpression to apply_filter

```python
expr = parse_filter('status = "active"')
expr.rename_field("status", "state")
filtered = apply_filter(query, MyModel, expr)
```

## Supported Features

| Feature      | Example                    | Notes                                  |
|--------------|----------------------------|----------------------------------------|
| Equality     | status = "active"          | String, int, float, bool, UUID         |
| Not equals   | status != "inactive"       |                                        |
| Comparisons  | priority > 3, score <= 4.5 | <, >, <=, >=                           |
| AND          | a = 1 AND b = 2            | Explicit                               |
| Implicit AND | a = 1 b = 2                | Adjacent terms                         |
| OR           | a = 1 OR a = 2             | Higher precedence than AND per AIP-160 |
| NOT          | NOT status = "active"      | Also -status = "active"                |
| Parentheses  | (a OR b) AND c             |                                        |
| Wildcards    | name = "*.txt"             | Converted to SQL LIKE                  |
| Presence     | field:*                    | Field is not null                      |
| Has value    | field:value                | Field equals value                     |
