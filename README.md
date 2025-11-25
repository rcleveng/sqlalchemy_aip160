# AIP-160 Filter Helper for SQLAlchemy

# Usage
```python
  from sqlalchemy import select
  from utils.aip160_filter import apply_filter
```
 ## Basic filtering
 ```
  query = select(MyModel)
  filtered = apply_filter(query, MyModel, 'status = "active"')
```
 ## Multiple conditions
```
  filtered = apply_filter(query, MyModel, 'status = "active" AND priority > 3')
```
## OR conditions  
```
  filtered = apply_filter(query, MyModel, 'status = "active" OR status = "pending"')
```
## NOT operator
```
  filtered = apply_filter(query, MyModel, 'NOT status = "inactive"')
```
## Wildcard pattern matching
```
  filtered = apply_filter(query, MyModel, 'name = "Widget*"')
```
## Presence check (field is not null)
```
  filtered = apply_filter(query, MyModel, 'category:*')
```
## Complex nested expressions
```
  filtered = apply_filter(
      query, MyModel,
      '(status = "active" OR status = "pending") AND priority >= 3'
  )
```
## Restrict filterable fields for security
```
  filtered = apply_filter(
      query, MyModel, 'status = "active"',
      allowed_fields={"status", "priority"}  # Only these fields can be filtered
  )
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