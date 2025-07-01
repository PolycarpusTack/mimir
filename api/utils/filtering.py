"""
Advanced filtering utilities for Mimir Enterprise API

Provides dynamic query building with type-safe filters and complex operations.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Supported filter operators"""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    LIKE = "like"
    ILIKE = "ilike"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    FULL_TEXT_SEARCH = "fts"
    ARRAY_CONTAINS = "array_contains"
    ARRAY_OVERLAP = "array_overlap"
    JSONB_CONTAINS = "jsonb_contains"
    JSONB_KEY_EXISTS = "jsonb_key_exists"


@dataclass
class FilterField:
    """Definition of a filterable field"""

    name: str
    db_column: str
    data_type: str  # 'string', 'integer', 'float', 'boolean', 'date', 'array', 'jsonb'
    allowed_operators: List[FilterOperator]
    validation_func: Optional[Callable] = None
    transform_func: Optional[Callable] = None


@dataclass
class SearchFilter:
    """Individual search filter"""

    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True


class FilterBuilder:
    """Dynamic filter builder for SQL queries"""

    def __init__(self):
        self.filters: List[SearchFilter] = []
        self.field_definitions: Dict[str, FilterField] = {}
        self.operator_mapping = {
            FilterOperator.EQUALS: "= %s",
            FilterOperator.NOT_EQUALS: "!= %s",
            FilterOperator.GREATER_THAN: "> %s",
            FilterOperator.GREATER_THAN_OR_EQUAL: ">= %s",
            FilterOperator.LESS_THAN: "< %s",
            FilterOperator.LESS_THAN_OR_EQUAL: "<= %s",
            FilterOperator.LIKE: "LIKE %s",
            FilterOperator.ILIKE: "ILIKE %s",
            FilterOperator.IN: "= ANY(%s)",
            FilterOperator.NOT_IN: "!= ALL(%s)",
            FilterOperator.IS_NULL: "IS NULL",
            FilterOperator.IS_NOT_NULL: "IS NOT NULL",
            FilterOperator.BETWEEN: "BETWEEN %s AND %s",
            FilterOperator.CONTAINS: "ILIKE %s",
            FilterOperator.STARTS_WITH: "ILIKE %s",
            FilterOperator.ENDS_WITH: "ILIKE %s",
            FilterOperator.REGEX: "~ %s",
            FilterOperator.FULL_TEXT_SEARCH: "@@ websearch_to_tsquery('english', %s)",
            FilterOperator.ARRAY_CONTAINS: "@> %s",
            FilterOperator.ARRAY_OVERLAP: "&& %s",
            FilterOperator.JSONB_CONTAINS: "@> %s",
            FilterOperator.JSONB_KEY_EXISTS: "? %s",
        }

    def register_field(self, field_def: FilterField) -> None:
        """Register a filterable field"""
        self.field_definitions[field_def.name] = field_def

    def register_article_fields(self) -> None:
        """Register common article fields"""
        fields = [
            FilterField(
                name="title",
                db_column="title",
                data_type="string",
                allowed_operators=[
                    FilterOperator.EQUALS,
                    FilterOperator.ILIKE,
                    FilterOperator.CONTAINS,
                    FilterOperator.STARTS_WITH,
                    FilterOperator.ENDS_WITH,
                    FilterOperator.FULL_TEXT_SEARCH,
                ],
            ),
            FilterField(
                name="content",
                db_column="content",
                data_type="string",
                allowed_operators=[FilterOperator.ILIKE, FilterOperator.CONTAINS, FilterOperator.FULL_TEXT_SEARCH],
            ),
            FilterField(
                name="source",
                db_column="source",
                data_type="string",
                allowed_operators=[
                    FilterOperator.EQUALS,
                    FilterOperator.ILIKE,
                    FilterOperator.IN,
                    FilterOperator.NOT_IN,
                ],
            ),
            FilterField(
                name="category",
                db_column="category",
                data_type="string",
                allowed_operators=[
                    FilterOperator.EQUALS,
                    FilterOperator.IN,
                    FilterOperator.IS_NULL,
                    FilterOperator.IS_NOT_NULL,
                ],
            ),
            FilterField(
                name="language",
                db_column="language",
                data_type="string",
                allowed_operators=[FilterOperator.EQUALS, FilterOperator.IN],
            ),
            FilterField(
                name="published_date",
                db_column="published_date",
                data_type="date",
                allowed_operators=[
                    FilterOperator.EQUALS,
                    FilterOperator.GREATER_THAN,
                    FilterOperator.GREATER_THAN_OR_EQUAL,
                    FilterOperator.LESS_THAN,
                    FilterOperator.LESS_THAN_OR_EQUAL,
                    FilterOperator.BETWEEN,
                ],
            ),
            FilterField(
                name="scraped_at",
                db_column="scraped_at",
                data_type="date",
                allowed_operators=[
                    FilterOperator.GREATER_THAN_OR_EQUAL,
                    FilterOperator.LESS_THAN_OR_EQUAL,
                    FilterOperator.BETWEEN,
                ],
            ),
            FilterField(
                name="status",
                db_column="status",
                data_type="string",
                allowed_operators=[FilterOperator.EQUALS, FilterOperator.IN],
            ),
            FilterField(
                name="tags",
                db_column="tags",
                data_type="array",
                allowed_operators=[FilterOperator.ARRAY_CONTAINS, FilterOperator.ARRAY_OVERLAP],
            ),
            FilterField(
                name="metadata",
                db_column="metadata",
                data_type="jsonb",
                allowed_operators=[FilterOperator.JSONB_CONTAINS, FilterOperator.JSONB_KEY_EXISTS],
            ),
        ]

        for field in fields:
            self.register_field(field)

    def add_filter(self, field: str, operator: FilterOperator, value: Any, case_sensitive: bool = True) -> None:
        """Add a filter to the builder"""
        # Validate field exists
        if field not in self.field_definitions:
            raise ValueError(f"Unknown field: {field}")

        field_def = self.field_definitions[field]

        # Validate operator is allowed for this field
        if operator not in field_def.allowed_operators:
            raise ValueError(f"Operator {operator.value} not allowed for field {field}")

        # Validate and transform value
        validated_value = self._validate_value(field_def, operator, value)

        # Add filter
        filter_obj = SearchFilter(field=field, operator=operator, value=validated_value, case_sensitive=case_sensitive)

        self.filters.append(filter_obj)

    def _validate_value(self, field_def: FilterField, operator: FilterOperator, value: Any) -> Any:
        """Validate and transform filter value"""
        # Handle null operators
        if operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return None

        # Apply custom validation if provided
        if field_def.validation_func:
            value = field_def.validation_func(value)

        # Type-specific validation
        if field_def.data_type == "integer":
            if operator == FilterOperator.IN:
                return [int(v) for v in value] if isinstance(value, list) else [int(value)]
            elif operator == FilterOperator.BETWEEN:
                return [int(value[0]), int(value[1])]
            else:
                return int(value)

        elif field_def.data_type == "float":
            if operator == FilterOperator.IN:
                return [float(v) for v in value] if isinstance(value, list) else [float(value)]
            elif operator == FilterOperator.BETWEEN:
                return [float(value[0]), float(value[1])]
            else:
                return float(value)

        elif field_def.data_type == "boolean":
            return bool(value)

        elif field_def.data_type == "array":
            if not isinstance(value, list):
                value = [value]
            return value

        elif field_def.data_type == "string":
            # Transform string values based on operator
            if operator == FilterOperator.CONTAINS:
                return f"%{value}%"
            elif operator == FilterOperator.STARTS_WITH:
                return f"{value}%"
            elif operator == FilterOperator.ENDS_WITH:
                return f"%{value}"
            elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                return value if isinstance(value, list) else [value]

        # Apply custom transform if provided
        if field_def.transform_func:
            value = field_def.transform_func(value)

        return value

    def build_where_clause(self, organization_id: str) -> tuple[str, List[Any]]:
        """
        Build WHERE clause and parameters from filters

        Args:
            organization_id: Organization ID for tenant isolation

        Returns:
            Tuple of (where_clause, parameters)
        """
        conditions = ["organization_id = %s"]
        params = [organization_id]

        for filter_obj in self.filters:
            field_def = self.field_definitions[filter_obj.field]
            db_column = field_def.db_column

            # Get SQL operator template
            sql_template = self.operator_mapping.get(filter_obj.operator)
            if not sql_template:
                continue

            # Handle special cases
            if filter_obj.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
                conditions.append(f"{db_column} {sql_template}")

            elif filter_obj.operator == FilterOperator.BETWEEN:
                conditions.append(f"{db_column} {sql_template}")
                params.extend(filter_obj.value)

            elif filter_obj.operator == FilterOperator.FULL_TEXT_SEARCH:
                # Use search_vector for full-text search
                conditions.append(f"search_vector {sql_template}")
                params.append(filter_obj.value)

            else:
                # Standard operators
                if not filter_obj.case_sensitive and field_def.data_type == "string":
                    db_column = f"LOWER({db_column})"
                    if isinstance(filter_obj.value, str):
                        filter_obj.value = filter_obj.value.lower()
                    elif isinstance(filter_obj.value, list):
                        filter_obj.value = [v.lower() if isinstance(v, str) else v for v in filter_obj.value]

                conditions.append(f"{db_column} {sql_template}")
                params.append(filter_obj.value)

        where_clause = " AND ".join(conditions)
        return where_clause, params

    def parse_query_filters(self, query_params: Dict[str, Any]) -> None:
        """
        Parse filters from query parameters

        Expected format: field__operator=value
        Example: title__ilike=news, published_date__gte=2024-01-01
        """
        for param_name, param_value in query_params.items():
            if "__" not in param_name:
                continue

            parts = param_name.split("__")
            if len(parts) != 2:
                continue

            field_name, operator_str = parts

            # Skip if field not registered
            if field_name not in self.field_definitions:
                continue

            # Parse operator
            try:
                operator = FilterOperator(operator_str)
            except ValueError:
                logger.warning(f"Unknown operator: {operator_str}")
                continue

            # Parse value (handle multiple values for IN operations)
            if isinstance(param_value, str) and "," in param_value:
                value = [v.strip() for v in param_value.split(",")]
            else:
                value = param_value

            try:
                self.add_filter(field_name, operator, value)
            except ValueError as e:
                logger.warning(f"Invalid filter {param_name}={param_value}: {e}")

    def clear_filters(self) -> None:
        """Clear all filters"""
        self.filters.clear()

    def get_applied_filters(self) -> Dict[str, Any]:
        """Get summary of applied filters"""
        return {
            "count": len(self.filters),
            "filters": [
                {"field": f.field, "operator": f.operator.value, "value": f.value, "case_sensitive": f.case_sensitive}
                for f in self.filters
            ],
        }


class AdvancedFilterBuilder(FilterBuilder):
    """Extended filter builder with advanced features"""

    def __init__(self):
        super().__init__()
        self.joins: List[str] = []
        self.subqueries: Dict[str, str] = {}

    def add_join(self, join_clause: str) -> None:
        """Add a JOIN clause for complex filtering"""
        if join_clause not in self.joins:
            self.joins.append(join_clause)

    def add_subquery_filter(self, name: str, subquery: str) -> None:
        """Add a subquery-based filter"""
        self.subqueries[name] = subquery

    def build_full_query(self, base_query: str, organization_id: str) -> tuple[str, List[Any]]:
        """
        Build complete query with joins and subqueries

        Args:
            base_query: Base SELECT query
            organization_id: Organization ID for tenant isolation

        Returns:
            Tuple of (complete_query, parameters)
        """
        # Add joins
        query = base_query
        for join in self.joins:
            query += f" {join}"

        # Build WHERE clause
        where_clause, params = self.build_where_clause(organization_id)

        # Add subquery conditions
        for subquery_name, subquery in self.subqueries.items():
            where_clause += f" AND EXISTS ({subquery})"

        # Add WHERE clause to query
        if "WHERE" in query.upper():
            query += f" AND {where_clause[len('organization_id = %s AND '):]}"
        else:
            query += f" WHERE {where_clause}"

        return query, params


# Global filter builder instance
filter_builder = FilterBuilder()
