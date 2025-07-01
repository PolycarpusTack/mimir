"""
Sorting utilities for Mimir Enterprise API

Provides advanced sorting functionality with multiple fields and custom orders.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SortDirection(Enum):
    """Sort direction options"""

    ASC = "asc"
    DESC = "desc"


@dataclass
class SortOption:
    """Individual sort option"""

    field: str
    direction: SortDirection
    nulls_first: bool = False


class SortHelper:
    """Helper for building complex sort clauses"""

    def __init__(self):
        self.allowed_fields: Dict[str, str] = {}  # field_name -> db_column
        self.sort_options: List[SortOption] = []

    def register_sortable_fields(self, fields: Dict[str, str]) -> None:
        """Register fields that can be sorted"""
        self.allowed_fields.update(fields)

    def add_sort(
        self, field: str, direction: Union[str, SortDirection] = SortDirection.ASC, nulls_first: bool = False
    ) -> None:
        """Add a sort option"""
        if field not in self.allowed_fields:
            raise ValueError(f"Field '{field}' is not sortable")

        if isinstance(direction, str):
            direction = SortDirection(direction.lower())

        sort_option = SortOption(field=field, direction=direction, nulls_first=nulls_first)

        self.sort_options.append(sort_option)

    def build_order_clause(self) -> str:
        """Build SQL ORDER BY clause"""
        if not self.sort_options:
            return ""

        order_parts = []
        for sort_option in self.sort_options:
            db_column = self.allowed_fields[sort_option.field]
            direction = sort_option.direction.value.upper()

            clause = f"{db_column} {direction}"

            if sort_option.nulls_first:
                clause += " NULLS FIRST"
            else:
                clause += " NULLS LAST"

            order_parts.append(clause)

        return "ORDER BY " + ", ".join(order_parts)

    def clear_sorts(self) -> None:
        """Clear all sort options"""
        self.sort_options.clear()


# Global sort helper
sort_helper = SortHelper()
