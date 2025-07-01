"""
Pagination utilities for Mimir Enterprise API

Provides advanced pagination functionality with cursor-based and offset-based pagination.
"""

import logging
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PageInfo:
    """Pagination information"""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None


@dataclass
class PageResponse(Generic[T]):
    """Generic paginated response"""

    data: List[T]
    page_info: PageInfo
    filters_applied: Optional[Dict[str, Any]] = None
    sort_applied: Optional[Dict[str, str]] = None


class PaginationHelper:
    """Helper class for pagination operations"""

    def __init__(self, max_page_size: int = 1000, default_page_size: int = 50):
        self.max_page_size = max_page_size
        self.default_page_size = default_page_size

    def validate_pagination_params(self, page: Optional[int] = None, page_size: Optional[int] = None) -> Dict[str, int]:
        """
        Validate and normalize pagination parameters

        Args:
            page: Page number (1-based)
            page_size: Number of items per page

        Returns:
            Dictionary with validated pagination parameters
        """
        # Validate page
        if page is None or page < 1:
            page = 1

        # Validate page_size
        if page_size is None:
            page_size = self.default_page_size
        elif page_size < 1:
            page_size = self.default_page_size
        elif page_size > self.max_page_size:
            page_size = self.max_page_size

        # Calculate offset
        offset = (page - 1) * page_size

        return {"page": page, "page_size": page_size, "offset": offset, "limit": page_size}

    def create_page_info(
        self,
        page: int,
        page_size: int,
        total_items: int,
        start_cursor: Optional[str] = None,
        end_cursor: Optional[str] = None,
    ) -> PageInfo:
        """
        Create pagination metadata

        Args:
            page: Current page number
            page_size: Items per page
            total_items: Total number of items
            start_cursor: Optional start cursor for cursor-based pagination
            end_cursor: Optional end cursor for cursor-based pagination

        Returns:
            PageInfo object with pagination metadata
        """
        total_pages = ceil(total_items / page_size) if page_size > 0 else 0

        return PageInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
            start_cursor=start_cursor,
            end_cursor=end_cursor,
        )

    def create_page_response(
        self,
        data: List[T],
        page: int,
        page_size: int,
        total_items: int,
        filters_applied: Optional[Dict[str, Any]] = None,
        sort_applied: Optional[Dict[str, str]] = None,
        start_cursor: Optional[str] = None,
        end_cursor: Optional[str] = None,
    ) -> PageResponse[T]:
        """
        Create a complete paginated response

        Args:
            data: List of items for current page
            page: Current page number
            page_size: Items per page
            total_items: Total number of items
            filters_applied: Filters that were applied
            sort_applied: Sorting that was applied
            start_cursor: Start cursor for cursor pagination
            end_cursor: End cursor for cursor pagination

        Returns:
            PageResponse object with data and metadata
        """
        page_info = self.create_page_info(page, page_size, total_items, start_cursor, end_cursor)

        return PageResponse(data=data, page_info=page_info, filters_applied=filters_applied, sort_applied=sort_applied)


class CursorPagination:
    """Cursor-based pagination for large datasets"""

    def __init__(self, cursor_field: str = "id", page_size: int = 50):
        self.cursor_field = cursor_field
        self.page_size = page_size

    def encode_cursor(self, value: Any) -> str:
        """Encode a cursor value"""
        import base64
        import json

        cursor_data = {"field": self.cursor_field, "value": str(value)}

        json_str = json.dumps(cursor_data)
        encoded = base64.b64encode(json_str.encode()).decode()
        return encoded

    def decode_cursor(self, cursor: str) -> Dict[str, Any]:
        """Decode a cursor value"""
        import base64
        import json

        try:
            decoded = base64.b64decode(cursor.encode()).decode()
            cursor_data = json.loads(decoded)
            return cursor_data
        except Exception as e:
            logger.warning(f"Failed to decode cursor: {e}")
            return {}

    def build_cursor_query(
        self,
        base_query: str,
        base_params: List[Any],
        after_cursor: Optional[str] = None,
        before_cursor: Optional[str] = None,
        order_direction: str = "ASC",
    ) -> tuple[str, List[Any]]:
        """
        Build a cursor-based query

        Args:
            base_query: Base SQL query
            base_params: Base query parameters
            after_cursor: Cursor to paginate after
            before_cursor: Cursor to paginate before
            order_direction: Sort direction (ASC or DESC)

        Returns:
            Tuple of (query, parameters)
        """
        query = base_query
        params = base_params.copy()

        # Add cursor conditions
        if after_cursor:
            cursor_data = self.decode_cursor(after_cursor)
            if cursor_data:
                operator = ">" if order_direction == "ASC" else "<"
                query += f" AND {self.cursor_field} {operator} %s"
                params.append(cursor_data["value"])

        if before_cursor:
            cursor_data = self.decode_cursor(before_cursor)
            if cursor_data:
                operator = "<" if order_direction == "ASC" else ">"
                query += f" AND {self.cursor_field} {operator} %s"
                params.append(cursor_data["value"])

        # Add ordering and limit
        query += f" ORDER BY {self.cursor_field} {order_direction}"
        query += f" LIMIT {self.page_size + 1}"  # +1 to check if there are more pages

        return query, params

    def extract_cursors(self, results: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str], bool]:
        """
        Extract start and end cursors from results

        Args:
            results: Query results

        Returns:
            Tuple of (start_cursor, end_cursor, has_next_page)
        """
        if not results:
            return None, None, False

        # Check if we have more pages (we requested page_size + 1)
        has_next_page = len(results) > self.page_size

        # Remove the extra item if present
        if has_next_page:
            results = results[: self.page_size]

        # Extract cursors
        start_cursor = self.encode_cursor(results[0][self.cursor_field]) if results else None
        end_cursor = self.encode_cursor(results[-1][self.cursor_field]) if results else None

        return start_cursor, end_cursor, has_next_page


class AdvancedPagination:
    """Advanced pagination with search-after functionality"""

    def __init__(self):
        self.default_page_size = 50
        self.max_page_size = 1000

    def paginate_with_search_after(
        self,
        query_builder,
        search_after: Optional[List[Any]] = None,
        page_size: Optional[int] = None,
        sort_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Implement search-after pagination for efficient deep pagination

        Args:
            query_builder: Query builder function
            search_after: Values from the last item of previous page
            page_size: Number of items per page
            sort_fields: Fields to sort by

        Returns:
            Pagination result with search_after values
        """
        if page_size is None:
            page_size = self.default_page_size
        elif page_size > self.max_page_size:
            page_size = self.max_page_size

        if sort_fields is None:
            sort_fields = ["id"]

        # Build query with search_after
        query, params = query_builder(search_after, page_size + 1, sort_fields)

        # Execute query (this would be done by the caller)
        # results = db.fetch_all(query, params)

        return {"query": query, "params": params, "page_size": page_size, "sort_fields": sort_fields}

    def extract_search_after_values(
        self, results: List[Dict[str, Any]], sort_fields: List[str]
    ) -> tuple[List[Dict[str, Any]], Optional[List[Any]], bool]:
        """
        Extract search_after values from results

        Args:
            results: Query results
            sort_fields: Fields used for sorting

        Returns:
            Tuple of (page_data, search_after_values, has_next_page)
        """
        if not results:
            return [], None, False

        # Check if we have more pages
        has_next_page = len(results) > self.default_page_size

        # Get actual page data
        page_data = results[: self.default_page_size]

        # Extract search_after values from last item
        search_after_values = None
        if page_data:
            last_item = page_data[-1]
            search_after_values = [last_item[field] for field in sort_fields]

        return page_data, search_after_values, has_next_page


# Global pagination helper instance
pagination_helper = PaginationHelper()
