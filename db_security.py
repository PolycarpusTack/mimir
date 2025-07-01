#!/usr/bin/env python3
"""
Database Security Utilities for Mimir News Scraper

Provides secure database query utilities to prevent SQL injection attacks
and implement security best practices.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import psycopg2
from psycopg2 import sql

logger = logging.getLogger(__name__)


class SecureQueryBuilder:
    """Secure SQL query builder that prevents injection attacks"""
    
    def __init__(self):
        """Initialize the secure query builder"""
        self._allowed_tables = {
            'articles', 'sources', 'keywords', 'keyword_alerts', 'scrape_runs',
            'users', 'organizations', 'api_keys', 'activity_log', 'subscriptions',
            'saved_searches', 'dashboards', 'notifications'
        }
        self._allowed_columns = {
            'articles': {
                'id', 'url', 'title', 'content', 'summary', 'published_date',
                'source', 'author', 'category', 'scraped_at', 'metadata', 'tags'
            },
            'sources': {
                'id', 'name', 'url', 'type', 'enabled', 'last_scraped',
                'success_count', 'error_count'
            },
            'users': {
                'id', 'email', 'name', 'role', 'organization_id', 'is_active',
                'created_at', 'last_login_at'
            }
        }
        
    def validate_identifier(self, identifier: str, identifier_type: str = "column") -> bool:
        """
        Validate SQL identifier (table/column name) to prevent injection
        
        Args:
            identifier: The identifier to validate
            identifier_type: Type of identifier ('table', 'column', 'schema')
            
        Returns:
            True if valid, False otherwise
        """
        # Check for basic SQL identifier rules
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            logger.warning(f"Invalid {identifier_type} identifier: {identifier}")
            return False
            
        # Check against whitelist if available
        if identifier_type == "table" and identifier not in self._allowed_tables:
            logger.warning(f"Table not in whitelist: {identifier}")
            return False
            
        return True
        
    def build_select_query(
        self,
        table: str,
        columns: List[str] = None,
        where_conditions: Dict[str, Any] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build secure SELECT query with parameterized values
        
        Args:
            table: Table name
            columns: List of column names to select
            where_conditions: Dictionary of WHERE conditions
            order_by: Column to order by
            limit: Query limit
            offset: Query offset
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Validate table name
        if not self.validate_identifier(table, "table"):
            raise ValueError(f"Invalid table name: {table}")
            
        # Build column list
        if columns:
            # Validate column names
            for col in columns:
                if not self.validate_identifier(col, "column"):
                    raise ValueError(f"Invalid column name: {col}")
            column_list = sql.SQL(", ").join(map(sql.Identifier, columns))
        else:
            column_list = sql.SQL("*")
            
        # Start building query
        query = sql.SQL("SELECT {columns} FROM {table}").format(
            columns=column_list,
            table=sql.Identifier(table)
        )
        
        params = []
        
        # Add WHERE conditions
        if where_conditions:
            where_parts = []
            for column, value in where_conditions.items():
                if not self.validate_identifier(column, "column"):
                    raise ValueError(f"Invalid column name in WHERE: {column}")
                    
                if isinstance(value, list):
                    # Handle IN clause
                    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in value)
                    where_parts.append(
                        sql.SQL("{column} IN ({placeholders})").format(
                            column=sql.Identifier(column),
                            placeholders=placeholders
                        )
                    )
                    params.extend(value)
                elif isinstance(value, dict) and 'operator' in value:
                    # Handle custom operators (LIKE, >, <, etc.)
                    operator = value['operator']
                    val = value['value']
                    
                    # Validate operator
                    allowed_operators = ['=', '!=', '<>', '>', '<', '>=', '<=', 'LIKE', 'ILIKE']
                    if operator not in allowed_operators:
                        raise ValueError(f"Invalid operator: {operator}")
                        
                    where_parts.append(
                        sql.SQL("{column} {operator} %s").format(
                            column=sql.Identifier(column),
                            operator=sql.SQL(operator)
                        )
                    )
                    params.append(val)
                else:
                    # Handle equality
                    where_parts.append(
                        sql.SQL("{column} = %s").format(column=sql.Identifier(column))
                    )
                    params.append(value)
                    
            if where_parts:
                query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)
                
        # Add ORDER BY
        if order_by:
            if not self.validate_identifier(order_by, "column"):
                raise ValueError(f"Invalid ORDER BY column: {order_by}")
            query += sql.SQL(" ORDER BY {column}").format(column=sql.Identifier(order_by))
            
        # Add LIMIT and OFFSET
        if limit:
            query += sql.SQL(" LIMIT %s")
            params.append(limit)
            
        if offset:
            query += sql.SQL(" OFFSET %s")
            params.append(offset)
            
        return query.as_string(psycopg2.extensions.connection("")).replace('"', ''), params
        
    def build_insert_query(
        self,
        table: str,
        data: Dict[str, Any],
        returning: List[str] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build secure INSERT query
        
        Args:
            table: Table name
            data: Dictionary of column->value pairs
            returning: List of columns to return
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Validate table name
        if not self.validate_identifier(table, "table"):
            raise ValueError(f"Invalid table name: {table}")
            
        # Validate column names
        for column in data.keys():
            if not self.validate_identifier(column, "column"):
                raise ValueError(f"Invalid column name: {column}")
                
        # Build query
        columns = list(data.keys())
        values = list(data.values())
        
        column_list = sql.SQL(", ").join(map(sql.Identifier, columns))
        placeholder_list = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
        
        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders})").format(
            table=sql.Identifier(table),
            columns=column_list,
            placeholders=placeholder_list
        )
        
        # Add RETURNING clause
        if returning:
            for column in returning:
                if not self.validate_identifier(column, "column"):
                    raise ValueError(f"Invalid RETURNING column: {column}")
            returning_list = sql.SQL(", ").join(map(sql.Identifier, returning))
            query += sql.SQL(" RETURNING {columns}").format(columns=returning_list)
            
        return query.as_string(psycopg2.extensions.connection("")).replace('"', ''), values
        
    def build_update_query(
        self,
        table: str,
        data: Dict[str, Any],
        where_conditions: Dict[str, Any],
        returning: List[str] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build secure UPDATE query
        
        Args:
            table: Table name
            data: Dictionary of column->value pairs to update
            where_conditions: Dictionary of WHERE conditions
            returning: List of columns to return
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Validate table name
        if not self.validate_identifier(table, "table"):
            raise ValueError(f"Invalid table name: {table}")
            
        # Validate column names
        for column in data.keys():
            if not self.validate_identifier(column, "column"):
                raise ValueError(f"Invalid column name: {column}")
                
        # Build SET clause
        set_parts = []
        params = []
        
        for column, value in data.items():
            set_parts.append(
                sql.SQL("{column} = %s").format(column=sql.Identifier(column))
            )
            params.append(value)
            
        query = sql.SQL("UPDATE {table} SET {set_clause}").format(
            table=sql.Identifier(table),
            set_clause=sql.SQL(", ").join(set_parts)
        )
        
        # Add WHERE conditions (required for UPDATE)
        if not where_conditions:
            raise ValueError("WHERE conditions are required for UPDATE queries")
            
        where_parts = []
        for column, value in where_conditions.items():
            if not self.validate_identifier(column, "column"):
                raise ValueError(f"Invalid column name in WHERE: {column}")
            where_parts.append(
                sql.SQL("{column} = %s").format(column=sql.Identifier(column))
            )
            params.append(value)
            
        query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)
        
        # Add RETURNING clause
        if returning:
            for column in returning:
                if not self.validate_identifier(column, "column"):
                    raise ValueError(f"Invalid RETURNING column: {column}")
            returning_list = sql.SQL(", ").join(map(sql.Identifier, returning))
            query += sql.SQL(" RETURNING {columns}").format(columns=returning_list)
            
        return query.as_string(psycopg2.extensions.connection("")).replace('"', ''), params


class SecureQueryExecutor:
    """Secure query executor with automatic injection prevention"""
    
    def __init__(self, connection_factory):
        """
        Initialize secure query executor
        
        Args:
            connection_factory: Function that returns database connection
        """
        self.connection_factory = connection_factory
        self.query_builder = SecureQueryBuilder()
        
    def execute_select(
        self,
        table: str,
        columns: List[str] = None,
        where_conditions: Dict[str, Any] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute secure SELECT query
        
        Returns:
            List of result dictionaries
        """
        query, params = self.query_builder.build_select_query(
            table, columns, where_conditions, order_by, limit, offset
        )
        
        with self.connection_factory() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def execute_insert(
        self,
        table: str,
        data: Dict[str, Any],
        returning: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute secure INSERT query
        
        Returns:
            Dictionary of returned values if RETURNING clause used
        """
        query, params = self.query_builder.build_insert_query(table, data, returning)
        
        with self.connection_factory() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, params)
            
            if returning:
                result = cursor.fetchone()
                return dict(result) if result else None
            return None
            
    def execute_update(
        self,
        table: str,
        data: Dict[str, Any],
        where_conditions: Dict[str, Any],
        returning: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute secure UPDATE query
        
        Returns:
            Dictionary of returned values if RETURNING clause used
        """
        query, params = self.query_builder.build_update_query(
            table, data, where_conditions, returning
        )
        
        with self.connection_factory() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, params)
            
            if returning:
                result = cursor.fetchone()
                return dict(result) if result else None
            return None
            
    def execute_raw_query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Execute raw query with validation (use sparingly)
        
        Args:
            query: Pre-validated SQL query
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        # Basic validation - no dynamic SQL construction allowed
        if any(dangerous in query.upper() for dangerous in ['DROP', 'ALTER', 'CREATE', 'TRUNCATE']):
            raise ValueError("DDL operations not allowed in raw queries")
            
        # Check for potential injection patterns
        if re.search(r'["\';]', query.replace('%s', '').replace('?', '')):
            logger.warning(f"Potentially dangerous query detected: {query}")
            
        with self.connection_factory() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, params or [])
            return [dict(row) for row in cursor.fetchall()]


# Global secure query executor instance
secure_executor = None

def get_secure_executor(connection_factory):
    """Get or create secure query executor instance"""
    global secure_executor
    if secure_executor is None:
        secure_executor = SecureQueryExecutor(connection_factory)
    return secure_executor


if __name__ == "__main__":
    # Test the secure query builder
    builder = SecureQueryBuilder()
    
    # Test SELECT query
    query, params = builder.build_select_query(
        "articles",
        columns=["id", "title", "url"],
        where_conditions={
            "source": "TechCrunch",
            "published_date": {
                "operator": ">",
                "value": "2024-01-01"
            }
        },
        order_by="published_date",
        limit=10
    )
    print(f"SELECT Query: {query}")
    print(f"Parameters: {params}")
    
    # Test INSERT query
    query, params = builder.build_insert_query(
        "articles",
        {
            "title": "Test Article",
            "url": "https://example.com/test",
            "source": "Test Source"
        },
        returning=["id", "created_at"]
    )
    print(f"\\nINSERT Query: {query}")
    print(f"Parameters: {params}")