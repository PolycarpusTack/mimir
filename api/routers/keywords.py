"""
Keywords Router for Mimir Enterprise API

Handles keyword management and monitoring for news alerts.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...auth.auth_manager import AuthManager
from ...db_manager_postgres import PostgreSQLManager
from ..dependencies import (RequireKeywordsWrite, get_auth_manager,
                            get_current_user, get_db, get_pagination_params)
from ..models import (APIRouter, BaseResponse, Depends, HTTPException,
                      KeywordCreate, KeywordResponse, KeywordUpdate,
                      PaginatedResponse, Query, fastapi, from, import, status)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get('/', response_model=PaginatedResponse)
async def list_keywords(
    pagination: Dict[str, int] = Depends(get_pagination_params),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: PostgreSQLManager = Depends(get_db)
):
    """List keywords with filtering and pagination"""
    try:
        organization_id = current_user['organization_id']
        
        where_conditions = ["organization_id = %s"]
        params = [organization_id]
        
        if enabled is not None:
            where_conditions.append("enabled = %s")
            params.append(enabled)
        
        where_clause = " AND ".join(where_conditions)
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM keywords WHERE {where_clause}"
        total_result = db.fetch_one(count_query, params)
        total_items = total_result['count'] if total_result else 0
        
        # Get keywords
        query = f"""
        SELECT id, keyword, is_regex, case_sensitive, enabled, alert_email,
               priority, metadata, organization_id, created_at
        FROM keywords 
        WHERE {where_clause}
        ORDER BY priority DESC, keyword
        LIMIT %s OFFSET %s
        """
        
        params.extend([pagination['limit'], pagination['offset']])
        keywords = db.fetch_all(query, params)
        
        keyword_responses = [KeywordResponse(**keyword) for keyword in keywords]
        
        from ..models import PaginationMeta
        total_pages = (total_items + pagination['page_size'] - 1) // pagination['page_size']
        meta = PaginationMeta(
            page=pagination['page'],
            page_size=pagination['page_size'],
            total_items=total_items,
            total_pages=total_pages,
            has_next=pagination['page'] < total_pages,
            has_previous=pagination['page'] > 1
        )
        
        return PaginatedResponse(data=keyword_responses, meta=meta)
        
    except Exception as e:
        logger.error(f"Error listing keywords: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve keywords")

@router.post('/', response_model=KeywordResponse, status_code=status.HTTP_201_CREATED)
async def create_keyword(
    keyword: KeywordCreate,
    current_user: Dict[str, Any] = RequireKeywordsWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Create a new keyword for monitoring"""
    try:
        organization_id = current_user['organization_id']
        
        # Check organization keyword limits
        if not auth_manager.check_organization_limits(organization_id, 'keywords'):
            raise HTTPException(status_code=403, detail="Organization keyword limit exceeded")
        
        # Insert new keyword
        query = """
        INSERT INTO keywords (
            organization_id, keyword, is_regex, case_sensitive, enabled,
            alert_email, priority, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, keyword, is_regex, case_sensitive, enabled, alert_email,
                  priority, metadata, organization_id, created_at
        """
        
        params = (
            organization_id, keyword.keyword, keyword.is_regex,
            keyword.case_sensitive, keyword.enabled, keyword.alert_email,
            keyword.priority, keyword.metadata
        )
        
        new_keyword = db.fetch_one(query, params)
        return KeywordResponse(**new_keyword)
        
    except Exception as e:
        logger.error(f"Error creating keyword: {e}")
        raise HTTPException(status_code=500, detail="Failed to create keyword")

# Additional keyword endpoints would go here...