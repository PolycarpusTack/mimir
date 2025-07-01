"""Dependencies for FastAPI routes."""

from typing import AsyncGenerator, Dict, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session, DatabaseManager
from ..core.services import ArticleService, SourceService, ScrapingService
from ..config import get_settings


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async with get_async_session() as session:
        yield session


async def get_article_service(
    session: AsyncSession = Depends(get_db_session)
) -> ArticleService:
    """Get article service dependency."""
    return ArticleService(session)


async def get_source_service(
    session: AsyncSession = Depends(get_db_session)
) -> SourceService:
    """Get source service dependency."""
    return SourceService(session)


async def get_scraping_service(
    session: AsyncSession = Depends(get_db_session)
) -> ScrapingService:
    """Get scraping service dependency."""
    return ScrapingService(session)


@lru_cache()
def get_settings_dependency():
    """Get settings dependency (cached)."""
    return get_settings()


# Mock authentication for MVP - replace with proper auth later
async def get_current_user() -> Dict[str, Any]:
    """
    Mock current user for MVP.
    
    In production, this would validate JWT tokens or API keys
    and return actual user information.
    """
    return {
        "user_id": "mock-user-id",
        "username": "admin",
        "email": "admin@mimir.local",
        "is_active": True,
        "permissions": ["read:articles", "write:articles", "read:sources", "write:sources"]
    }


async def require_permission(permission: str):
    """Require specific permission for endpoint access."""
    def _check_permission(user: Dict[str, Any] = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return user
    return _check_permission


# Convenience permission dependencies
RequireReadArticles = Depends(require_permission("read:articles"))
RequireWriteArticles = Depends(require_permission("write:articles"))
RequireReadSources = Depends(require_permission("read:sources"))
RequireWriteSources = Depends(require_permission("write:sources"))


class PaginationParams:
    """Pagination parameters for list endpoints."""
    
    def __init__(
        self,
        page: int = 1,
        limit: int = 50,
        max_limit: int = 100
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page must be >= 1"
            )
        
        if limit < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be >= 1"
            )
        
        if limit > max_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Limit must be <= {max_limit}"
            )
        
        self.page = page
        self.limit = limit
        self.offset = (page - 1) * limit


def get_pagination_params(
    page: int = 1,
    limit: int = 50
) -> PaginationParams:
    """Get pagination parameters dependency."""
    return PaginationParams(page=page, limit=limit)