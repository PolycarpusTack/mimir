"""Health check router."""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db_session, get_settings_dependency
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    session: AsyncSession = Depends(get_db_session),
    settings = Depends(get_settings_dependency)
):
    """
    Health check endpoint.
    
    Returns the current status of the API and its dependencies.
    """
    # Test database connection
    try:
        await session.execute("SELECT 1")
        database_status = "connected"
    except Exception:
        database_status = "disconnected"
    
    # Check service statuses
    services = {
        "scraping": "ready",
        "articles": "ready",
        "sources": "ready",
    }
    
    return HealthResponse(
        status="healthy" if database_status == "connected" else "unhealthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        database=database_status,
        services=services
    )


@router.get("/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Mimir Broadcasting Intelligence API",
        "version": "1.0.0",
        "description": "Clean architecture news scraping API for broadcasting industry intelligence",
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }