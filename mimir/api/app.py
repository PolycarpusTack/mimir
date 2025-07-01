"""Modern FastAPI application for Mimir news scraper."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from ..config import get_settings
from ..core.database import DatabaseManager
from .routers import articles, sources, scraping, health
from .schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    settings = get_settings()
    logger.info("Starting Mimir API...")
    
    # Initialize database
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    app.state.db_manager = db_manager
    app.state.settings = settings
    
    yield
    
    # Shutdown
    logger.info("Shutting down Mimir API...")
    if hasattr(app.state, 'db_manager'):
        await app.state.db_manager.close()


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    # Default configuration
    app_config = {
        "title": "Mimir Broadcasting Intelligence API",
        "description": "Clean architecture news scraping API for broadcasting industry intelligence",
        "version": "1.0.0",
        "debug": settings.debug,
    }
    
    if config:
        app_config.update(config)
    
    # Create FastAPI application
    app = FastAPI(
        title=app_config["title"],
        description=app_config["description"],
        version=app_config["version"],
        docs_url="/docs" if app_config["debug"] else None,
        redoc_url="/redoc" if app_config["debug"] else None,
        openapi_url="/openapi.json" if app_config["debug"] else None,
        lifespan=lifespan,
    )
    
    # Store configuration
    app.state.config = app_config
    
    # Setup middleware
    setup_middleware(app, settings)
    
    # Setup routes
    setup_routes(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Setup OpenAPI schema
    if app_config["debug"]:
        setup_openapi(app)
    
    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """Setup application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_routes(app: FastAPI) -> None:
    """Setup API routes."""
    
    # Include routers
    app.include_router(
        health.router,
        prefix="/api/v1",
        tags=["Health"]
    )
    
    app.include_router(
        articles.router,
        prefix="/api/v1/articles",
        tags=["Articles"]
    )
    
    app.include_router(
        sources.router,
        prefix="/api/v1/sources",
        tags=["Sources"]
    )
    
    app.include_router(
        scraping.router,
        prefix="/api/v1/scraping",
        tags=["Scraping"]
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": "The requested resource was not found",
                "path": str(request.url.path),
            },
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An internal server error occurred"
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "message": str(exc)
            }
        )


def setup_openapi(app: FastAPI) -> None:
    """Setup custom OpenAPI schema."""
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.state.config["title"],
            version=app.state.config["version"],
            description=app.state.config["description"],
            routes=app.routes,
        )
        
        # Add custom info
        openapi_schema["info"]["contact"] = {
            "name": "Mimir Support",
            "email": "support@mimir.local"
        }
        
        openapi_schema["info"]["license"] = {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi


# Create default application instance
app = create_app()