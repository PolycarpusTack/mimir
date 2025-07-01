"""
FastAPI Application Factory for Mimir Enterprise

Creates and configures the FastAPI application with all routes, middleware,
and enterprise features including authentication, rate limiting, and monitoring.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..auth.auth_manager import AuthManager
from ..db_manager_postgres import PostgreSQLManager
from .middleware import LoggingMiddleware, SecurityMiddleware, TenantMiddleware
from .routers import (
    advanced_nlu_router,
    analytics_router,
    api_keys_router,
    articles_router,
    auth_router,
    clustering_router,
    content_generation_router,
    dashboards_router,
    keywords_router,
    notifications_router,
    organizations_router,
    recommendations_router,
    saved_search_analytics_router,
    saved_searches_router,
    search_router,
    sources_router,
    subscriptions_router,
    summarization_router,
    translation_router,
    usage_router,
    users_router,
    webhooks_router,
)

logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Mimir Enterprise API...")

    # Initialize database connections
    db_manager = app.state.db_manager
    try:
        # Test database connection
        db_manager.test_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

    # Initialize auth manager
    auth_manager = AuthManager(db_manager)
    app.state.auth_manager = auth_manager
    logger.info("Auth manager initialized")

    yield

    # Shutdown
    logger.info("Shutting down Mimir Enterprise API...")

    # Cleanup database connections
    if hasattr(db_manager, "close_all_connections"):
        db_manager.close_all_connections()


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Create and configure FastAPI application

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured FastAPI application
    """
    # Default configuration
    default_config = {
        "title": "Mimir Enterprise API",
        "description": "AI-powered news intelligence platform with multi-tenant architecture",
        "version": "1.0.0",
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "allowed_hosts": ["*"],  # Configure properly in production
        "cors_origins": ["*"],  # Configure properly in production
        "rate_limit": "1000/hour",
    }

    if config:
        default_config.update(config)

    # Create FastAPI app
    app = FastAPI(
        title=default_config["title"],
        description=default_config["description"],
        version=default_config["version"],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store configuration
    app.state.config = default_config

    # Initialize database manager
    app.state.db_manager = PostgreSQLManager()

    # Add middleware
    setup_middleware(app, default_config)

    # Setup rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Include routers
    setup_routes(app)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Setup OpenAPI schema
    setup_openapi(app)

    return app


def setup_middleware(app: FastAPI, config: Dict[str, Any]) -> None:
    """Setup application middleware"""

    # Security middleware (should be first)
    app.add_middleware(SecurityMiddleware)

    # Trusted host middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config["allowed_hosts"])

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(TenantMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Add usage tracking middleware if configured
    if config.get("enable_usage_tracking", True):
        from ..services.usage_metering import UsageMeteringService
        from .middleware.usage_tracking import StorageTrackingMiddleware, UsageTrackingMiddleware

        # Initialize usage service
        usage_service = UsageMeteringService(app.state.db_manager)
        app.state.usage_service = usage_service

        # Add middlewares
        app.add_middleware(UsageTrackingMiddleware, usage_service=usage_service)
        app.add_middleware(StorageTrackingMiddleware, db_manager=app.state.db_manager)


def setup_routes(app: FastAPI) -> None:
    """Setup API routes"""

    # Include all routers with their prefixes
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(articles_router, prefix="/api/v1/articles", tags=["Articles"])
    app.include_router(sources_router, prefix="/api/v1/sources", tags=["Sources"])
    app.include_router(keywords_router, prefix="/api/v1/keywords", tags=["Keywords"])
    app.include_router(search_router, prefix="/api/v1/search", tags=["Search"])
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(organizations_router, prefix="/api/v1/organizations", tags=["Organizations"])
    app.include_router(api_keys_router, prefix="/api/v1/api-keys", tags=["API Keys"])
    app.include_router(webhooks_router, prefix="/api/v1/webhooks", tags=["Webhooks"])
    app.include_router(saved_searches_router, prefix="/api/v1/saved-searches", tags=["Saved Searches"])
    app.include_router(dashboards_router, prefix="/api/v1/dashboards", tags=["Dashboards"])
    app.include_router(
        saved_search_analytics_router, prefix="/api/v1/analytics/saved-searches", tags=["Saved Search Analytics"]
    )
    app.include_router(notifications_router, prefix="/api/v1/notifications", tags=["Notifications"])
    app.include_router(subscriptions_router, prefix="/api/v1/subscriptions", tags=["Subscriptions"])
    app.include_router(summarization_router, prefix="/api/v1/summarization", tags=["Summarization"])
    app.include_router(translation_router, prefix="/api/v1/translation", tags=["Translation"])
    app.include_router(content_generation_router, prefix="/api/v1/content-generation", tags=["Content Generation"])
    app.include_router(advanced_nlu_router, prefix="/api/v1/advanced-nlu", tags=["Advanced NLU"])
    app.include_router(clustering_router, prefix="/api/v1/clustering", tags=["Article Clustering"])
    app.include_router(recommendations_router, prefix="/api/v1/recommendations", tags=["Recommendations"])
    app.include_router(usage_router, prefix="/api/v1/usage", tags=["Usage & Billing"])

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": app.state.config["version"],
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        }

    # API info endpoint
    @app.get("/api/v1/info", tags=["Information"])
    async def api_info():
        """API information endpoint"""
        return {
            "name": app.state.config["title"],
            "version": app.state.config["version"],
            "description": app.state.config["description"],
            "documentation": "/docs",
            "openapi": "/openapi.json",
        }


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers"""

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
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
            status_code=500, content={"error": "Internal Server Error", "message": "An internal server error occurred"}
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": "Bad Request", "message": str(exc)})


def setup_openapi(app: FastAPI) -> None:
    """Setup custom OpenAPI schema"""

    def custom_openapi():
        """TODO: Add docstring for custom_openapi"""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.state.config["title"],
            version=app.state.config["version"],
            description=app.state.config["description"],
            routes=app.routes,
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        }

        # Add global security
        openapi_schema["security"] = [{"BearerAuth": []}, {"ApiKeyAuth": []}]

        # Add custom info
        openapi_schema["info"]["contact"] = {"name": "Mimir Support", "email": "support@mimir.ai"}

        openapi_schema["info"]["license"] = {"name": "MIT", "url": "https://opensource.org/licenses/MIT"}

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi


# Create the main application instance
app = create_app()


def run_development_server() -> None:
    """Run development server"""
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


def run_production_server() -> None:
    """Run production server"""
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, workers=4, log_level="warning")


if __name__ == "__main__":
    run_development_server()
