"""
FastAPI application for Mimir Enterprise

RESTful API with OpenAPI documentation, authentication, and multi-tenant support.
"""

from .app import app, create_app
from .dependencies import get_auth_manager, get_current_user, get_db

__all__ = ["create_app", "app", "get_auth_manager", "get_current_user", "get_db"]
