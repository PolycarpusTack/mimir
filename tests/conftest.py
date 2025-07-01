"""
Pytest Configuration and Fixtures for Mimir Enterprise

Provides shared fixtures and configuration for all test types.
"""

import asyncio
import os
import tempfile
from typing import Any, AsyncGenerator, Dict
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from api.app import create_app
from auth.auth_manager import AuthManager
from db_manager_postgres import PostgreSQLManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "title": "Mimir Test API",
        "description": "Test instance",
        "version": "1.0.0-test",
        "debug": True,
        "allowed_hosts": ["*"],
        "cors_origins": ["*"],
        "rate_limit": "10000/hour",
        "enable_usage_tracking": False,
    }


@pytest.fixture
def test_db():
    """Test database fixture"""
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables
    # Note: In real implementation, you'd run migrations here

    yield engine
    engine.dispose()


@pytest.fixture
def test_app(test_config, test_db):
    """Test FastAPI application"""
    app = create_app(test_config)

    # Override database dependency for testing
    app.state.db_manager = test_db

    return app


@pytest.fixture
def client(test_app):
    """Test client fixture"""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
async def async_client(test_app):
    """Async test client fixture"""
    async with httpx.AsyncClient(app=test_app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def auth_headers():
    """Authentication headers for testing"""
    # In real implementation, create valid JWT token
    return {"Authorization": "Bearer test-token", "X-Organization-ID": str(uuid4())}


@pytest.fixture
def test_organization():
    """Test organization data"""
    return {
        "id": str(uuid4()),
        "name": "Test Organization",
        "slug": "test-org",
        "domain": "test.com",
        "plan": "professional",
        "settings": {"features": {"advanced_search": True, "ai_analysis": True, "custom_dashboards": True}},
    }


@pytest.fixture
def test_user(test_organization):
    """Test user data"""
    return {
        "id": str(uuid4()),
        "email": "test@test.com",
        "username": "testuser",
        "organization_id": test_organization["id"],
        "role": "admin",
        "is_active": True,
        "permissions": [
            "articles:read",
            "articles:create",
            "articles:update",
            "articles:delete",
            "search:advanced",
            "dashboards:create",
        ],
    }


@pytest.fixture
def test_articles():
    """Test article data"""
    return [
        {
            "id": str(uuid4()),
            "title": "Test Article 1",
            "content": "This is test content for article 1",
            "url": "https://example.com/article1",
            "source": "Test Source",
            "author": "Test Author",
            "published_date": "2024-01-01T00:00:00Z",
            "category": "Technology",
            "tags": ["test", "api", "automation"],
            "sentiment": {"score": 0.8, "label": "positive"},
        },
        {
            "id": str(uuid4()),
            "title": "Test Article 2",
            "content": "This is test content for article 2",
            "url": "https://example.com/article2",
            "source": "Test Source",
            "author": "Test Author 2",
            "published_date": "2024-01-02T00:00:00Z",
            "category": "Business",
            "tags": ["test", "business", "news"],
            "sentiment": {"score": -0.2, "label": "negative"},
        },
    ]


@pytest.fixture
def test_keywords():
    """Test keyword data"""
    return [
        {
            "id": str(uuid4()),
            "keyword": "artificial intelligence",
            "category": "technology",
            "priority": "high",
            "alert_enabled": True,
        },
        {
            "id": str(uuid4()),
            "keyword": "machine learning",
            "category": "technology",
            "priority": "medium",
            "alert_enabled": False,
        },
    ]


@pytest.fixture
def test_saved_search():
    """Test saved search data"""
    return {
        "id": str(uuid4()),
        "name": "AI Technology Search",
        "description": "Articles about AI and ML",
        "query": {
            "terms": ["artificial intelligence", "machine learning"],
            "categories": ["technology"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        },
        "alert_enabled": True,
        "alert_frequency": "daily",
    }


@pytest.fixture
def test_dashboard():
    """Test dashboard data"""
    return {
        "id": str(uuid4()),
        "name": "Test Dashboard",
        "description": "Dashboard for testing",
        "layout": {
            "widgets": [
                {
                    "id": "widget1",
                    "type": "article_count",
                    "title": "Total Articles",
                    "position": {"x": 0, "y": 0, "w": 6, "h": 4},
                    "config": {"time_range": "7d", "filters": {}},
                },
                {
                    "id": "widget2",
                    "type": "sentiment_chart",
                    "title": "Sentiment Analysis",
                    "position": {"x": 6, "y": 0, "w": 6, "h": 4},
                    "config": {"chart_type": "pie", "time_range": "30d"},
                },
            ]
        },
        "is_public": False,
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "This article discusses artificial intelligence and its applications in healthcare."
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def performance_test_data():
    """Data for performance testing"""
    return {
        "concurrent_users": [1, 5, 10, 25, 50],
        "test_duration": 60,  # seconds
        "ramp_up_time": 10,  # seconds
        "endpoints": ["/api/v1/articles", "/api/v1/search", "/api/v1/analytics/summary", "/api/v1/dashboards"],
    }


@pytest.fixture
def load_test_scenarios():
    """Load test scenarios"""
    return {
        "light_load": {"users": 10, "duration": 30, "requests_per_second": 10},
        "normal_load": {"users": 50, "duration": 60, "requests_per_second": 50},
        "heavy_load": {"users": 100, "duration": 120, "requests_per_second": 100},
        "stress_test": {"users": 500, "duration": 300, "requests_per_second": 200},
    }


@pytest.fixture
def contract_test_schemas():
    """API contract test schemas"""
    return {
        "article_schema": {
            "type": "object",
            "required": ["id", "title", "content", "url", "source"],
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "title": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
                "url": {"type": "string", "format": "uri"},
                "source": {"type": "string", "minLength": 1},
                "author": {"type": ["string", "null"]},
                "published_date": {"type": "string", "format": "date-time"},
                "category": {"type": ["string", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "sentiment": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": -1, "maximum": 1},
                        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    },
                },
            },
        },
        "search_result_schema": {
            "type": "object",
            "required": ["items", "total", "page", "page_size"],
            "properties": {
                "items": {"type": "array"},
                "total": {"type": "integer", "minimum": 0},
                "page": {"type": "integer", "minimum": 1},
                "page_size": {"type": "integer", "minimum": 1},
                "pages": {"type": "integer", "minimum": 1},
            },
        },
    }


# Setup and teardown functions
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

    yield

    # Cleanup
    os.environ.pop("TESTING", None)
    os.environ.pop("DATABASE_URL", None)


@pytest.fixture
def temp_file():
    """Temporary file fixture"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_dir():
    """Temporary directory fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Async fixtures for integration tests
@pytest.fixture
async def async_db_session():
    """Async database session for integration tests"""
    # In real implementation, create async database session
    pass


@pytest.fixture
async def integration_test_data():
    """Integration test data setup"""
    # Setup test data in database
    pass
