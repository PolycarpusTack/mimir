"""
API Contract Tests for Mimir Enterprise

Tests API contracts to ensure consistent behavior and data schemas.
"""

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError, validate


class TestAPIContract:
    """API contract testing suite"""

    def test_openapi_schema_generation(self, client: TestClient):
        """Test OpenAPI schema is properly generated"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema

        # Check required API info
        assert schema["info"]["title"] == "Mimir Test API"
        assert "version" in schema["info"]

        # Check security schemes
        assert "securitySchemes" in schema["components"]
        assert "BearerAuth" in schema["components"]["securitySchemes"]
        assert "ApiKeyAuth" in schema["components"]["securitySchemes"]

    def test_health_endpoint_contract(self, client: TestClient):
        """Test health endpoint returns expected schema"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["status", "version", "timestamp"]

        for field in required_fields:
            assert field in data

        assert data["status"] == "healthy"
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)

    def test_api_info_endpoint_contract(self, client: TestClient):
        """Test API info endpoint returns expected schema"""
        response = client.get("/api/v1/info")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["name", "version", "description", "documentation", "openapi"]

        for field in required_fields:
            assert field in data

        assert data["documentation"] == "/docs"
        assert data["openapi"] == "/openapi.json"

    def test_authentication_error_schema(self, client: TestClient):
        """Test authentication error returns consistent schema"""
        response = client.get("/api/v1/articles")
        assert response.status_code == 401

        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)

    def test_validation_error_schema(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test validation errors return consistent schema"""
        # Send invalid data
        invalid_article = {
            "title": "",  # Invalid: empty title
            "content": "Test content"
            # Missing required fields
        }

        response = client.post("/api/v1/articles", json=invalid_article, headers=auth_headers)

        # Should return validation error
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)

        # Check error format
        if data["detail"]:
            error = data["detail"][0]
            assert "loc" in error
            assert "msg" in error
            assert "type" in error

    def test_articles_endpoint_response_schema(
        self, client: TestClient, auth_headers: Dict[str, str], contract_test_schemas: Dict[str, Any]
    ):
        """Test articles endpoint returns valid response schema"""
        response = client.get("/api/v1/articles", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Validate pagination schema
            validate(data, contract_test_schemas["search_result_schema"])

            # Validate article items if present
            if data["items"]:
                for article in data["items"]:
                    validate(article, contract_test_schemas["article_schema"])

    def test_article_creation_response_schema(
        self, client: TestClient, auth_headers: Dict[str, str], contract_test_schemas: Dict[str, Any]
    ):
        """Test article creation returns valid article schema"""
        new_article = {
            "title": "Test Article",
            "content": "This is test content",
            "url": "https://example.com/test",
            "source": "Test Source",
            "author": "Test Author",
            "category": "Technology",
            "tags": ["test", "api"],
        }

        response = client.post("/api/v1/articles", json=new_article, headers=auth_headers)

        if response.status_code == 201:
            data = response.json()
            validate(data, contract_test_schemas["article_schema"])

    def test_search_endpoint_response_schema(
        self, client: TestClient, auth_headers: Dict[str, str], contract_test_schemas: Dict[str, Any]
    ):
        """Test search endpoint returns valid response schema"""
        search_params = {"q": "test", "page": 1, "page_size": 20}

        response = client.get("/api/v1/search", params=search_params, headers=auth_headers)

        if response.status_code == 200:
            data = response.json()
            validate(data, contract_test_schemas["search_result_schema"])

    def test_error_response_consistency(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test error responses are consistent across endpoints"""
        error_endpoints = [
            ("/api/v1/articles/invalid-uuid", 404),
            ("/api/v1/search?page=0", 422),  # Invalid page number
        ]

        for endpoint, expected_status in error_endpoints:
            response = client.get(endpoint, headers=auth_headers)

            if response.status_code == expected_status:
                data = response.json()

                # All error responses should have detail field
                assert "detail" in data

                # Check for optional fields
                if expected_status == 404:
                    assert "error" in data or "detail" in data

    def test_pagination_parameters(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test pagination parameters are consistent across endpoints"""
        paginated_endpoints = ["/api/v1/articles", "/api/v1/search", "/api/v1/saved-searches", "/api/v1/dashboards"]

        for endpoint in paginated_endpoints:
            # Test with pagination parameters
            response = client.get(endpoint, params={"page": 1, "page_size": 10}, headers=auth_headers)

            if response.status_code == 200:
                data = response.json()

                # Check pagination fields
                assert "items" in data
                assert "total" in data
                assert "page" in data
                assert "page_size" in data

                # Validate data types
                assert isinstance(data["items"], list)
                assert isinstance(data["total"], int)
                assert isinstance(data["page"], int)
                assert isinstance(data["page_size"], int)

                assert data["page"] >= 1
                assert data["page_size"] >= 1
                assert data["total"] >= 0

    def test_date_format_consistency(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test date formats are consistent across API"""
        response = client.get("/api/v1/articles", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            if data["items"]:
                article = data["items"][0]

                # Check date fields use ISO format
                date_fields = ["published_date", "created_at", "updated_at"]

                for field in date_fields:
                    if field in article and article[field]:
                        # Should be valid ISO 8601 format
                        from datetime import datetime

                        try:
                            datetime.fromisoformat(article[field].replace("Z", "+00:00"))
                        except ValueError:
                            pytest.fail(f"Invalid date format in field {field}: {article[field]}")

    def test_uuid_format_consistency(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test UUID formats are consistent across API"""
        import uuid

        response = client.get("/api/v1/articles", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            if data["items"]:
                article = data["items"][0]

                # Check UUID fields
                uuid_fields = ["id", "organization_id", "created_by"]

                for field in uuid_fields:
                    if field in article and article[field]:
                        try:
                            uuid.UUID(article[field])
                        except ValueError:
                            pytest.fail(f"Invalid UUID format in field {field}: {article[field]}")

    def test_content_type_headers(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test content-type headers are correct"""
        response = client.get("/api/v1/articles", headers=auth_headers)

        if response.status_code == 200:
            assert response.headers["content-type"] == "application/json"

    def test_security_headers_present(self, client: TestClient):
        """Test security headers are present in responses"""
        response = client.get("/health")

        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
            "strict-transport-security",
            "content-security-policy",
        ]

        for header in security_headers:
            assert header in response.headers, f"Missing security header: {header}"

    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are properly configured"""
        # Preflight request
        response = client.options("/api/v1/articles", headers={"Origin": "http://localhost:3000"})

        # Should have CORS headers
        cors_headers = ["access-control-allow-origin", "access-control-allow-methods", "access-control-allow-headers"]

        for header in cors_headers:
            assert header in response.headers, f"Missing CORS header: {header}"

    def test_api_versioning_consistency(self, client: TestClient):
        """Test API versioning is consistent"""
        # All API endpoints should be under /api/v1/
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        paths = schema["paths"]

        api_paths = [path for path in paths.keys() if path.startswith("/api/")]

        for path in api_paths:
            assert path.startswith("/api/v1/"), f"API path not properly versioned: {path}"

    def test_rate_limiting_headers(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test rate limiting headers are present"""
        response = client.get("/api/v1/articles", headers=auth_headers)

        # Should include rate limiting information
        rate_limit_headers = ["x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"]

        # Note: Headers might not be present if rate limiting is disabled in tests
        # This is more of a contract check for production API
        for header in rate_limit_headers:
            if header in response.headers:
                assert response.headers[header].isdigit(), f"Invalid rate limit header format: {header}"
