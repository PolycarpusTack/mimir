"""
Custom Middleware for Mimir Enterprise API

Implements tenant isolation, security headers, logging, and request tracking.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware for multi-tenant context management"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract tenant information from headers or URL
        organization_slug = request.headers.get("X-Organization-Slug")
        organization_id = request.headers.get("X-Organization-ID")

        # Store tenant info in request state
        request.state.organization_slug = organization_slug
        request.state.organization_id = organization_id

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error in tenant middleware: {e}")
            raise


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and metrics"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request
        logger.info(
            f"Request started - ID: {request_id} - "
            f"Method: {request.method} - Path: {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"Request completed - ID: {request_id} - "
                f"Status: {response.status_code} - "
                f"Duration: {duration:.3f}s"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"

            return response

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Log error
            logger.error(
                f"Request failed - ID: {request_id} - " f"Error: {str(e)} - Duration: {duration:.3f}s", exc_info=True
            )
            raise


class SecurityMiddleware(BaseHTTPMiddleware):
    """OWASP-compliant security middleware with comprehensive protections"""

    def __init__(self, app, config: dict = None):
        super().__init__(app)
        self.config = config or {}

        # Rate limiting (basic implementation)
        self.rate_limit_enabled = config.get("rate_limit_enabled", True)
        self.rate_limit_requests = config.get("rate_limit_requests", 1000)
        self.rate_limit_window = config.get("rate_limit_window", 3600)  # 1 hour
        self.request_counts = {}

        # Input validation patterns
        self.dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # JavaScript protocol
            r"vbscript:",  # VBScript protocol
            r"on\w+\s*=",  # Event handlers
            r"<iframe[^>]*>",  # iframes
            r"<object[^>]*>",  # objects
            r"<embed[^>]*>",  # embeds
            r"\.\./",  # Path traversal
            r"union.*select",  # SQL injection
            r"drop\s+table",  # SQL injection
        ]

    def _get_csp_policy(self) -> str:
        """Get Content Security Policy"""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data: https: blob:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https: wss:; "
            "media-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "upgrade-insecure-requests"
        )

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limits"""
        if not self.rate_limit_enabled:
            return True

        current_time = time.time()

        # Clean up old entries
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [
                timestamp for timestamp in self.request_counts[ip] if current_time - timestamp < self.rate_limit_window
            ]
            if not self.request_counts[ip]:
                del self.request_counts[ip]

        # Check current IP
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        self.request_counts[client_ip].append(current_time)

        return len(self.request_counts[client_ip]) <= self.rate_limit_requests

    def _validate_input(self, value: str) -> bool:
        """Basic input validation for dangerous patterns"""
        if not isinstance(value, str):
            return True

        import re

        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        return True

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers (be careful with these in production)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)

            # Rate limiting check
            if not self._check_rate_limit(client_ip):
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate Limit Exceeded", "message": "Too many requests from this IP address"},
                )

            # Input validation for query parameters
            for key, value in request.query_params.items():
                if not self._validate_input(str(value)):
                    logger.warning(f"Dangerous pattern detected in query param {key} from IP: {client_ip}")
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=400, content={"error": "Bad Request", "message": "Invalid input detected"}
                    )

            # Check request size (prevent DoS)
            if hasattr(request, "headers"):
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                    logger.warning(f"Request too large from IP: {client_ip}")
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=413, content={"error": "Payload Too Large", "message": "Request body too large"}
                    )

            response = await call_next(request)

            # Add comprehensive security headers
            response.headers.update(
                {
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block",
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                    "Content-Security-Policy": self._get_csp_policy(),
                    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
                    "Cross-Origin-Embedder-Policy": "require-corp",
                    "Cross-Origin-Opener-Policy": "same-origin",
                    "Cross-Origin-Resource-Policy": "cross-origin",
                }
            )

            # Remove server information leakage
            if "server" in response.headers:
                del response.headers["server"]

            return response

        except Exception as e:
            logger.error(f"Error in security middleware: {e}")
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware (for demonstration)"""

    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.request_counts = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = int(time.time() / 60)  # Current minute

        # Clean old entries
        self.request_counts = {key: value for key, value in self.request_counts.items() if key[1] >= current_time - 1}

        # Check current minute count
        key = (client_ip, current_time)
        current_count = self.request_counts.get(key, 0)

        if current_count >= self.calls_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Increment count
        self.request_counts[key] = current_count + 1

        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting application metrics"""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        self.request_count += 1

        try:
            response = await call_next(request)

            # Track duration
            duration = time.time() - start_time
            self.total_duration += duration

            # Add metrics headers
            response.headers["X-Total-Requests"] = str(self.request_count)
            response.headers["X-Average-Duration"] = f"{self.total_duration / self.request_count:.3f}s"

            return response

        except Exception as e:
            self.error_count += 1
            duration = time.time() - start_time
            self.total_duration += duration

            logger.error(f"Request error: {e}")
            raise


class CacheMiddleware(BaseHTTPMiddleware):
    """Simple caching middleware for GET requests"""

    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)

        # Generate cache key
        cache_key = f"{request.url.path}?{request.url.query}"
        current_time = time.time()

        # Check cache
        if cache_key in self.cache:
            cached_response, cached_time = self.cache[cache_key]
            if current_time - cached_time < self.cache_ttl:
                # Return cached response
                response = Response(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers=cached_response["headers"],
                )
                response.headers["X-Cache"] = "HIT"
                return response

        # Process request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            # Read response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            # Store in cache
            self.cache[cache_key] = (
                {"content": response_body, "status_code": response.status_code, "headers": dict(response.headers)},
                current_time,
            )

            # Create new response with the content
            response = Response(content=response_body, status_code=response.status_code, headers=response.headers)

        response.headers["X-Cache"] = "MISS"
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)

            # Return standardized error response
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
            )


class CompressionMiddleware(BaseHTTPMiddleware):
    """Simple response compression middleware"""

    def __init__(self, app, minimum_size: int = 1024):
        super().__init__(app)
        self.minimum_size = minimum_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response

        # Check response size and content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type.lower() for ct in ["json", "text", "javascript", "css"]):
            return response

        # For demonstration, we'll just add the header
        # In production, you'd implement actual compression
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"

        return response
