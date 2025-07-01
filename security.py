"""
Security Module for Mimir Async Scraper
EPIC 4 - Critical Fix: Input validation, rate limiting, authentication
"""

import asyncio
import hashlib
import hmac
import logging
import re
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    enable_input_validation: bool = True
    require_api_key: bool = False
    api_key: Optional[str] = None
    allowed_origins: List[str] = None
    enable_cors: bool = True
    max_url_length: int = 2048
    max_input_length: int = 10000
    blocked_domains: List[str] = None

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["localhost", "127.0.0.1"]
        if self.blocked_domains is None:
            self.blocked_domains = []


class InputValidator:
    """Comprehensive input validation."""

    def __init__(self, config: SecurityConfig):
        self.config = config

        # URL validation patterns
        self.url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        # Safe characters for different input types
        self.safe_chars = {
            "alphanumeric": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "text": re.compile(r"^[a-zA-Z0-9\s\.,!?;:()\'-]+$"),
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "filename": re.compile(r"^[a-zA-Z0-9._-]+$"),
        }

        # Dangerous patterns to block
        self.dangerous_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),  # XSS
            re.compile(r"javascript:", re.IGNORECASE),  # JavaScript URLs
            re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers
            re.compile(r"(union|select|insert|update|delete|drop|create|alter)\s+", re.IGNORECASE),  # SQL injection
            re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),  # Path traversal
        ]

    def validate_url(self, url: str) -> bool:
        """Validate URL format and safety."""
        if not url or len(url) > self.config.max_url_length:
            return False

        # Check URL format
        if not self.url_pattern.match(url):
            return False

        # Parse URL for further validation
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        # Check for blocked domains
        if parsed.netloc.lower() in self.config.blocked_domains:
            logger.warning(f"Blocked domain attempted: {parsed.netloc}")
            return False

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(url):
                logger.warning(f"Dangerous pattern detected in URL: {url}")
                return False

        return True

    def validate_text_input(self, text: str, input_type: str = "text") -> bool:
        """Validate text input based on type."""
        if not text:
            return True  # Empty input is okay

        if len(text) > self.config.max_input_length:
            return False

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(text):
                logger.warning(f"Dangerous pattern detected in text input")
                return False

        # Type-specific validation
        if input_type in self.safe_chars:
            return bool(self.safe_chars[input_type].match(text))

        return True

    def sanitize_input(self, text: str) -> str:
        """Sanitize input by removing dangerous content."""
        if not text:
            return text

        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            text = pattern.sub("", text)

        # Remove null bytes and control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

        # Limit length
        if len(text) > self.config.max_input_length:
            text = text[: self.config.max_input_length]

        return text.strip()


class RateLimiter:
    """Rate limiting implementation with sliding window."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.block_duration = 300  # 5 minutes
        self.blocked_until: Dict[str, float] = {}

    def _clean_old_requests(self, client_id: str):
        """Remove old requests outside the window."""
        now = time.time()
        cutoff = now - self.window_seconds
        self.requests[client_id] = [req_time for req_time in self.requests[client_id] if req_time > cutoff]

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()

        # Check if client is temporarily blocked
        if client_id in self.blocked_until:
            if now < self.blocked_until[client_id]:
                return False
            else:
                # Unblock client
                del self.blocked_until[client_id]
                self.blocked_ips.discard(client_id)

        # Clean old requests
        self._clean_old_requests(client_id)

        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            # Block client temporarily
            self.blocked_ips.add(client_id)
            self.blocked_until[client_id] = now + self.block_duration
            logger.warning(f"Rate limit exceeded for client {client_id}, blocked for {self.block_duration}s")
            return False

        # Record the request
        self.requests[client_id].append(now)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        active_clients = len([client for client, reqs in self.requests.items() if reqs])
        blocked_clients = len(self.blocked_ips)

        return {
            "active_clients": active_clients,
            "blocked_clients": blocked_clients,
            "total_clients": len(self.requests),
            "max_requests_per_window": self.max_requests,
            "window_seconds": self.window_seconds,
            "timestamp": now,
        }


class APIKeyManager:
    """API key management and validation."""

    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_usage: Dict[str, List[float]] = defaultdict(list)

    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        api_key = secrets.token_urlsafe(32)

        self.api_keys[api_key] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "permissions": permissions or ["read"],
            "active": True,
            "last_used": None,
            "usage_count": 0,
        }

        logger.info(f"Generated API key for {name}")
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        if not api_key or api_key not in self.api_keys:
            return False

        key_info = self.api_keys[api_key]
        if not key_info["active"]:
            return False

        # Update usage
        key_info["last_used"] = datetime.now().isoformat()
        key_info["usage_count"] += 1
        self.key_usage[api_key].append(time.time())

        return True

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"Revoked API key: {self.api_keys[api_key]['name']}")
            return True
        return False

    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key."""
        return self.api_keys.get(api_key)


class SecurityManager:
    """Central security manager."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validator = InputValidator(config)
        self.rate_limiter = (
            RateLimiter(max_requests=config.max_requests_per_minute, window_seconds=60)
            if config.enable_rate_limiting
            else None
        )
        self.api_key_manager = APIKeyManager() if config.require_api_key else None

        # Initialize API key if provided
        if config.api_key:
            self.api_key_manager.api_keys[config.api_key] = {
                "name": "default",
                "created_at": datetime.now().isoformat(),
                "permissions": ["read", "write", "admin"],
                "active": True,
                "last_used": None,
                "usage_count": 0,
            }

    def validate_request(self, url: str = None, api_key: str = None, client_ip: str = None, **kwargs) -> Dict[str, Any]:
        """Comprehensive request validation."""
        result = {"valid": True, "errors": [], "warnings": []}

        # URL validation
        if url and self.config.enable_input_validation:
            if not self.validator.validate_url(url):
                result["valid"] = False
                result["errors"].append("Invalid URL format or blocked domain")

        # Rate limiting
        if self.rate_limiter and client_ip:
            if not self.rate_limiter.is_allowed(client_ip):
                result["valid"] = False
                result["errors"].append("Rate limit exceeded")

        # API key validation
        if self.config.require_api_key:
            if not api_key:
                result["valid"] = False
                result["errors"].append("API key required")
            elif not self.api_key_manager.validate_api_key(api_key):
                result["valid"] = False
                result["errors"].append("Invalid API key")

        # Additional input validation
        for key, value in kwargs.items():
            if isinstance(value, str) and self.config.enable_input_validation:
                if not self.validator.validate_text_input(value):
                    result["warnings"].append(f"Potentially unsafe input in {key}")

        return result

    def sanitize_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all string inputs in a dictionary."""
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.validator.sanitize_input(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_inputs(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.validator.sanitize_input(item) if isinstance(item, str) else item for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        status = {
            "config": {
                "rate_limiting_enabled": self.config.enable_rate_limiting,
                "input_validation_enabled": self.config.enable_input_validation,
                "api_key_required": self.config.require_api_key,
                "cors_enabled": self.config.enable_cors,
            },
            "timestamp": datetime.now().isoformat(),
        }

        if self.rate_limiter:
            status["rate_limiter"] = self.rate_limiter.get_stats()

        if self.api_key_manager:
            active_keys = sum(1 for key_info in self.api_key_manager.api_keys.values() if key_info["active"])
            status["api_keys"] = {"total_keys": len(self.api_key_manager.api_keys), "active_keys": active_keys}

        return status


# Security middleware for web frameworks
class SecurityMiddleware:
    """Security middleware that can be integrated with web frameworks."""

    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request through security checks."""
        # Extract relevant data from request
        url = request_data.get("url")
        api_key = request_data.get("headers", {}).get("X-API-Key")
        client_ip = request_data.get("client_ip", "unknown")

        # Validate request
        validation_result = self.security_manager.validate_request(
            url=url, api_key=api_key, client_ip=client_ip, **request_data.get("data", {})
        )

        # Sanitize inputs if validation passed with warnings
        if validation_result["valid"] and "data" in request_data:
            request_data["data"] = self.security_manager.sanitize_inputs(request_data["data"])

        return {"request_data": request_data, "validation_result": validation_result}


# Factory function for easy setup
def create_security_manager(config_dict: Dict[str, Any]) -> SecurityManager:
    """Create security manager from configuration dictionary."""
    security_config = SecurityConfig(**config_dict.get("security", {}))
    return SecurityManager(security_config)


# Global security manager instance (to be initialized)
_security_manager: Optional[SecurityManager] = None


def initialize_security(config: Dict[str, Any]):
    """Initialize global security manager."""
    global _security_manager
    _security_manager = create_security_manager(config)


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    if _security_manager is None:
        raise RuntimeError("Security manager not initialized. Call initialize_security() first.")
    return _security_manager
