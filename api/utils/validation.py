"""
Validation utilities for Mimir Enterprise API

Provides comprehensive validation for API requests and data.
"""

import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationHelper:
    """Helper for common validation tasks"""

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return bool(url_pattern.match(url))

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        return bool(email_pattern.match(email))

    @staticmethod
    def validate_organization_slug(slug: str) -> bool:
        """Validate organization slug format"""
        slug_pattern = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
        return bool(slug_pattern.match(slug)) and len(slug) >= 3


class APIValidator:
    """API-specific validation logic"""

    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}

    def validate_article_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate article creation/update data"""
        errors = []

        # Required fields for creation
        if "url" in data and not ValidationHelper.validate_url(data["url"]):
            errors.append("Invalid URL format")

        if "title" in data and (not data["title"] or len(data["title"].strip()) == 0):
            errors.append("Title cannot be empty")

        return errors


# Global validators
validation_helper = ValidationHelper()
api_validator = APIValidator()
