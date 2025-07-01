"""
API Routers for Mimir Enterprise

Contains all API route definitions organized by resource type.
"""

from .advanced_nlu import router as advanced_nlu_router
from .analytics import router as analytics_router
from .api_keys import router as api_keys_router
from .articles import router as articles_router
from .auth import router as auth_router
from .clustering import router as clustering_router
from .content_generation import router as content_generation_router
from .dashboards import router as dashboards_router
from .keywords import router as keywords_router
from .notifications import router as notifications_router
from .organizations import router as organizations_router
from .recommendations import router as recommendations_router
from .saved_search_analytics import router as saved_search_analytics_router
from .saved_searches import router as saved_searches_router
from .search import router as search_router
from .sources import router as sources_router
from .subscriptions import router as subscriptions_router
from .summarization import router as summarization_router
from .translation import router as translation_router
from .usage import router as usage_router
from .users import router as users_router
from .webhooks import router as webhooks_router

__all__ = [
    "advanced_nlu_router",
    "auth_router",
    "articles_router",
    "sources_router",
    "keywords_router",
    "search_router",
    "analytics_router",
    "users_router",
    "organizations_router",
    "api_keys_router",
    "webhooks_router",
    "saved_searches_router",
    "dashboards_router",
    "saved_search_analytics_router",
    "notifications_router",
    "subscriptions_router",
    "summarization_router",
    "translation_router",
    "content_generation_router",
    "clustering_router",
    "recommendations_router",
    "usage_router",
]
