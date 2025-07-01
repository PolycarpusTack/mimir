
Technical Debt Fix Report
========================

Files Fixed: 37
Total Issues Found: 88

Issues by File:
--------------

/mnt/c/Projects/Mimir/api/__init__.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/app.py:
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Reorganized imports
  - Added TODO docstring for custom_openapi
  - WARNING - Found hardcoded port number

/mnt/c/Projects/Mimir/api/dependencies.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/middleware.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/models.py:
  - Reorganized imports
  - WARNING - Found hardcoded max value

/mnt/c/Projects/Mimir/api/models/dashboard.py:
  - Added return type hint to
  - Reorganized imports
  - WARNING - Found hardcoded max value

/mnt/c/Projects/Mimir/api/routers/__init__.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/analytics.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/api_keys.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/articles.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/auth.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/dashboards.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/keywords.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/organizations.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/saved_search_analytics.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/saved_searches.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/search.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/sources.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/users.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/routers/webhooks.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/utils/__init__.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/utils/bulk_operations.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/utils/filtering.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/utils/pagination.py:
  - Reorganized imports
  - WARNING - Found hardcoded max value

/mnt/c/Projects/Mimir/api/utils/sorting.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/api/utils/validation.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/auth/__init__.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/auth/api_key_manager.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/auth/auth_manager.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/auth/decorators.py:
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Reorganized imports
  - Added TODO docstring for decorator
  - Added TODO docstring for wrapper
  - Added TODO docstring for decorator
  - Added TODO docstring for wrapper
  - Added TODO docstring for wrapper
  - Added TODO docstring for decorator
  - Added TODO docstring for wrapper
  - Added TODO docstring for decorator
  - Added TODO docstring for wrapper
  - Added TODO docstring for decorator
  - Added TODO docstring for wrapper
  - Added TODO docstring for decorator
  - Added TODO docstring for wrapper
  - Added TODO docstring for decorator

/mnt/c/Projects/Mimir/auth/jwt_handler.py:
  - Reorganized imports
  - Added TODO docstring for from_dict
  - Added TODO docstring for to_dict

/mnt/c/Projects/Mimir/auth/rbac.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/auth/tenant_context.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/services/alert_scheduler.py:
  - Added return type hint to
  - Added return type hint to
  - Added return type hint to
  - Reorganized imports

/mnt/c/Projects/Mimir/services/dashboard_manager.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports

/mnt/c/Projects/Mimir/services/saved_search_analytics.py:
  - Reorganized imports

/mnt/c/Projects/Mimir/services/saved_search_manager.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports

/mnt/c/Projects/Mimir/services/widget_data_service.py:
  - WARNING - Found f-string in SQL query (potential SQL injection)
  - Reorganized imports
  - WARNING - Synchronous DB calls in async function
