"""
Widget Data Service for Mimir Enterprise

Fetches and prepares data for dashboard widgets based on their type and configuration.
"""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from ..api.models.dashboard import WidgetType

logger = logging.getLogger(__name__)


class WidgetDataService:
    """Service for fetching widget data"""

    def __init__(self, db_manager, cache_manager=None):
        """
        Initialize widget data service

        Args:
            db_manager: Database manager instance
            cache_manager: Optional cache manager
        """
        self.db = db_manager
        self.cache = cache_manager

        # Map widget types to data fetchers
        self.data_fetchers = {
            WidgetType.ARTICLE_COUNT: self._fetch_article_count,
            WidgetType.ARTICLE_LIST: self._fetch_article_list,
            WidgetType.SEARCH_RESULTS: self._fetch_search_results,
            WidgetType.CHART_LINE: self._fetch_chart_line_data,
            WidgetType.CHART_BAR: self._fetch_chart_bar_data,
            WidgetType.CHART_PIE: self._fetch_chart_pie_data,
            WidgetType.METRIC_CARD: self._fetch_metric_card_data,
            WidgetType.SOURCE_STATS: self._fetch_source_stats,
            WidgetType.KEYWORD_CLOUD: self._fetch_keyword_cloud,
            WidgetType.ALERT_FEED: self._fetch_alert_feed,
            WidgetType.ACTIVITY_TIMELINE: self._fetch_activity_timeline,
            WidgetType.CUSTOM_HTML: self._fetch_custom_html,
        }

    def get_widget_data(
        self,
        widget_id: str,
        widget_type: WidgetType,
        data_source: Dict[str, Any],
        organization_id: str,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get data for a widget

        Args:
            widget_id: Widget ID
            widget_type: Type of widget
            data_source: Data source configuration
            organization_id: Organization ID for filtering
            user_id: Optional user ID for personalization
            use_cache: Whether to use cached data

        Returns:
            Widget data with metadata
        """
        try:
            # Check cache first
            if use_cache and self.cache:
                cache_key = self._generate_cache_key(widget_id, data_source)
                cached_data = self._get_cached_data(widget_id, cache_key)
                if cached_data:
                    return cached_data

            # Fetch fresh data
            fetcher = self.data_fetchers.get(widget_type)
            if not fetcher:
                raise ValueError(f"Unknown widget type: {widget_type}")

            data = fetcher(data_source, organization_id, user_id)

            # Prepare response
            response = {
                "widget_id": widget_id,
                "data": data,
                "metadata": {
                    "widget_type": widget_type,
                    "organization_id": organization_id,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
                "last_updated": datetime.now(timezone.utc),
                "next_update": self._calculate_next_update(data_source),
            }

            # Cache the data
            if use_cache and self.cache:
                self._cache_data(widget_id, cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Error fetching widget data: {e}")
            return {
                "widget_id": widget_id,
                "data": None,
                "error": str(e),
                "metadata": {"widget_type": widget_type, "error_at": datetime.now(timezone.utc).isoformat()},
            }

    def _fetch_article_count(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Union[int, Dict[str, Any]]:
        """Fetch article count data"""
        try:
            time_range = data_source.get("time_range", "24h")
            filters = data_source.get("filters", {})

            # Parse time range
            start_date = self._parse_time_range(time_range)

            # Build query
            query = """
            SELECT COUNT(*) as count
            FROM articles
            WHERE organization_id = %s
              AND scraped_at >= %s
            """
            params = [organization_id, start_date]

            # Apply filters
            if filters.get("source"):
                query += " AND source = %s"
                params.append(filters["source"])

            if filters.get("category"):
                query += " AND category = %s"
                params.append(filters["category"])

            result = self.db.fetch_one(query, params)
            return result["count"] if result else 0

        except Exception as e:
            logger.error(f"Error fetching article count: {e}")
            return 0

    def _fetch_article_list(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch article list data"""
        try:
            query_params = data_source.get("query", {})
            limit = min(data_source.get("limit", 20), 100)

            # Build query
            query = """
            SELECT 
                id, title, content, url, source, category,
                published_date, scraped_at, author
            FROM articles
            WHERE organization_id = %s
            """
            params = [organization_id]

            # Apply search query
            if query_params.get("q"):
                query += " AND search_vector @@ plainto_tsquery('english', %s)"
                params.append(query_params["q"])

            # Apply filters
            if query_params.get("source"):
                query += " AND source = %s"
                params.append(query_params["source"])

            if query_params.get("date_from"):
                query += " AND published_date >= %s"
                params.append(query_params["date_from"])

            # Add ordering
            order_by = query_params.get("sort_by", "published_date")
            order_dir = query_params.get("sort_order", "desc")
            query += f" ORDER BY {order_by} {order_dir}"

            # Add limit
            query += " LIMIT %s"
            params.append(limit)

            results = self.db.fetch_all(query, params)

            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"][:200] + "..." if row["content"] else "",
                    "url": row["url"],
                    "source": row["source"],
                    "category": row["category"],
                    "published_date": row["published_date"].isoformat() if row["published_date"] else None,
                    "author": row["author"],
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error fetching article list: {e}")
            return []

    def _fetch_search_results(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch saved search results"""
        try:
            saved_search_id = data_source.get("saved_search_id")
            if not saved_search_id:
                return {"error": "No saved search ID provided"}

            # Import here to avoid circular dependency
            from .saved_search_manager import SavedSearchManager

            manager = SavedSearchManager(self.db)
            results = manager.execute_saved_search(
                saved_search_id, user_id or "system", limit=data_source.get("limit", 50)
            )

            return results

        except Exception as e:
            logger.error(f"Error fetching search results: {e}")
            return {"error": str(e)}

    def _fetch_chart_line_data(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch line chart data"""
        try:
            metric = data_source.get("metric", "article_count")
            time_range = data_source.get("time_range", "7d")
            group_by = data_source.get("group_by", "day")

            start_date = self._parse_time_range(time_range)

            # Determine date truncation based on group_by
            date_trunc = {"hour": "hour", "day": "day", "week": "week", "month": "month"}.get(group_by, "day")

            if metric == "article_count":
                query = """
                SELECT 
                    DATE_TRUNC('{date_trunc}', scraped_at) as date,
                    COUNT(*) as value
                FROM articles
                WHERE organization_id = %s
                  AND scraped_at >= %s
                GROUP BY date
                ORDER BY date
                """
                params = [organization_id, start_date]

            elif metric == "source_diversity":
                query = """
                SELECT 
                    DATE_TRUNC('{date_trunc}', scraped_at) as date,
                    COUNT(DISTINCT source) as value
                FROM articles
                WHERE organization_id = %s
                  AND scraped_at >= %s
                GROUP BY date
                ORDER BY date
                """
                params = [organization_id, start_date]

            else:
                return {"error": f"Unknown metric: {metric}"}

            results = self.db.fetch_all(query, params)

            return {
                "labels": [row["date"].isoformat() for row in results],
                "datasets": [{"label": metric.replace("_", " ").title(), "data": [row["value"] for row in results]}],
            }

        except Exception as e:
            logger.error(f"Error fetching line chart data: {e}")
            return {"error": str(e)}

    def _fetch_chart_bar_data(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch bar chart data"""
        try:
            metric = data_source.get("metric", "articles_by_source")
            limit = data_source.get("limit", 10)
            time_range = data_source.get("time_range", "7d")

            start_date = self._parse_time_range(time_range)

            if metric == "articles_by_source":
                query = """
                SELECT 
                    source as label,
                    COUNT(*) as value
                FROM articles
                WHERE organization_id = %s
                  AND scraped_at >= %s
                GROUP BY source
                ORDER BY value DESC
                LIMIT %s
                """
                params = [organization_id, start_date, limit]

            elif metric == "articles_by_category":
                query = """
                SELECT 
                    COALESCE(category, 'Uncategorized') as label,
                    COUNT(*) as value
                FROM articles
                WHERE organization_id = %s
                  AND scraped_at >= %s
                GROUP BY category
                ORDER BY value DESC
                LIMIT %s
                """
                params = [organization_id, start_date, limit]

            else:
                return {"error": f"Unknown metric: {metric}"}

            results = self.db.fetch_all(query, params)

            return {
                "labels": [row["label"] for row in results],
                "datasets": [{"label": metric.replace("_", " ").title(), "data": [row["value"] for row in results]}],
            }

        except Exception as e:
            logger.error(f"Error fetching bar chart data: {e}")
            return {"error": str(e)}

    def _fetch_chart_pie_data(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch pie chart data"""
        try:
            metric = data_source.get("metric", "source_distribution")
            limit = data_source.get("limit", 8)
            time_range = data_source.get("time_range", "7d")

            start_date = self._parse_time_range(time_range)

            query = """
            SELECT 
                source as label,
                COUNT(*) as value
            FROM articles
            WHERE organization_id = %s
              AND scraped_at >= %s
            GROUP BY source
            ORDER BY value DESC
            LIMIT %s
            """

            results = self.db.fetch_all(query, [organization_id, start_date, limit])

            # Get total for "Others" calculation
            total_query = """
            SELECT COUNT(*) as total
            FROM articles
            WHERE organization_id = %s AND scraped_at >= %s
            """

            total_result = self.db.fetch_one(total_query, [organization_id, start_date])
            total = total_result["total"] if total_result else 0

            # Calculate "Others" if needed
            labels = [row["label"] for row in results]
            values = [row["value"] for row in results]

            shown_total = sum(values)
            if total > shown_total:
                labels.append("Others")
                values.append(total - shown_total)

            return {"labels": labels, "datasets": [{"data": values}]}

        except Exception as e:
            logger.error(f"Error fetching pie chart data: {e}")
            return {"error": str(e)}

    def _fetch_metric_card_data(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch metric card data"""
        try:
            metric = data_source.get("metric")
            aggregation = data_source.get("aggregation", "count")
            time_range = data_source.get("time_range", "24h")
            comparison = data_source.get("comparison", False)

            current_start = self._parse_time_range(time_range)

            # Get current period value
            current_value = self._get_metric_value(metric, aggregation, organization_id, current_start)

            result = {"value": current_value, "metric": metric, "aggregation": aggregation}

            # Add comparison if requested
            if comparison:
                # Calculate previous period
                time_delta = datetime.now(timezone.utc) - current_start
                previous_start = current_start - time_delta
                previous_end = current_start

                previous_value = self._get_metric_value(
                    metric, aggregation, organization_id, previous_start, previous_end
                )

                if previous_value and previous_value > 0:
                    change = ((current_value - previous_value) / previous_value) * 100
                    result["comparison"] = {
                        "previous_value": previous_value,
                        "change_percent": round(change, 1),
                        "trend": "up" if change > 0 else "down" if change < 0 else "stable",
                    }

            return result

        except Exception as e:
            logger.error(f"Error fetching metric card data: {e}")
            return {"error": str(e)}

    def _fetch_source_stats(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch source statistics"""
        try:
            limit = data_source.get("limit", 10)
            time_range = data_source.get("time_range", "7d")

            start_date = self._parse_time_range(time_range)

            query = """
            SELECT 
                s.name as source_name,
                s.url as source_url,
                COUNT(a.id) as article_count,
                MAX(a.scraped_at) as last_article,
                COUNT(DISTINCT DATE(a.scraped_at)) as active_days,
                AVG(LENGTH(a.content)) as avg_content_length
            FROM sources s
            LEFT JOIN articles a ON s.name = a.source 
                AND a.organization_id = %s
                AND a.scraped_at >= %s
            WHERE s.organization_id = %s
            GROUP BY s.name, s.url
            ORDER BY article_count DESC
            LIMIT %s
            """

            results = self.db.fetch_all(query, [organization_id, start_date, organization_id, limit])

            return [
                {
                    "source": row["source_name"],
                    "url": row["source_url"],
                    "article_count": row["article_count"],
                    "last_article": row["last_article"].isoformat() if row["last_article"] else None,
                    "active_days": row["active_days"],
                    "avg_content_length": int(row["avg_content_length"]) if row["avg_content_length"] else 0,
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error fetching source stats: {e}")
            return []

    def _fetch_keyword_cloud(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch keyword cloud data"""
        try:
            limit = data_source.get("limit", 50)
            time_range = data_source.get("time_range", "24h")
            min_frequency = data_source.get("min_frequency", 2)

            start_date = self._parse_time_range(time_range)

            # Use full-text search statistics
            query = """
            SELECT 
                word,
                ndoc as frequency
            FROM ts_stat(
                'SELECT search_vector FROM articles 
                 WHERE organization_id = ''' || %s || '''
                   AND scraped_at >= ''' || %s || ''''
            )
            WHERE length(word) > 3
              AND ndoc >= %s
            ORDER BY ndoc DESC
            LIMIT %s
            """

            results = self.db.fetch_all(query, [organization_id, start_date.isoformat(), min_frequency, limit])

            if results:
                max_freq = results[0]["frequency"]

                return [
                    {
                        "text": row["word"],
                        "value": row["frequency"],
                        "size": int(20 + (row["frequency"] / max_freq) * 30),
                    }
                    for row in results
                ]

            return []

        except Exception as e:
            logger.error(f"Error fetching keyword cloud: {e}")
            return []

    def _fetch_alert_feed(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch alert feed data"""
        try:
            limit = data_source.get("limit", 20)
            include_dismissed = data_source.get("include_dismissed", False)

            query = """
            SELECT 
                al.id,
                al.resource_id,
                al.action,
                al.details,
                al.created_at,
                ss.name as alert_name,
                ss.query as alert_query
            FROM activity_log al
            JOIN saved_searches ss ON al.resource_id = ss.id::text
            WHERE al.organization_id = %s
              AND al.action = 'alert_triggered'
            """

            params = [organization_id]

            if user_id:
                query += " AND (ss.user_id = %s OR ss.is_shared = true)"
                params.append(user_id)

            query += " ORDER BY al.created_at DESC LIMIT %s"
            params.append(limit)

            results = self.db.fetch_all(query, params)

            return [
                {
                    "id": row["id"],
                    "alert_name": row["alert_name"],
                    "triggered_at": row["created_at"].isoformat(),
                    "result_count": row["details"].get("result_count", 0) if row["details"] else 0,
                    "alert_query": row["alert_query"],
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error fetching alert feed: {e}")
            return []

    def _fetch_activity_timeline(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch activity timeline data"""
        try:
            limit = data_source.get("limit", 50)
            activity_types = data_source.get("types", ["all"])

            query = """
            SELECT 
                id, user_id, action, resource_type, resource_id,
                details, ip_address, user_agent, created_at
            FROM activity_log
            WHERE organization_id = %s
            """

            params = [organization_id]

            if user_id and "all" not in activity_types:
                query += " AND user_id = %s"
                params.append(user_id)

            if "all" not in activity_types:
                placeholders = ",".join(["%s"] * len(activity_types))
                query += f" AND action IN ({placeholders})"
                params.extend(activity_types)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            results = self.db.fetch_all(query, params)

            return [
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "action": row["action"],
                    "resource_type": row["resource_type"],
                    "resource_id": row["resource_id"],
                    "details": row["details"],
                    "timestamp": row["created_at"].isoformat(),
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error fetching activity timeline: {e}")
            return []

    def _fetch_custom_html(
        self, data_source: Dict[str, Any], organization_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch custom HTML widget data"""
        try:
            html_content = data_source.get("html", "")
            css_content = data_source.get("css", "")

            return {"html": html_content, "css": css_content, "sandbox": True}  # Always sandbox custom HTML

        except Exception as e:
            logger.error(f"Error fetching custom HTML: {e}")
            return {"error": str(e)}

    async def _get_metric_value(
        self,
        metric: str,
        aggregation: str,
        organization_id: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Union[int, float]:
        """Get single metric value"""
        try:
            if not end_date:
                end_date = datetime.now(timezone.utc)

            if metric == "article_count":
                query = """
                SELECT COUNT(*) as value
                FROM articles
                WHERE organization_id = %s
                  AND scraped_at >= %s
                  AND scraped_at < %s
                """
            elif metric == "unique_sources":
                query = """
                SELECT COUNT(DISTINCT source) as value
                FROM articles
                WHERE organization_id = %s
                  AND scraped_at >= %s
                  AND scraped_at < %s
                """
            elif metric == "active_alerts":
                query = """
                SELECT COUNT(*) as value
                FROM saved_searches
                WHERE organization_id = %s
                  AND is_alert = true
                  AND created_at < %s
                """
                end_date = start_date  # For alerts, we want current count
            else:
                return 0

            result = self.db.fetch_one(query, [organization_id, start_date, end_date])
            return result["value"] if result else 0

        except Exception as e:
            logger.error(f"Error getting metric value: {e}")
            return 0

    def _parse_time_range(self, time_range: str) -> datetime:
        """Parse time range string to datetime"""
        now = datetime.now(timezone.utc)

        if time_range == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Parse relative time ranges
        time_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "12h": timedelta(hours=12),
            "24h": timedelta(hours=24),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
        }

        delta = time_map.get(time_range, timedelta(days=1))
        return now - delta

    def _generate_cache_key(self, widget_id: str, data_source: Dict[str, Any]) -> str:
        """Generate cache key for widget data"""
        # Create stable hash of data source
        source_str = json.dumps(data_source, sort_keys=True)
        source_hash = hashlib.md5(source_str.encode()).hexdigest()

        return f"widget:{widget_id}:{source_hash}"

    def _get_cached_data(self, widget_id: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached widget data"""
        try:
            query = """
            SELECT data, metadata, expires_at
            FROM widget_cache
            WHERE widget_id = %s AND cache_key = %s
              AND expires_at > CURRENT_TIMESTAMP
            """

            result = self.db.fetch_one(query, (widget_id, cache_key))

            if result:
                return {
                    "widget_id": widget_id,
                    "data": result["data"],
                    "metadata": result["metadata"],
                    "last_updated": datetime.now(timezone.utc),
                    "from_cache": True,
                }

            return None

        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None

    def _cache_data(self, widget_id: str, cache_key: str, data: Dict[str, Any], ttl_seconds: int = 300):
        """Cache widget data"""
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

            query = """
            INSERT INTO widget_cache (widget_id, cache_key, data, metadata, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (widget_id, cache_key) 
            DO UPDATE SET
                data = EXCLUDED.data,
                metadata = EXCLUDED.metadata,
                expires_at = EXCLUDED.expires_at,
                created_at = CURRENT_TIMESTAMP
            """

            self.db.execute_query(
                query,
                (widget_id, cache_key, json.dumps(data["data"]), json.dumps(data.get("metadata", {})), expires_at),
            )

        except Exception as e:
            logger.error(f"Error caching data: {e}")

    def _calculate_next_update(self, data_source: Dict[str, Any]) -> Optional[datetime]:
        """Calculate next update time based on data source"""
        refresh_interval = data_source.get("refresh_interval", 300)
        if refresh_interval > 0:
            return datetime.now(timezone.utc) + timedelta(seconds=refresh_interval)
        return None
