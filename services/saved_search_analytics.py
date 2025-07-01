"""
Saved Search Analytics Service for Mimir Enterprise

Provides analytics and insights for saved searches and alerts.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SavedSearchAnalytics:
    """Analytics service for saved searches and alerts"""

    def __init__(self, db_manager):
        """
        Initialize analytics service

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def get_search_usage_metrics(
        self, organization_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage metrics for saved searches

        Args:
            organization_id: Organization ID
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary with usage metrics
        """
        try:
            if not start_date:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if not end_date:
                end_date = datetime.now(timezone.utc)

            # Get execution frequency
            execution_query = """
            SELECT 
                ss.id,
                ss.name,
                ss.is_alert,
                COUNT(DISTINCT DATE(al.created_at)) as days_executed,
                COUNT(al.id) as total_executions,
                MIN(al.created_at) as first_execution,
                MAX(al.created_at) as last_execution
            FROM saved_searches ss
            LEFT JOIN activity_log al ON 
                al.resource_id = ss.id::text AND 
                al.action = 'execute_saved_search' AND
                al.created_at BETWEEN %s AND %s
            WHERE ss.organization_id = %s
            GROUP BY ss.id, ss.name, ss.is_alert
            ORDER BY total_executions DESC
            """

            execution_stats = self.db.fetch_all(execution_query, (start_date, end_date, organization_id))

            # Get user engagement
            user_query = """
            SELECT 
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT id) as total_searches,
                COUNT(DISTINCT id) FILTER (WHERE is_alert = true) as alert_count,
                COUNT(DISTINCT id) FILTER (WHERE is_shared = true) as shared_count,
                AVG(array_length(tags, 1)) as avg_tags_per_search
            FROM saved_searches
            WHERE organization_id = %s
            """

            user_stats = self.db.fetch_one(user_query, (organization_id,))

            # Get alert performance
            alert_query = """
            SELECT 
                COUNT(*) FILTER (WHERE next_check_at < CURRENT_TIMESTAMP) as overdue_alerts,
                COUNT(*) FILTER (WHERE last_checked_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as active_alerts_24h,
                AVG(EXTRACT(EPOCH FROM (last_checked_at - LAG(last_checked_at) OVER (PARTITION BY id ORDER BY last_checked_at)))) as avg_check_interval_seconds
            FROM saved_searches
            WHERE organization_id = %s AND is_alert = true
            """

            alert_stats = self.db.fetch_one(alert_query, (organization_id,))

            return {
                "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
                "execution_stats": execution_stats,
                "user_engagement": user_stats,
                "alert_performance": alert_stats,
            }

        except Exception as e:
            logger.error(f"Error getting search usage metrics: {e}")
            return {}

    def get_popular_search_queries(self, organization_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular search queries

        Args:
            organization_id: Organization ID
            limit: Number of top queries to return

        Returns:
            List of popular queries with usage stats
        """
        try:
            query = """
            SELECT 
                query,
                COUNT(*) as usage_count,
                COUNT(DISTINCT user_id) as unique_users,
                MAX(created_at) as last_used,
                BOOL_OR(is_alert) as has_alert,
                ARRAY_AGG(DISTINCT name) as search_names
            FROM saved_searches
            WHERE organization_id = %s
            GROUP BY query
            ORDER BY usage_count DESC, unique_users DESC
            LIMIT %s
            """

            results = self.db.fetch_all(query, (organization_id, limit))

            # Parse and analyze queries
            popular_queries = []
            for row in results:
                query_data = row["query"]
                if isinstance(query_data, str):
                    query_data = json.loads(query_data)

                # Extract key components
                components = {
                    "text_search": query_data.get("q", ""),
                    "filters": {k: v for k, v in query_data.items() if k not in ["q", "sort_by", "sort_order"]},
                    "sorting": {"sort_by": query_data.get("sort_by"), "sort_order": query_data.get("sort_order")},
                }

                popular_queries.append(
                    {
                        "query_components": components,
                        "usage_count": row["usage_count"],
                        "unique_users": row["unique_users"],
                        "last_used": row["last_used"].isoformat() if row["last_used"] else None,
                        "has_alert": row["has_alert"],
                        "search_names": row["search_names"][:3],  # Limit to 3 names
                    }
                )

            return popular_queries

        except Exception as e:
            logger.error(f"Error getting popular search queries: {e}")
            return []

    def get_alert_effectiveness(self, organization_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze alert effectiveness and engagement

        Args:
            organization_id: Organization ID
            days: Number of days to analyze

        Returns:
            Alert effectiveness metrics
        """
        try:
            # Get alert trigger statistics
            trigger_query = """
            SELECT 
                ss.id,
                ss.name,
                ss.alert_frequency,
                COUNT(al.id) as trigger_count,
                AVG(CAST(al.details->>'result_count' AS INTEGER)) as avg_results_per_trigger,
                MAX(CAST(al.details->>'result_count' AS INTEGER)) as max_results
            FROM saved_searches ss
            LEFT JOIN activity_log al ON 
                al.resource_id = ss.id::text AND 
                al.action = 'alert_triggered' AND
                al.created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
            WHERE ss.organization_id = %s AND ss.is_alert = true
            GROUP BY ss.id, ss.name, ss.alert_frequency
            """

            trigger_stats = self.db.fetch_all(trigger_query, (days, organization_id))

            # Calculate effectiveness metrics
            total_alerts = len(trigger_stats)
            active_alerts = sum(1 for stat in trigger_stats if stat["trigger_count"] > 0)
            total_triggers = sum(stat["trigger_count"] for stat in trigger_stats)

            # Get notification channel usage
            channel_query = """
            SELECT 
                notification_channels,
                COUNT(*) as alert_count
            FROM saved_searches
            WHERE organization_id = %s AND is_alert = true
            GROUP BY notification_channels
            """

            channel_stats = self.db.fetch_all(channel_query, (organization_id,))

            # Process channel statistics
            channel_usage = defaultdict(int)
            for row in channel_stats:
                channels = row["notification_channels"] or []
                for channel in channels:
                    channel_usage[channel] += row["alert_count"]

            # Get alert frequency distribution
            frequency_query = """
            SELECT 
                alert_frequency,
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_checked_at))) as avg_time_since_check
            FROM saved_searches
            WHERE organization_id = %s AND is_alert = true
            GROUP BY alert_frequency
            ORDER BY alert_frequency
            """

            frequency_dist = self.db.fetch_all(frequency_query, (organization_id,))

            return {
                "summary": {
                    "total_alerts": total_alerts,
                    "active_alerts": active_alerts,
                    "inactive_alerts": total_alerts - active_alerts,
                    "total_triggers": total_triggers,
                    "avg_triggers_per_alert": round(total_triggers / active_alerts, 2) if active_alerts > 0 else 0,
                },
                "alert_details": [
                    {
                        "id": stat["id"],
                        "name": stat["name"],
                        "frequency_seconds": stat["alert_frequency"],
                        "trigger_count": stat["trigger_count"],
                        "avg_results": stat["avg_results_per_trigger"],
                        "max_results": stat["max_results"],
                        "effectiveness_score": self._calculate_effectiveness_score(stat),
                    }
                    for stat in trigger_stats
                ],
                "channel_usage": dict(channel_usage),
                "frequency_distribution": [
                    {
                        "frequency_seconds": row["alert_frequency"],
                        "frequency_label": self._format_frequency(row["alert_frequency"]),
                        "count": row["count"],
                        "avg_delay_seconds": row["avg_time_since_check"],
                    }
                    for row in frequency_dist
                ],
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Error analyzing alert effectiveness: {e}")
            return {}

    def get_search_performance_metrics(
        self, organization_id: str, search_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for saved searches

        Args:
            organization_id: Organization ID
            search_ids: Optional list of specific search IDs

        Returns:
            Performance metrics
        """
        try:
            # Build query
            base_query = """
            SELECT 
                ss.id,
                ss.name,
                ss.query,
                COUNT(al.id) as execution_count,
                AVG(CAST(al.details->>'execution_time_ms' AS FLOAT)) as avg_execution_time,
                MIN(CAST(al.details->>'execution_time_ms' AS FLOAT)) as min_execution_time,
                MAX(CAST(al.details->>'execution_time_ms' AS FLOAT)) as max_execution_time,
                AVG(CAST(al.details->>'result_count' AS INTEGER)) as avg_result_count
            FROM saved_searches ss
            LEFT JOIN activity_log al ON 
                al.resource_id = ss.id::text AND 
                al.action = 'execute_saved_search'
            WHERE ss.organization_id = %s
            """

            params = [organization_id]

            if search_ids:
                base_query += " AND ss.id = ANY(%s)"
                params.append(search_ids)

            base_query += " GROUP BY ss.id, ss.name, ss.query"

            results = self.db.fetch_all(base_query, params)

            # Calculate performance scores
            performance_data = []
            for row in results:
                query_data = row["query"]
                if isinstance(query_data, str):
                    query_data = json.loads(query_data)

                # Calculate complexity score based on query
                complexity_score = self._calculate_query_complexity(query_data)

                performance_data.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "execution_count": row["execution_count"],
                        "performance": {
                            "avg_execution_time_ms": round(row["avg_execution_time"] or 0, 2),
                            "min_execution_time_ms": round(row["min_execution_time"] or 0, 2),
                            "max_execution_time_ms": round(row["max_execution_time"] or 0, 2),
                            "avg_result_count": row["avg_result_count"] or 0,
                        },
                        "complexity_score": complexity_score,
                        "efficiency_score": self._calculate_efficiency_score(
                            row["avg_execution_time"], row["avg_result_count"], complexity_score
                        ),
                    }
                )

            # Sort by efficiency score
            performance_data.sort(key=lambda x: x["efficiency_score"], reverse=True)

            return {
                "search_performance": performance_data,
                "summary": {
                    "total_searches": len(performance_data),
                    "avg_complexity": round(
                        sum(p["complexity_score"] for p in performance_data) / len(performance_data), 2
                    )
                    if performance_data
                    else 0,
                    "high_performers": sum(1 for p in performance_data if p["efficiency_score"] > 0.7),
                    "low_performers": sum(1 for p in performance_data if p["efficiency_score"] < 0.3),
                },
            }

        except Exception as e:
            logger.error(f"Error getting search performance metrics: {e}")
            return {}

    def get_tag_analytics(self, organization_id: str) -> Dict[str, Any]:
        """
        Analyze tag usage across saved searches

        Args:
            organization_id: Organization ID

        Returns:
            Tag analytics
        """
        try:
            # Get tag frequency
            tag_query = """
            SELECT 
                unnest(tags) as tag,
                COUNT(*) as usage_count,
                COUNT(DISTINCT user_id) as unique_users,
                BOOL_OR(is_alert) as used_in_alerts,
                BOOL_OR(is_shared) as used_in_shared
            FROM saved_searches
            WHERE organization_id = %s AND array_length(tags, 1) > 0
            GROUP BY tag
            ORDER BY usage_count DESC
            """

            tag_stats = self.db.fetch_all(tag_query, (organization_id,))

            # Get co-occurrence patterns
            cooccurrence_query = """
            SELECT 
                t1.tag as tag1,
                t2.tag as tag2,
                COUNT(*) as cooccurrence_count
            FROM (
                SELECT id, unnest(tags) as tag
                FROM saved_searches
                WHERE organization_id = %s
            ) t1
            JOIN (
                SELECT id, unnest(tags) as tag
                FROM saved_searches
                WHERE organization_id = %s
            ) t2 ON t1.id = t2.id AND t1.tag < t2.tag
            GROUP BY t1.tag, t2.tag
            HAVING COUNT(*) > 1
            ORDER BY cooccurrence_count DESC
            LIMIT 20
            """

            cooccurrence = self.db.fetch_all(cooccurrence_query, (organization_id, organization_id))

            return {
                "tag_frequency": [
                    {
                        "tag": row["tag"],
                        "usage_count": row["usage_count"],
                        "unique_users": row["unique_users"],
                        "used_in_alerts": row["used_in_alerts"],
                        "used_in_shared": row["used_in_shared"],
                    }
                    for row in tag_stats
                ],
                "tag_relationships": [
                    {"tags": [row["tag1"], row["tag2"]], "cooccurrence_count": row["cooccurrence_count"]}
                    for row in cooccurrence
                ],
                "summary": {
                    "unique_tags": len(tag_stats),
                    "total_tag_usage": sum(row["usage_count"] for row in tag_stats),
                    "avg_tags_per_search": round(sum(row["usage_count"] for row in tag_stats) / len(tag_stats), 2)
                    if tag_stats
                    else 0,
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing tags: {e}")
            return {}

    def _calculate_effectiveness_score(self, alert_stats: Dict[str, Any]) -> float:
        """Calculate effectiveness score for an alert"""
        if alert_stats["trigger_count"] == 0:
            return 0.0

        # Factors: trigger count, average results, consistency
        trigger_score = min(alert_stats["trigger_count"] / 100, 1.0)
        result_score = min((alert_stats["avg_results_per_trigger"] or 0) / 10, 1.0)

        # Combined score
        effectiveness = trigger_score * 0.6 + result_score * 0.4
        return round(effectiveness, 2)

    def _calculate_query_complexity(self, query: Dict[str, Any]) -> float:
        """Calculate complexity score for a query"""
        complexity = 0.0

        # Text search
        if query.get("q"):
            complexity += 0.3

        # Filters
        filter_count = sum(1 for k, v in query.items() if k not in ["q", "sort_by", "sort_order"] and v)
        complexity += min(filter_count * 0.15, 0.5)

        # Special operations
        if query.get("semantic_search"):
            complexity += 0.2

        return round(min(complexity, 1.0), 2)

    def _calculate_efficiency_score(
        self, avg_execution_time: Optional[float], avg_result_count: Optional[float], complexity: float
    ) -> float:
        """Calculate efficiency score"""
        if not avg_execution_time or avg_execution_time == 0:
            return 0.0

        # Normalize execution time (lower is better)
        time_score = max(0, 1 - (avg_execution_time / 5000))  # 5 seconds as baseline

        # Result relevance (some results are good)
        result_score = min((avg_result_count or 0) / 50, 1.0)

        # Adjust for complexity
        complexity_adjusted = time_score * (1 + complexity * 0.2)

        # Combined score
        efficiency = complexity_adjusted * 0.7 + result_score * 0.3
        return round(min(efficiency, 1.0), 2)

    def _format_frequency(self, seconds: int) -> str:
        """Format frequency in human-readable form"""
        if seconds < 3600:
            return f"{seconds // 60} minutes"
        elif seconds < 86400:
            return f"{seconds // 3600} hours"
        else:
            return f"{seconds // 86400} days"
