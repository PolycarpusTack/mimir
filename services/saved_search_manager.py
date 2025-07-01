"""
Saved Search Manager for Mimir Enterprise

Handles saved search management, execution, and alert generation.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlertFrequency(Enum):
    """Alert frequency options in seconds"""

    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    THIRTY_MINUTES = 1800
    HOURLY = 3600
    DAILY = 86400
    WEEKLY = 604800


@dataclass
class SavedSearchConfig:
    """Configuration for a saved search"""

    id: str
    organization_id: str
    user_id: str
    name: str
    description: Optional[str]
    query: Dict[str, Any]
    is_alert: bool
    alert_frequency: int
    notification_channels: List[str]
    notification_settings: Dict[str, Any]
    is_shared: bool
    shared_with: List[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    last_checked_at: Optional[datetime]
    next_check_at: Optional[datetime]


class SavedSearchManager:
    """Manages saved searches and alerts"""

    def __init__(self, db_manager, search_service=None, notification_service=None):
        """
        Initialize saved search manager

        Args:
            db_manager: Database manager instance
            search_service: Search service for executing queries
            notification_service: Service for sending notifications
        """
        self.db = db_manager
        self.search_service = search_service
        self.notification_service = notification_service

    def create_saved_search(
        self,
        organization_id: str,
        user_id: str,
        name: str,
        query: Dict[str, Any],
        description: Optional[str] = None,
        is_alert: bool = False,
        alert_frequency: int = AlertFrequency.HOURLY.value,
        notification_channels: Optional[List[str]] = None,
        notification_settings: Optional[Dict[str, Any]] = None,
        is_shared: bool = False,
        shared_with: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> SavedSearchConfig:
        """
        Create a new saved search

        Args:
            organization_id: Organization ID
            user_id: User ID
            name: Search name
            query: Search query parameters
            description: Optional description
            is_alert: Whether this is an alert
            alert_frequency: Alert check frequency in seconds
            notification_channels: List of notification channels
            notification_settings: Channel-specific settings
            is_shared: Whether search is shared
            shared_with: List of user IDs to share with
            tags: Optional tags

        Returns:
            Created saved search configuration
        """
        try:
            search_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            # Calculate next check time if this is an alert
            next_check_at = None
            if is_alert:
                next_check_at = now + timedelta(seconds=alert_frequency)

            # Insert into database
            query_str = """
            INSERT INTO saved_searches (
                id, organization_id, user_id, name, description, query,
                is_alert, alert_frequency, notification_channels, notification_settings,
                is_shared, shared_with, tags, created_at, updated_at, next_check_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING *
            """

            params = (
                search_id,
                organization_id,
                user_id,
                name,
                description,
                json.dumps(query),
                is_alert,
                alert_frequency,
                notification_channels or [],
                notification_settings or {},
                is_shared,
                shared_with or [],
                tags or [],
                now,
                now,
                next_check_at,
            )

            result = self.db.fetch_one(query_str, params)

            if not result:
                raise ValueError("Failed to create saved search")

            logger.info(f"Created saved search '{name}' (ID: {search_id}) for user {user_id}")

            return self._dict_to_saved_search(result)

        except Exception as e:
            logger.error(f"Error creating saved search: {e}")
            raise

    def get_saved_search(self, search_id: str, user_id: str) -> Optional[SavedSearchConfig]:
        """
        Get a saved search by ID

        Args:
            search_id: Saved search ID
            user_id: User ID for access control

        Returns:
            Saved search if found and accessible
        """
        try:
            query = """
            SELECT * FROM saved_searches 
            WHERE id = %s AND (user_id = %s OR is_shared = true OR %s = ANY(shared_with))
            """

            result = self.db.fetch_one(query, (search_id, user_id, user_id))

            return self._dict_to_saved_search(result) if result else None

        except Exception as e:
            logger.error(f"Error getting saved search {search_id}: {e}")
            return None

    def list_user_saved_searches(
        self, organization_id: str, user_id: str, include_shared: bool = True, only_alerts: bool = False
    ) -> List[SavedSearchConfig]:
        """
        List saved searches for a user

        Args:
            organization_id: Organization ID
            user_id: User ID
            include_shared: Include shared searches
            only_alerts: Only return alert searches

        Returns:
            List of saved searches
        """
        try:
            conditions = ["organization_id = %s"]
            params = [organization_id]

            if include_shared:
                conditions.append("(user_id = %s OR is_shared = true OR %s = ANY(shared_with))")
                params.extend([user_id, user_id])
            else:
                conditions.append("user_id = %s")
                params.append(user_id)

            if only_alerts:
                conditions.append("is_alert = true")

            query = """
            SELECT * FROM saved_searches 
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            """

            results = self.db.fetch_all(query, params)

            return [self._dict_to_saved_search(r) for r in results]

        except Exception as e:
            logger.error(f"Error listing saved searches: {e}")
            return []

    def update_saved_search(self, search_id: str, user_id: str, **updates) -> Optional[SavedSearchConfig]:
        """
        Update a saved search

        Args:
            search_id: Saved search ID
            user_id: User ID for access control
            **updates: Fields to update

        Returns:
            Updated saved search if successful
        """
        try:
            # Check ownership
            existing = self.get_saved_search(search_id, user_id)
            if not existing or existing.user_id != user_id:
                logger.warning(f"User {user_id} cannot update saved search {search_id}")
                return None

            # Build update query
            update_fields = []
            params = []

            allowed_fields = {
                "name",
                "description",
                "query",
                "is_alert",
                "alert_frequency",
                "notification_channels",
                "notification_settings",
                "is_shared",
                "shared_with",
                "tags",
            }

            for field, value in updates.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    if field == "query":
                        params.append(json.dumps(value))
                    else:
                        params.append(value)

            if not update_fields:
                return existing

            # Add updated_at
            update_fields.append("updated_at = %s")
            params.append(datetime.now(timezone.utc))

            # Update next_check_at if alert settings changed
            if "is_alert" in updates or "alert_frequency" in updates:
                is_alert = updates.get("is_alert", existing.is_alert)
                alert_frequency = updates.get("alert_frequency", existing.alert_frequency)

                if is_alert:
                    next_check = datetime.now(timezone.utc) + timedelta(seconds=alert_frequency)
                    update_fields.append("next_check_at = %s")
                    params.append(next_check)
                else:
                    update_fields.append("next_check_at = NULL")

            params.append(search_id)

            query = """
            UPDATE saved_searches 
            SET {', '.join(update_fields)}
            WHERE id = %s
            RETURNING *
            """

            result = self.db.fetch_one(query, params)

            return self._dict_to_saved_search(result) if result else None

        except Exception as e:
            logger.error(f"Error updating saved search {search_id}: {e}")
            return None

    def delete_saved_search(self, search_id: str, user_id: str) -> bool:
        """
        Delete a saved search

        Args:
            search_id: Saved search ID
            user_id: User ID for access control

        Returns:
            True if deleted successfully
        """
        try:
            # Check ownership
            existing = self.get_saved_search(search_id, user_id)
            if not existing or existing.user_id != user_id:
                logger.warning(f"User {user_id} cannot delete saved search {search_id}")
                return False

            query = "DELETE FROM saved_searches WHERE id = %s"
            result = self.db.execute_query(query, (search_id,))

            if result:
                logger.info(f"Deleted saved search {search_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting saved search {search_id}: {e}")
            return False

    def execute_saved_search(self, search_id: str, user_id: str, limit: int = 100) -> Dict[str, Any]:
        """
        Execute a saved search

        Args:
            search_id: Saved search ID
            user_id: User ID for access control
            limit: Maximum results

        Returns:
            Search results
        """
        try:
            # Get saved search
            saved_search = self.get_saved_search(search_id, user_id)
            if not saved_search:
                raise ValueError("Saved search not found or not accessible")

            # Execute search using search service
            if self.search_service:
                results = self.search_service.search(
                    organization_id=saved_search.organization_id, **saved_search.query, limit=limit
                )
            else:
                # Fallback to direct database query
                results = self._execute_search_query(saved_search.organization_id, saved_search.query, limit)

            # Update last checked time
            self._update_last_checked(search_id)

            return {
                "saved_search_id": search_id,
                "saved_search_name": saved_search.name,
                "query": saved_search.query,
                "results": results,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error executing saved search {search_id}: {e}")
            raise

    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check all due alerts and generate notifications

        Returns:
            List of processed alerts with results
        """
        try:
            # Get alerts that need checking
            query = """
            SELECT * FROM saved_searches 
            WHERE is_alert = true 
              AND next_check_at <= %s
            ORDER BY next_check_at
            LIMIT 100
            """

            now = datetime.now(timezone.utc)
            alerts = self.db.fetch_all(query, (now,))

            processed_alerts = []

            for alert_dict in alerts:
                alert = self._dict_to_saved_search(alert_dict)

                try:
                    # Execute the search
                    results = self._execute_search_query(alert.organization_id, alert.query, limit=50)

                    # Check if there are new results since last check
                    new_results = []
                    if alert.last_checked_at and results:
                        for result in results:
                            # Assuming results have a 'scraped_at' field
                            if result.get("scraped_at") > alert.last_checked_at:
                                new_results.append(result)
                    else:
                        new_results = results

                    # Send notifications if there are new results
                    if new_results and self.notification_service:
                        self._send_alert_notifications(alert, new_results)

                    # Update next check time
                    self._update_alert_schedule(alert.id, alert.alert_frequency)

                    processed_alerts.append(
                        {
                            "alert_id": alert.id,
                            "alert_name": alert.name,
                            "new_results_count": len(new_results),
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                            "status": "success",
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing alert {alert.id}: {e}")
                    processed_alerts.append(
                        {
                            "alert_id": alert.id,
                            "alert_name": alert.name,
                            "error": str(e),
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                            "status": "failed",
                        }
                    )

            return processed_alerts

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []

    def _execute_search_query(
        self, organization_id: str, query_params: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Execute search query directly against database"""
        try:
            # Build WHERE clause from query parameters
            conditions = ["organization_id = %s"]
            params = [organization_id]

            # Text search
            if query_params.get("q"):
                conditions.append("search_vector @@ websearch_to_tsquery('english', %s)")
                params.append(query_params["q"])

            # Filters
            if query_params.get("source"):
                conditions.append("source ILIKE %s")
                params.append(f"%{query_params['source']}%")

            if query_params.get("category"):
                conditions.append("category = %s")
                params.append(query_params["category"])

            if query_params.get("language"):
                conditions.append("language = %s")
                params.append(query_params["language"])

            if query_params.get("start_date"):
                conditions.append("published_date >= %s")
                params.append(query_params["start_date"])

            if query_params.get("end_date"):
                conditions.append("published_date <= %s")
                params.append(query_params["end_date"])

            if query_params.get("tags"):
                conditions.append("tags && %s")
                params.append(query_params["tags"])

            # Build query
            query = """
            SELECT 
                id, url, title, content, summary, published_date, scraped_at,
                source, author, category, language, tags
            FROM articles 
            WHERE {' AND '.join(conditions)}
            ORDER BY published_date DESC, scraped_at DESC
            LIMIT %s
            """

            params.append(limit)

            results = self.db.fetch_all(query, params)

            # Convert to dictionaries
            return [dict(r) for r in results] if results else []

        except Exception as e:
            logger.error(f"Error executing search query: {e}")
            return []

    def _send_alert_notifications(self, alert: SavedSearchConfig, new_results: List[Dict[str, Any]]) -> None:
        """Send notifications for alert results"""
        try:
            if not self.notification_service:
                logger.warning("No notification service configured")
                return

            # Prepare notification content
            notification_data = {
                "alert_id": alert.id,
                "alert_name": alert.name,
                "result_count": len(new_results),
                "results": new_results[:10],  # Limit to first 10
                "query": alert.query,
                "user_id": alert.user_id,
                "organization_id": alert.organization_id,
            }

            # Send to each configured channel
            for channel in alert.notification_channels:
                try:
                    channel_settings = alert.notification_settings.get(channel, {})

                    # Get user's notification preferences
                    pref_query = """
                    SELECT enabled FROM notification_preferences
                    WHERE user_id = %s AND channel = %s AND notification_type = 'saved_search_alert'
                    """
                    pref = self.db.fetch_one(pref_query, (alert.user_id, channel))

                    # Skip if user has disabled this notification type
                    if pref and not pref["enabled"]:
                        logger.info(f"Skipping {channel} notification for alert {alert.id} - disabled by user")
                        continue

                    # Send notification asynchronously
                    import asyncio

                    loop = asyncio.get_event_loop()
                    success = loop.run_until_complete(
                        self.notification_service.send_notification(
                            channel=channel,
                            recipient=channel_settings.get("recipient", alert.user_id),
                            notification_type="saved_search_alert",
                            data=notification_data,
                            template_id=channel_settings.get("template_id"),
                        )
                    )

                    if success:
                        logger.info(f"Sent alert notification for {alert.id} via {channel}")
                    else:
                        logger.error(f"Failed to send notification for alert {alert.id} via {channel}")

                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {e}")

        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")

    def _update_last_checked(self, search_id: str) -> None:
        """Update last checked timestamp"""
        try:
            query = """
            UPDATE saved_searches 
            SET last_checked_at = %s
            WHERE id = %s
            """

            self.db.execute_query(query, (datetime.now(timezone.utc), search_id))

        except Exception as e:
            logger.error(f"Error updating last checked time: {e}")

    def _update_alert_schedule(self, alert_id: str, frequency: int) -> None:
        """Update alert schedule after checking"""
        try:
            now = datetime.now(timezone.utc)
            next_check = now + timedelta(seconds=frequency)

            query = """
            UPDATE saved_searches 
            SET last_checked_at = %s, next_check_at = %s
            WHERE id = %s
            """

            self.db.execute_query(query, (now, next_check, alert_id))

        except Exception as e:
            logger.error(f"Error updating alert schedule: {e}")

    def _dict_to_saved_search(self, data: Dict[str, Any]) -> SavedSearchConfig:
        """Convert database row to SavedSearchConfig"""
        if not data:
            return None

        # Parse JSON fields
        query = data["query"]
        if isinstance(query, str):
            query = json.loads(query)

        return SavedSearchConfig(
            id=data["id"],
            organization_id=data["organization_id"],
            user_id=data["user_id"],
            name=data["name"],
            description=data.get("description"),
            query=query,
            is_alert=data["is_alert"],
            alert_frequency=data["alert_frequency"],
            notification_channels=data.get("notification_channels", []),
            notification_settings=data.get("notification_settings", {}),
            is_shared=data["is_shared"],
            shared_with=data.get("shared_with", []),
            tags=data.get("tags", []),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            last_checked_at=data.get("last_checked_at"),
            next_check_at=data.get("next_check_at"),
        )

    def get_alert_statistics(self, organization_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get alert statistics for an organization

        Args:
            organization_id: Organization ID
            days: Number of days to analyze

        Returns:
            Alert statistics
        """
        try:
            # Total alerts
            total_query = """
            SELECT 
                COUNT(*) as total_alerts,
                COUNT(*) FILTER (WHERE is_alert = true) as active_alerts,
                COUNT(DISTINCT user_id) as unique_users
            FROM saved_searches
            WHERE organization_id = %s
            """

            total_stats = self.db.fetch_one(total_query, (organization_id,))

            # Alert frequency distribution
            frequency_query = """
            SELECT 
                alert_frequency,
                COUNT(*) as count
            FROM saved_searches
            WHERE organization_id = %s AND is_alert = true
            GROUP BY alert_frequency
            ORDER BY alert_frequency
            """

            frequency_dist = self.db.fetch_all(frequency_query, (organization_id,))

            # Recent alert activity
            activity_query = """
            SELECT 
                DATE(last_checked_at) as date,
                COUNT(*) as alerts_checked
            FROM saved_searches
            WHERE organization_id = %s 
              AND is_alert = true
              AND last_checked_at >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY DATE(last_checked_at)
            ORDER BY date DESC
            """

            recent_activity = self.db.fetch_all(activity_query, (organization_id, days))

            return {
                "summary": total_stats,
                "frequency_distribution": frequency_dist,
                "recent_activity": recent_activity,
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {}
