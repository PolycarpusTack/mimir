"""
Alert Scheduler Service for Mimir Enterprise

Handles scheduling and execution of saved search alerts using Celery or asyncio.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlertJob:
    """Represents a scheduled alert job"""

    alert_id: str
    organization_id: str
    user_id: str
    name: str
    query: Dict[str, Any]
    notification_channels: List[str]
    notification_settings: Dict[str, Any]
    next_check_at: datetime
    frequency: int


class AlertScheduler:
    """Schedules and executes saved search alerts"""

    def __init__(
        self,
        saved_search_manager,
        notification_service=None,
        max_workers: int = 4,
        check_interval: int = 60,  # Check for due alerts every minute
    ):
        """
        Initialize alert scheduler

        Args:
            saved_search_manager: SavedSearchManager instance
            notification_service: Notification service instance
            max_workers: Maximum concurrent alert executions
            check_interval: Seconds between checking for due alerts
        """
        self.saved_search_manager = saved_search_manager
        self.notification_service = notification_service
        self.max_workers = max_workers
        self.check_interval = check_interval
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._task = None

        # Track active jobs to prevent duplicates
        self._active_jobs: Dict[str, AlertJob] = {}

    async def start(self):
        """Start the alert scheduler"""
        if self._running:
            logger.warning("Alert scheduler already running")
            return

        self._running = True
        logger.info(f"Starting alert scheduler (check interval: {self.check_interval}s)")

        # Start the main scheduling loop
        self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """Stop the alert scheduler"""
        logger.info("Stopping alert scheduler...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self.executor.shutdown(wait=True)
        logger.info("Alert scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                # Check for due alerts
                await self._check_and_execute_alerts()

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_and_execute_alerts(self):
        """Check for due alerts and execute them"""
        try:
            # Get alerts that need checking
            due_alerts = await self._get_due_alerts()

            if not due_alerts:
                return

            logger.info(f"Found {len(due_alerts)} alerts to check")

            # Execute alerts concurrently
            tasks = []
            for alert in due_alerts:
                # Skip if already being processed
                if alert.alert_id in self._active_jobs:
                    logger.debug(f"Alert {alert.alert_id} already being processed")
                    continue

                # Mark as active
                self._active_jobs[alert.alert_id] = alert

                # Create execution task
                task = asyncio.create_task(self._execute_alert(alert))
                tasks.append(task)

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    async def _get_due_alerts(self) -> List[AlertJob]:
        """Get alerts that are due for checking"""
        try:
            query = """
            SELECT 
                id, organization_id, user_id, name, query,
                notification_channels, notification_settings,
                next_check_at, alert_frequency
            FROM saved_searches 
            WHERE is_alert = true 
              AND next_check_at <= %s
              AND (next_check_at IS NULL OR next_check_at < %s)
            ORDER BY next_check_at
            LIMIT 50
            """

            now = datetime.now(timezone.utc)
            # Prevent checking alerts too frequently (minimum 1 minute between checks)
            cutoff_time = now + timedelta(seconds=60)

            results = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.saved_search_manager.db.fetch_all, query, (now, cutoff_time)
            )

            alerts = []
            for row in results:
                # Parse JSON fields
                query_data = row["query"]
                if isinstance(query_data, str):
                    query_data = json.loads(query_data)

                alert = AlertJob(
                    alert_id=row["id"],
                    organization_id=row["organization_id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    query=query_data,
                    notification_channels=row.get("notification_channels", []),
                    notification_settings=row.get("notification_settings", {}),
                    next_check_at=row["next_check_at"],
                    frequency=row["alert_frequency"],
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Error getting due alerts: {e}")
            return []

    async def _execute_alert(self, alert: AlertJob):
        """Execute a single alert"""
        try:
            logger.info(f"Executing alert '{alert.name}' (ID: {alert.alert_id})")

            # Execute the saved search
            results = await self._execute_saved_search(alert)

            # Check for new results
            new_results = await self._filter_new_results(alert, results)

            # Send notifications if there are new results
            if new_results:
                await self._send_notifications(alert, new_results)

            # Update alert schedule
            await self._update_alert_schedule(alert)

            logger.info(f"Alert '{alert.name}' completed. Found {len(new_results)} new results")

        except Exception as e:
            logger.error(f"Error executing alert {alert.alert_id}: {e}")

        finally:
            # Remove from active jobs
            self._active_jobs.pop(alert.alert_id, None)

    async def _execute_saved_search(self, alert: AlertJob) -> List[Dict[str, Any]]:
        """Execute the saved search query"""
        try:
            # Use saved search manager to execute
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.saved_search_manager.execute_saved_search,
                alert.alert_id,
                alert.user_id,
                100,  # Limit results
            )

            return result.get("results", [])

        except Exception as e:
            logger.error(f"Error executing saved search for alert {alert.alert_id}: {e}")
            return []

    async def _filter_new_results(self, alert: AlertJob, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results to only include new ones since last check"""
        try:
            if not results:
                return []

            # Get last checked timestamp
            query = "SELECT last_checked_at FROM saved_searches WHERE id = %s"

            row = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.saved_search_manager.db.fetch_one, query, (alert.alert_id,)
            )

            if not row or not row["last_checked_at"]:
                # First time checking, return all results
                return results

            last_checked = row["last_checked_at"]

            # Filter results newer than last check
            new_results = []
            for result in results:
                # Check scraped_at timestamp
                scraped_at = result.get("scraped_at")
                if scraped_at and scraped_at > last_checked:
                    new_results.append(result)

            return new_results

        except Exception as e:
            logger.error(f"Error filtering new results: {e}")
            return results  # Return all results on error

    async def _send_notifications(self, alert: AlertJob, new_results: List[Dict[str, Any]]):
        """Send notifications for new results"""
        try:
            if not self.notification_service:
                logger.warning("No notification service configured")
                return

            # Prepare notification data
            notification_data = {
                "alert_id": alert.alert_id,
                "alert_name": alert.name,
                "query": alert.query,
                "result_count": len(new_results),
                "results": new_results[:10],  # Limit to first 10
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Send to each channel
            for channel in alert.notification_channels:
                try:
                    settings = alert.notification_settings.get(channel, {})

                    # Get recipient from settings or use user ID
                    recipient = settings.get("recipient", alert.user_id)

                    # Send notification asynchronously
                    success = await self.notification_service.send_notification(
                        channel=channel,
                        recipient=recipient,
                        notification_type="saved_search_alert",
                        data=notification_data,
                        template_id=settings.get("template_id"),
                        priority=settings.get("priority", "medium"),
                    )

                    logger.info(f"Sent notification for alert {alert.alert_id} via {channel}")

                except Exception as e:
                    logger.error(f"Failed to send {channel} notification: {e}")

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")

    async def _update_alert_schedule(self, alert: AlertJob):
        """Update alert schedule after execution"""
        try:
            now = datetime.now(timezone.utc)
            next_check = now + timedelta(seconds=alert.frequency)

            query = """
            UPDATE saved_searches 
            SET last_checked_at = %s, next_check_at = %s
            WHERE id = %s
            """

            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.saved_search_manager.db.execute_query, query, (now, next_check, alert.alert_id)
            )

        except Exception as e:
            logger.error(f"Error updating alert schedule: {e}")

    async def execute_alert_immediately(self, alert_id: str) -> Dict[str, Any]:
        """
        Execute a specific alert immediately (for testing)

        Args:
            alert_id: Alert ID to execute

        Returns:
            Execution result
        """
        try:
            # Get alert details
            query = """
            SELECT 
                id, organization_id, user_id, name, query,
                notification_channels, notification_settings, alert_frequency
            FROM saved_searches 
            WHERE id = %s AND is_alert = true
            """

            row = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.saved_search_manager.db.fetch_one, query, (alert_id,)
            )

            if not row:
                raise ValueError(f"Alert {alert_id} not found")

            # Create alert job
            query_data = row["query"]
            if isinstance(query_data, str):
                query_data = json.loads(query_data)

            alert = AlertJob(
                alert_id=row["id"],
                organization_id=row["organization_id"],
                user_id=row["user_id"],
                name=row["name"],
                query=query_data,
                notification_channels=row.get("notification_channels", []),
                notification_settings=row.get("notification_settings", {}),
                next_check_at=datetime.now(timezone.utc),
                frequency=row["alert_frequency"],
            )

            # Execute alert
            await self._execute_alert(alert)

            return {
                "status": "success",
                "alert_id": alert_id,
                "alert_name": alert.name,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error executing alert immediately: {e}")
            return {"status": "error", "alert_id": alert_id, "error": str(e)}


class CeleryAlertScheduler:
    """Celery-based alert scheduler for production use"""

    def __init__(self, saved_search_manager, notification_service=None):
        """
        Initialize Celery alert scheduler

        Args:
            saved_search_manager: SavedSearchManager instance
            notification_service: Notification service instance
        """
        self.saved_search_manager = saved_search_manager
        self.notification_service = notification_service

    def schedule_periodic_check(self) -> None:
        """Schedule periodic alert checking with Celery"""
        # This would be implemented with Celery beat
        # Example configuration:
        # from celery.schedules import crontab
        #
        # CELERYBEAT_SCHEDULE = {
        #     'check-alerts': {
        #         'task': 'services.alert_scheduler.check_alerts_task',
        #         'schedule': crontab(minute='*/5'),  # Every 5 minutes
        #     },
        # }
        pass

    def check_alerts_task(self) -> None:
        """Celery task to check alerts"""
        try:
            # Get and process due alerts
            processed = self.saved_search_manager.check_alerts()

            logger.info(f"Processed {len(processed)} alerts")

            return {
                "status": "success",
                "alerts_processed": len(processed),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in check_alerts_task: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


# Example Celery task definition
try:
    from celery import shared_task

    @shared_task
    def check_alerts_task() -> Any:
        """Celery task to check all due alerts"""
        from ..db_manager_postgres import PostgreSQLManager
        from .saved_search_manager import SavedSearchManager

        db = PostgreSQLManager()
        saved_search_manager = SavedSearchManager(db)
        scheduler = CeleryAlertScheduler(saved_search_manager)

        return scheduler.check_alerts_task()

except ImportError:
    # Celery not available
    pass
