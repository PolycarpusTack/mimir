"""
Advanced Performance Monitoring for Mimir Enterprise

Real-time monitoring, alerting, and performance dashboards.
"""

import asyncio
import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp.isoformat(), "value": self.value, "labels": self.labels}


@dataclass
class Alert:
    """Performance alert"""

    id: str
    metric_name: str
    severity: str  # critical, warning, info
    message: str
    threshold: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "severity": self.severity,
            "message": self.message,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "notification_sent": self.notification_sent,
        }


class MetricCollector:
    """Collects and stores performance metrics"""

    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.retention_hours = retention_hours
        self.collectors: Dict[str, Callable] = {}
        self.collection_interval = 10  # seconds
        self.running = False
        self.collection_thread = None

    def register_collector(self, metric_name: str, collector_func: Callable[[], float]):
        """Register a metric collector function"""
        self.collectors[metric_name] = collector_func

    def add_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Add a metric data point"""
        metric_point = MetricPoint(timestamp=datetime.now(), value=value, labels=labels or {})

        self.metrics[metric_name].append(metric_point)

        # Clean old metrics
        self._cleanup_old_metrics(metric_name)

    def get_metric_history(self, metric_name: str, hours: int = 1, aggregation: str = "raw") -> List[Dict[str, Any]]:
        """Get metric history with optional aggregation"""
        if metric_name not in self.metrics:
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_points = [point for point in self.metrics[metric_name] if point.timestamp > cutoff]

        if aggregation == "raw":
            return [point.to_dict() for point in recent_points]

        elif aggregation == "minute":
            return self._aggregate_by_time(recent_points, minutes=1)

        elif aggregation == "hour":
            return self._aggregate_by_time(recent_points, hours=1)

        else:
            return [point.to_dict() for point in recent_points]

    def get_current_value(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        return self.metrics[metric_name][-1].value

    def get_statistics(self, metric_name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary for a metric"""
        if metric_name not in self.metrics:
            return {}

        cutoff = datetime.now() - timedelta(hours=hours)
        values = [point.value for point in self.metrics[metric_name] if point.timestamp > cutoff]

        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def start_collection(self):
        """Start automatic metric collection"""
        if self.running:
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Started metric collection")

    def stop_collection(self):
        """Stop automatic metric collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metric collection")

    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                for metric_name, collector_func in self.collectors.items():
                    try:
                        value = collector_func()
                        self.add_metric(metric_name, value)
                    except Exception as e:
                        logger.error(f"Error collecting metric {metric_name}: {e}")

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(1)

    def _cleanup_old_metrics(self, metric_name: str):
        """Remove old metric points beyond retention"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)

        while self.metrics[metric_name] and self.metrics[metric_name][0].timestamp < cutoff:
            self.metrics[metric_name].popleft()

    def _aggregate_by_time(self, points: List[MetricPoint], minutes: int = 0, hours: int = 0) -> List[Dict[str, Any]]:
        """Aggregate metrics by time window"""
        if not points:
            return []

        window_seconds = minutes * 60 + hours * 3600
        grouped = defaultdict(list)

        for point in points:
            window_start = int(point.timestamp.timestamp() // window_seconds) * window_seconds
            grouped[window_start].append(point.value)

        aggregated = []
        for window_start, values in grouped.items():
            aggregated.append(
                {
                    "timestamp": datetime.fromtimestamp(window_start).isoformat(),
                    "value": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
            )

        return sorted(aggregated, key=lambda x: x["timestamp"])

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class AlertManager:
    """Manages performance alerts and notifications"""

    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable] = []
        self.check_interval = 30  # seconds
        self.running = False
        self.alert_thread = None

    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "greater",  # greater, less, equal
        severity: str = "warning",
        message_template: str = None,
    ):
        """Add an alert rule"""
        rule_id = f"{metric_name}_{comparison}_{threshold}"

        self.alert_rules[rule_id] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "message_template": message_template or f"{metric_name} is {comparison} than {threshold}",
        }

        logger.info(f"Added alert rule: {rule_id}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)

    def start_monitoring(self):
        """Start alert monitoring"""
        if self.running:
            return

        self.running = True
        self.alert_thread = threading.Thread(target=self._monitoring_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        logger.info("Started alert monitoring")

    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        logger.info("Stopped alert monitoring")

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return [alert for alert in self.alerts.values() if alert.resolved_at is None]

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts.values() if alert.triggered_at > cutoff]

    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert"""
        if alert_id in self.alerts and self.alerts[alert_id].resolved_at is None:
            self.alerts[alert_id].resolved_at = datetime.now()
            logger.info(f"Resolved alert: {alert_id}")

    def _monitoring_loop(self):
        """Main alert monitoring loop"""
        while self.running:
            try:
                self._check_alert_rules()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                time.sleep(5)

    def _check_alert_rules(self):
        """Check all alert rules"""
        for rule_id, rule in self.alert_rules.items():
            try:
                current_value = self.metric_collector.get_current_value(rule["metric_name"])
                if current_value is None:
                    continue

                threshold = rule["threshold"]
                comparison = rule["comparison"]

                # Check condition
                triggered = False
                if comparison == "greater" and current_value > threshold:
                    triggered = True
                elif comparison == "less" and current_value < threshold:
                    triggered = True
                elif comparison == "equal" and abs(current_value - threshold) < 0.001:
                    triggered = True

                # Handle alert state
                existing_alert = self.alerts.get(rule_id)

                if triggered and (not existing_alert or existing_alert.resolved_at is not None):
                    # Create new alert
                    alert = Alert(
                        id=rule_id,
                        metric_name=rule["metric_name"],
                        severity=rule["severity"],
                        message=rule["message_template"].format(
                            metric=rule["metric_name"], value=current_value, threshold=threshold
                        ),
                        threshold=threshold,
                        current_value=current_value,
                        triggered_at=datetime.now(),
                    )

                    self.alerts[rule_id] = alert
                    self._send_notifications(alert)
                    logger.warning(f"Alert triggered: {alert.message}")

                elif not triggered and existing_alert and existing_alert.resolved_at is None:
                    # Resolve alert
                    existing_alert.resolved_at = datetime.now()
                    logger.info(f"Alert resolved: {existing_alert.message}")

            except Exception as e:
                logger.error(f"Error checking alert rule {rule_id}: {e}")

    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

        alert.notification_sent = True


class SystemMetrics:
    """System-level metric collectors"""

    @staticmethod
    def cpu_percent() -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)

    @staticmethod
    def memory_percent() -> float:
        """Get memory usage percentage"""
        return psutil.virtual_memory().percent

    @staticmethod
    def memory_used_mb() -> float:
        """Get memory used in MB"""
        return psutil.virtual_memory().used / 1024 / 1024

    @staticmethod
    def disk_percent() -> float:
        """Get disk usage percentage"""
        return psutil.disk_usage("/").percent

    @staticmethod
    def disk_io_read_mb() -> float:
        """Get disk read MB/s"""
        disk_io = psutil.disk_io_counters()
        return disk_io.read_bytes / 1024 / 1024 if disk_io else 0.0

    @staticmethod
    def disk_io_write_mb() -> float:
        """Get disk write MB/s"""
        disk_io = psutil.disk_io_counters()
        return disk_io.write_bytes / 1024 / 1024 if disk_io else 0.0

    @staticmethod
    def network_bytes_sent() -> float:
        """Get network bytes sent"""
        net_io = psutil.net_io_counters()
        return net_io.bytes_sent if net_io else 0.0

    @staticmethod
    def network_bytes_recv() -> float:
        """Get network bytes received"""
        net_io = psutil.net_io_counters()
        return net_io.bytes_recv if net_io else 0.0


class ApplicationMetrics:
    """Application-specific metric collectors"""

    def __init__(self, app_state=None):
        self.app_state = app_state
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=100)
        self.active_connections = 0

    def record_request(self, response_time: float, is_error: bool = False):
        """Record a request metric"""
        self.request_count += 1
        self.response_times.append(response_time)
        if is_error:
            self.error_count += 1

    def requests_per_second(self) -> float:
        """Calculate requests per second (estimated)"""
        # This would be calculated based on recent request timestamps
        return len(self.response_times) / 60.0  # Rough estimate

    def avg_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    def error_rate(self) -> float:
        """Get error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def get_active_connections(self) -> float:
        """Get active database connections"""
        # This would integrate with actual connection pool
        return float(self.active_connections)


class PerformanceMonitor:
    """Main performance monitoring coordinator"""

    def __init__(self, app_state=None):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager(self.metric_collector)
        self.app_metrics = ApplicationMetrics(app_state)

        # Register system metrics
        self._register_system_metrics()

        # Register application metrics
        self._register_application_metrics()

        # Setup default alerts
        self._setup_default_alerts()

    def _register_system_metrics(self):
        """Register system metric collectors"""
        self.metric_collector.register_collector("cpu_percent", SystemMetrics.cpu_percent)
        self.metric_collector.register_collector("memory_percent", SystemMetrics.memory_percent)
        self.metric_collector.register_collector("memory_used_mb", SystemMetrics.memory_used_mb)
        self.metric_collector.register_collector("disk_percent", SystemMetrics.disk_percent)

    def _register_application_metrics(self):
        """Register application metric collectors"""
        self.metric_collector.register_collector("requests_per_second", self.app_metrics.requests_per_second)
        self.metric_collector.register_collector("avg_response_time", self.app_metrics.avg_response_time)
        self.metric_collector.register_collector("error_rate", self.app_metrics.error_rate)
        self.metric_collector.register_collector("active_connections", self.app_metrics.get_active_connections)

    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # System alerts
        self.alert_manager.add_alert_rule("cpu_percent", 80.0, "greater", "warning", "High CPU usage: {value:.1f}%")
        self.alert_manager.add_alert_rule(
            "memory_percent", 85.0, "greater", "warning", "High memory usage: {value:.1f}%"
        )
        self.alert_manager.add_alert_rule(
            "disk_percent", 90.0, "greater", "critical", "Disk space critical: {value:.1f}%"
        )

        # Application alerts
        self.alert_manager.add_alert_rule(
            "avg_response_time", 2.0, "greater", "warning", "Slow response time: {value:.2f}s"
        )
        self.alert_manager.add_alert_rule("error_rate", 0.05, "greater", "critical", "High error rate: {value:.2%}")

    def start(self):
        """Start monitoring"""
        self.metric_collector.start_collection()
        self.alert_manager.start_monitoring()
        logger.info("Performance monitoring started")

    def stop(self):
        """Stop monitoring"""
        self.metric_collector.stop_collection()
        self.alert_manager.stop_monitoring()
        logger.info("Performance monitoring stopped")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "system": {
                    "cpu": self.metric_collector.get_statistics("cpu_percent", 1),
                    "memory": self.metric_collector.get_statistics("memory_percent", 1),
                    "disk": self.metric_collector.get_statistics("disk_percent", 1),
                },
                "application": {
                    "requests_per_second": self.metric_collector.get_current_value("requests_per_second"),
                    "avg_response_time": self.metric_collector.get_current_value("avg_response_time"),
                    "error_rate": self.metric_collector.get_current_value("error_rate"),
                    "active_connections": self.metric_collector.get_current_value("active_connections"),
                },
            },
            "alerts": {
                "active": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
                "recent": [alert.to_dict() for alert in self.alert_manager.get_alert_history(24)],
            },
        }

    def record_request(self, response_time: float, status_code: int = 200):
        """Record an API request"""
        is_error = status_code >= 400
        self.app_metrics.record_request(response_time, is_error)

    def add_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add a custom metric"""
        self.metric_collector.add_metric(name, value, labels)


# Notification handlers
def console_notification_handler(alert: Alert):
    """Console notification handler"""
    print(f"ALERT [{alert.severity.upper()}]: {alert.message}")


def email_notification_handler(alert: Alert):
    """Email notification handler (placeholder)"""
    # Would integrate with email service
    logger.info(f"Would send email alert: {alert.message}")


def webhook_notification_handler(alert: Alert):
    """Webhook notification handler (placeholder)"""
    # Would send to webhook endpoint
    logger.info(f"Would send webhook alert: {alert.message}")


# Example usage
def setup_monitoring(app_state=None) -> PerformanceMonitor:
    """Setup performance monitoring with default configuration"""
    monitor = PerformanceMonitor(app_state)

    # Add notification handlers
    monitor.alert_manager.add_notification_handler(console_notification_handler)
    monitor.alert_manager.add_notification_handler(email_notification_handler)

    return monitor
