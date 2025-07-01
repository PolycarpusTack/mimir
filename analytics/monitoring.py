"""Enhanced Monitoring and Logging for Mimir Analytics.

This module provides comprehensive monitoring, logging, and alerting capabilities
for the analytics system with structured logging, metrics collection, and alerting.
"""

import json
import logging
import logging.handlers
import psutil
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Callable, Union

import pandas as pd
from dataclasses import dataclass, asdict

from .exceptions import AnalyticsBaseException, PerformanceException
from .utils import format_bytes


@dataclass
class MetricPoint:
    """Represents a single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertRule:
    """Defines an alerting rule."""
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    duration_minutes: int = 5
    severity: str = "warning"  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True


class StructuredLogger:
    """Enhanced structured logger with context and correlation tracking."""
    
    def __init__(self, name: str, level: int = logging.INFO,
                 log_file: Path = None, max_file_size: int = 10 * 1024 * 1024):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            max_file_size: Maximum log file size in bytes
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Context storage
        self._context = {}
        self._correlation_id = None
        
        # Setup handlers
        self._setup_handlers(log_file, max_file_size)
        
        # Custom formatter
        formatter = self._create_structured_formatter()
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)
    
    def _setup_handlers(self, log_file: Path, max_file_size: int):
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
    
    def _create_structured_formatter(self):
        """Create structured JSON formatter."""
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add context if available
                if hasattr(record, 'context'):
                    log_entry['context'] = record.context
                
                # Add correlation ID if available
                if hasattr(record, 'correlation_id'):
                    log_entry['correlation_id'] = record.correlation_id
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }
                
                return json.dumps(log_entry)
        
        return StructuredFormatter()
    
    def set_context(self, **kwargs):
        """Set logging context."""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        self._context.clear()
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self._correlation_id = correlation_id
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with context and correlation ID."""
        extra = {
            'context': {**self._context, **kwargs},
            'correlation_id': self._correlation_id
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, message, **kwargs)


class MetricsCollector:
    """Collects and stores metrics with time-series capabilities."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.retention_hours = retention_hours
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.lock = Lock()
        
        # Start cleanup thread
        self.cleanup_thread = Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Record a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for grouping
            unit: Unit of measurement
        """
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics[name].append(metric_point)
    
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record(name, value, tags, "count")
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Record a gauge metric."""
        self.record(name, value, tags, unit)
    
    def timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        self.record(name, duration_ms, tags, "ms")
    
    def get_metrics(self, name: str, since: datetime = None) -> List[MetricPoint]:
        """Get metrics for a given name since a timestamp."""
        with self.lock:
            if name not in self.metrics:
                return []
            
            if since is None:
                return list(self.metrics[name])
            
            return [
                point for point in self.metrics[name]
                if point.timestamp >= since
            ]
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            return self.metrics[name][-1].value
    
    def get_average(self, name: str, since: datetime = None) -> Optional[float]:
        """Get average value for a metric."""
        points = self.get_metrics(name, since)
        if not points:
            return None
        
        return sum(point.value for point in points) / len(points)
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self.lock:
            return list(self.metrics.keys())
    
    def _cleanup_old_metrics(self):
        """Cleanup old metrics in background."""
        while True:
            time.sleep(300)  # Run every 5 minutes
            
            cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
            
            with self.lock:
                for name, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector,
                 collection_interval: int = 30):
        """Initialize system monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            collection_interval: Collection interval in seconds
        """
        self.metrics = metrics_collector
        self.collection_interval = collection_interval
        self.running = False
        self.monitor_thread = None
    
    def start(self):
        """Start system monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop system monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                # Log error but continue monitoring
                logging.getLogger(__name__).error(f"System monitoring error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.gauge("system.cpu_percent", cpu_percent, unit="%")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.gauge("system.memory_percent", memory.percent, unit="%")
        self.metrics.gauge("system.memory_used", memory.used, unit="bytes")
        self.metrics.gauge("system.memory_available", memory.available, unit="bytes")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.gauge("system.disk_percent", (disk.used / disk.total) * 100, unit="%")
        self.metrics.gauge("system.disk_used", disk.used, unit="bytes")
        self.metrics.gauge("system.disk_free", disk.free, unit="bytes")
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.metrics.gauge("system.network_bytes_sent", network.bytes_sent, unit="bytes")
            self.metrics.gauge("system.network_bytes_recv", network.bytes_recv, unit="bytes")
        except:
            pass  # Network stats might not be available
        
        # Process-specific metrics
        try:
            process = psutil.Process()
            self.metrics.gauge("process.memory_percent", process.memory_percent(), unit="%")
            self.metrics.gauge("process.cpu_percent", process.cpu_percent(), unit="%")
            
            memory_info = process.memory_info()
            self.metrics.gauge("process.memory_rss", memory_info.rss, unit="bytes")
            self.metrics.gauge("process.memory_vms", memory_info.vms, unit="bytes")
        except:
            pass  # Process stats might not be available


class AlertManager:
    """Manages alerting based on metric thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector,
                 logger: StructuredLogger = None):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
            logger: Logger for alert notifications
        """
        self.metrics = metrics_collector
        self.logger = logger or StructuredLogger(__name__)
        self.rules = {}
        self.alert_states = {}
        self.lock = Lock()
        
        # Alert checking thread
        self.running = False
        self.check_thread = None
    
    def add_rule(self, rule: AlertRule):
        """Add an alerting rule."""
        with self.lock:
            self.rules[rule.name] = rule
            self.alert_states[rule.name] = {
                'active': False,
                'triggered_at': None,
                'last_check': None
            }
    
    def remove_rule(self, rule_name: str):
        """Remove an alerting rule."""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                del self.alert_states[rule_name]
    
    def start(self, check_interval: int = 60):
        """Start alert checking.
        
        Args:
            check_interval: Check interval in seconds
        """
        if self.running:
            return
        
        self.running = True
        self.check_interval = check_interval
        self.check_thread = Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
    
    def stop(self):
        """Stop alert checking."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
    
    def _check_loop(self):
        """Main alert checking loop."""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error("Alert checking error", error=str(e))
                time.sleep(self.check_interval)
    
    def _check_alerts(self):
        """Check all alert rules."""
        now = datetime.utcnow()
        
        with self.lock:
            rules_to_check = {
                name: rule for name, rule in self.rules.items()
                if rule.enabled
            }
        
        for rule_name, rule in rules_to_check.items():
            try:
                self._check_rule(rule_name, rule, now)
            except Exception as e:
                self.logger.error(f"Error checking rule {rule_name}", error=str(e))
    
    def _check_rule(self, rule_name: str, rule: AlertRule, now: datetime):
        """Check a specific alert rule."""
        # Get recent metric values
        since = now - timedelta(minutes=rule.duration_minutes)
        points = self.metrics.get_metrics(rule.metric_name, since)
        
        if not points:
            return
        
        # Check if condition is met
        latest_value = points[-1].value
        condition_met = self._evaluate_condition(
            latest_value, rule.condition, rule.threshold
        )
        
        with self.lock:
            state = self.alert_states[rule_name]
            state['last_check'] = now
            
            if condition_met and not state['active']:
                # Check if condition has been met for duration
                duration_met = all(
                    self._evaluate_condition(point.value, rule.condition, rule.threshold)
                    for point in points
                )
                
                if duration_met:
                    # Trigger alert
                    state['active'] = True
                    state['triggered_at'] = now
                    self._trigger_alert(rule, latest_value)
            
            elif not condition_met and state['active']:
                # Resolve alert
                state['active'] = False
                state['triggered_at'] = None
                self._resolve_alert(rule, latest_value)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return abs(value - threshold) < 0.001
        elif condition == 'ne':
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert."""
        self.logger.warning(
            f"ALERT TRIGGERED: {rule.name}",
            alert_name=rule.name,
            metric_name=rule.metric_name,
            current_value=value,
            threshold=rule.threshold,
            condition=rule.condition,
            severity=rule.severity
        )
    
    def _resolve_alert(self, rule: AlertRule, value: float):
        """Resolve an alert."""
        self.logger.info(
            f"ALERT RESOLVED: {rule.name}",
            alert_name=rule.name,
            metric_name=rule.metric_name,
            current_value=value,
            threshold=rule.threshold
        )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        with self.lock:
            active_alerts = []
            for rule_name, state in self.alert_states.items():
                if state['active']:
                    rule = self.rules[rule_name]
                    active_alerts.append({
                        'name': rule_name,
                        'severity': rule.severity,
                        'triggered_at': state['triggered_at'].isoformat(),
                        'metric_name': rule.metric_name,
                        'condition': f"{rule.condition} {rule.threshold}"
                    })
            return active_alerts


class PerformanceProfiler:
    """Profile performance of functions and operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance profiler."""
        self.metrics = metrics_collector
        self.active_profiles = {}
        self.lock = Lock()
    
    @contextmanager
    def profile(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager for profiling operations."""
        start_time = time.perf_counter()
        operation_id = f"{operation_name}_{id(start_time)}"
        
        try:
            with self.lock:
                self.active_profiles[operation_id] = {
                    'name': operation_name,
                    'start_time': start_time,
                    'tags': tags or {}
                }
            
            yield
            
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Record timing metric
            self.metrics.timing(f"operation.{operation_name}", duration_ms, tags)
            
            with self.lock:
                if operation_id in self.active_profiles:
                    del self.active_profiles[operation_id]
    
    def profile_function(self, func_name: str = None):
        """Decorator for profiling functions."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_performance_summary(self, since: datetime = None) -> Dict[str, Any]:
        """Get performance summary."""
        if since is None:
            since = datetime.utcnow() - timedelta(hours=1)
        
        summary = {}
        
        # Get all timing metrics
        for metric_name in self.metrics.get_metric_names():
            if metric_name.startswith("operation."):
                operation_name = metric_name.replace("operation.", "")
                points = self.metrics.get_metrics(metric_name, since)
                
                if points:
                    durations = [point.value for point in points]
                    summary[operation_name] = {
                        'count': len(durations),
                        'avg_duration_ms': sum(durations) / len(durations),
                        'min_duration_ms': min(durations),
                        'max_duration_ms': max(durations),
                        'total_time_ms': sum(durations)
                    }
        
        return summary


class MonitoringDashboard:
    """Simple text-based monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector,
                 alert_manager: AlertManager):
        """Initialize monitoring dashboard."""
        self.metrics = metrics_collector
        self.alerts = alert_manager
    
    def generate_status_report(self) -> str:
        """Generate a status report."""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append("MIMIR ANALYTICS MONITORING DASHBOARD")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("=" * 60)
        
        # System metrics
        report.append("\nSYSTEM METRICS:")
        report.append("-" * 20)
        
        system_metrics = [
            "system.cpu_percent",
            "system.memory_percent",
            "system.disk_percent",
            "process.memory_percent",
            "process.cpu_percent"
        ]
        
        for metric in system_metrics:
            value = self.metrics.get_latest_value(metric)
            if value is not None:
                report.append(f"{metric:30} {value:8.1f}")
        
        # Active alerts
        active_alerts = self.alerts.get_active_alerts()
        report.append(f"\nACTIVE ALERTS ({len(active_alerts)}):")
        report.append("-" * 20)
        
        if active_alerts:
            for alert in active_alerts:
                report.append(f"[{alert['severity'].upper()}] {alert['name']}")
                report.append(f"  Triggered: {alert['triggered_at']}")
                report.append(f"  Condition: {alert['metric_name']} {alert['condition']}")
        else:
            report.append("No active alerts")
        
        # Recent metrics
        report.append("\nRECENT ACTIVITY:")
        report.append("-" * 20)
        
        for metric_name in self.metrics.get_metric_names():
            if not metric_name.startswith("system.") and not metric_name.startswith("process."):
                recent_points = self.metrics.get_metrics(
                    metric_name,
                    datetime.utcnow() - timedelta(minutes=5)
                )
                if recent_points:
                    report.append(f"{metric_name:30} {len(recent_points):4} points")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def get_metrics_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for time range."""
        since = datetime.utcnow() - timedelta(minutes=time_range_minutes)
        summary = {}
        
        for metric_name in self.metrics.get_metric_names():
            points = self.metrics.get_metrics(metric_name, since)
            if points:
                values = [point.value for point in points]
                summary[metric_name] = {
                    'count': len(values),
                    'latest': values[-1],
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return summary


# Global monitoring components
_global_logger = None
_global_metrics = None
_global_system_monitor = None
_global_alert_manager = None
_global_profiler = None


def get_logger(name: str = None) -> StructuredLogger:
    """Get global structured logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name or __name__)
    return _global_logger


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(get_metrics())
    return _global_profiler


def setup_monitoring(log_file: Path = None, enable_system_monitoring: bool = True,
                    enable_alerting: bool = True) -> Dict[str, Any]:
    """Setup global monitoring components.
    
    Args:
        log_file: Optional log file path
        enable_system_monitoring: Enable system resource monitoring
        enable_alerting: Enable alerting
        
    Returns:
        Dictionary of monitoring components
    """
    global _global_logger, _global_metrics, _global_system_monitor, _global_alert_manager
    
    # Initialize components
    _global_logger = StructuredLogger(__name__, log_file=log_file)
    _global_metrics = MetricsCollector()
    
    components = {
        'logger': _global_logger,
        'metrics': _global_metrics
    }
    
    if enable_system_monitoring:
        _global_system_monitor = SystemMonitor(_global_metrics)
        _global_system_monitor.start()
        components['system_monitor'] = _global_system_monitor
    
    if enable_alerting:
        _global_alert_manager = AlertManager(_global_metrics, _global_logger)
        
        # Setup default alert rules
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu_percent",
                condition="gt",
                threshold=90.0,
                duration_minutes=5,
                severity="warning"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory_percent",
                condition="gt",
                threshold=90.0,
                duration_minutes=5,
                severity="warning"
            ),
            AlertRule(
                name="low_disk_space",
                metric_name="system.disk_percent",
                condition="gt",
                threshold=90.0,
                duration_minutes=10,
                severity="error"
            )
        ]
        
        for rule in default_rules:
            _global_alert_manager.add_rule(rule)
        
        _global_alert_manager.start()
        components['alert_manager'] = _global_alert_manager
    
    _global_logger.info("Monitoring system initialized", components=list(components.keys()))
    
    return components


def shutdown_monitoring():
    """Shutdown global monitoring components."""
    global _global_system_monitor, _global_alert_manager
    
    if _global_system_monitor:
        _global_system_monitor.stop()
    
    if _global_alert_manager:
        _global_alert_manager.stop()
    
    if _global_logger:
        _global_logger.info("Monitoring system shutdown")


# Decorators for easy monitoring
def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_execution(logger_name: str = None, level: str = "info"):
    """Decorator to log function execution."""
    def decorator(func):
        logger = get_logger(logger_name)
        log_method = getattr(logger, level)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            logger.set_context(function=func_name)
            
            log_method(f"Starting {func_name}")
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.perf_counter() - start_time) * 1000
                log_method(f"Completed {func_name}", duration_ms=duration)
                return result
            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000
                logger.error(f"Failed {func_name}", duration_ms=duration, error=str(e))
                raise
            finally:
                logger.clear_context()
        
        return wrapper
    return decorator


def count_calls(metric_name: str = None):
    """Decorator to count function calls."""
    def decorator(func):
        name = metric_name or f"calls.{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            metrics.increment(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator