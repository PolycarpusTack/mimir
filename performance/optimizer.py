"""
Performance Optimization Tools for Mimir Enterprise

Provides database query optimization, caching strategies, and performance monitoring.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Query performance statistics"""

    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: Optional[datetime] = None
    slow_executions: List[float] = field(default_factory=list)

    def add_execution(self, execution_time: float):
        """Add a query execution measurement"""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.now()

        # Track slow executions (> 1 second)
        if execution_time > 1.0:
            self.slow_executions.append(execution_time)
            if len(self.slow_executions) > 10:
                self.slow_executions.pop(0)


@dataclass
class PerformanceMetrics:
    """System performance metrics"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_connections: int
    queries_per_second: float
    avg_response_time: float
    error_rate: float
    cache_hit_rate: float = 0.0


class QueryOptimizer:
    """Database query optimization and monitoring"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.query_stats: Dict[str, QueryStats] = {}
        self.slow_query_threshold = 1.0  # seconds
        self.monitoring_enabled = True

        # Query optimization suggestions
        self.optimization_rules = {
            "missing_index": [
                r"WHERE.*=.*AND.*=",  # Multiple WHERE conditions
                r"ORDER BY.*LIMIT",  # ORDER BY with LIMIT
                r"GROUP BY.*HAVING",  # GROUP BY with HAVING
            ],
            "inefficient_join": [
                r"LEFT JOIN.*WHERE.*IS NOT NULL",  # Convert to INNER JOIN
                r"DISTINCT.*JOIN",  # DISTINCT with JOIN
            ],
            "full_table_scan": [
                r"SELECT \* FROM",  # SELECT *
                r"WHERE.*LIKE \'%",  # Leading wildcard
            ],
        }

    def monitor_query(self, query: str, execution_time: float):
        """Monitor query execution"""
        if not self.monitoring_enabled:
            return

        query_hash = hash(query.strip().lower())
        query_key = str(query_hash)

        if query_key not in self.query_stats:
            self.query_stats[query_key] = QueryStats(
                query_hash=query_key, query_text=query[:200] + "..." if len(query) > 200 else query
            )

        self.query_stats[query_key].add_execution(execution_time)

        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: {execution_time:.3f}s - {query[:100]}...")

    def get_slow_queries(self, limit: int = 10) -> List[QueryStats]:
        """Get slowest queries"""
        return sorted(self.query_stats.values(), key=lambda x: x.avg_time, reverse=True)[:limit]

    def get_frequent_queries(self, limit: int = 10) -> List[QueryStats]:
        """Get most frequently executed queries"""
        return sorted(self.query_stats.values(), key=lambda x: x.execution_count, reverse=True)[:limit]

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for optimization opportunities"""
        issues = []
        suggestions = []

        query_lower = query.lower()

        # Check for optimization opportunities
        for issue_type, patterns in self.optimization_rules.items():
            for pattern in patterns:
                import re

                if re.search(pattern, query_lower, re.IGNORECASE):
                    issues.append(issue_type)
                    suggestions.extend(self._get_suggestions(issue_type))

        return {
            "query": query,
            "issues": issues,
            "suggestions": suggestions,
            "complexity_score": self._calculate_complexity(query),
        }

    def _get_suggestions(self, issue_type: str) -> List[str]:
        """Get optimization suggestions for issue type"""
        suggestions_map = {
            "missing_index": [
                "Consider adding indexes on WHERE clause columns",
                "Add composite index for multiple WHERE conditions",
                "Consider partial indexes for filtered queries",
            ],
            "inefficient_join": [
                "Convert LEFT JOIN to INNER JOIN where possible",
                "Remove DISTINCT if JOIN guarantees uniqueness",
                "Consider denormalization for frequently joined tables",
            ],
            "full_table_scan": [
                "Avoid SELECT * - specify only needed columns",
                "Add indexes to avoid full table scans",
                "Use LIMIT to reduce result set size",
            ],
        }
        return suggestions_map.get(issue_type, [])

    def _calculate_complexity(self, query: str) -> int:
        """Calculate query complexity score"""
        score = 0
        query_lower = query.lower()

        # Basic complexity factors
        score += query_lower.count("join") * 2
        score += query_lower.count("union") * 3
        score += query_lower.count("subquery") * 3
        score += query_lower.count("group by") * 2
        score += query_lower.count("order by") * 1
        score += query_lower.count("having") * 2
        score += query_lower.count("distinct") * 2

        return score

    def suggest_indexes(self) -> List[Dict[str, Any]]:
        """Suggest database indexes based on query patterns"""
        suggestions = []

        # Analyze frequent queries for index opportunities
        frequent_queries = self.get_frequent_queries(20)

        for query_stat in frequent_queries:
            query = query_stat.query_text

            # Extract table and column patterns
            tables, columns = self._extract_query_patterns(query)

            for table, table_columns in columns.items():
                if len(table_columns) > 1:
                    suggestions.append(
                        {
                            "type": "composite_index",
                            "table": table,
                            "columns": list(table_columns),
                            "reason": f"Frequent query with multiple WHERE conditions (executed {query_stat.execution_count} times)",
                            "priority": "high" if query_stat.avg_time > 0.5 else "medium",
                        }
                    )

        return suggestions

    def _extract_query_patterns(self, query: str) -> Tuple[set, Dict[str, set]]:
        """Extract table and column patterns from query"""
        import re

        tables = set()
        columns = defaultdict(set)

        # Simple pattern matching (would be more sophisticated in production)
        table_pattern = r"FROM\s+(\w+)"
        column_pattern = r"WHERE\s+(\w+)\.(\w+)\s*="

        # Extract tables
        for match in re.finditer(table_pattern, query, re.IGNORECASE):
            tables.add(match.group(1))

        # Extract columns used in WHERE clauses
        for match in re.finditer(column_pattern, query, re.IGNORECASE):
            table = match.group(1)
            column = match.group(2)
            columns[table].add(column)

        return tables, dict(columns)


class CacheManager:
    """Advanced caching with multiple strategies"""

    def __init__(self):
        self.caches = {
            "memory": {},  # In-memory cache
            "query_result": {},  # Query result cache
            "api_response": {},  # API response cache
            "user_session": {},  # User session cache
        }
        self.cache_stats = {"hits": defaultdict(int), "misses": defaultdict(int), "invalidations": defaultdict(int)}
        self.ttl_data = {}  # TTL tracking

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        if cache_type not in self.caches:
            return None

        cache = self.caches[cache_type]

        # Check TTL
        if self._is_expired(cache_type, key):
            self.invalidate(cache_type, key)
            self.cache_stats["misses"][cache_type] += 1
            return None

        if key in cache:
            self.cache_stats["hits"][cache_type] += 1
            return cache[key]

        self.cache_stats["misses"][cache_type] += 1
        return None

    def set(self, cache_type: str, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL"""
        if cache_type not in self.caches:
            return

        self.caches[cache_type][key] = value
        self.ttl_data[f"{cache_type}:{key}"] = time.time() + ttl

    def invalidate(self, cache_type: str, key: str = None):
        """Invalidate cache entries"""
        if cache_type not in self.caches:
            return

        if key:
            self.caches[cache_type].pop(key, None)
            self.ttl_data.pop(f"{cache_type}:{key}", None)
            self.cache_stats["invalidations"][cache_type] += 1
        else:
            # Clear entire cache type
            self.caches[cache_type].clear()
            # Clear related TTL entries
            keys_to_remove = [k for k in self.ttl_data.keys() if k.startswith(f"{cache_type}:")]
            for k in keys_to_remove:
                del self.ttl_data[k]
            self.cache_stats["invalidations"][cache_type] += len(keys_to_remove)

    def _is_expired(self, cache_type: str, key: str) -> bool:
        """Check if cache entry is expired"""
        ttl_key = f"{cache_type}:{key}"
        if ttl_key not in self.ttl_data:
            return False
        return time.time() > self.ttl_data[ttl_key]

    def get_hit_rate(self, cache_type: str) -> float:
        """Get cache hit rate"""
        hits = self.cache_stats["hits"][cache_type]
        misses = self.cache_stats["misses"][cache_type]
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [key for key, expiry in self.ttl_data.items() if current_time > expiry]

        for ttl_key in expired_keys:
            cache_type, key = ttl_key.split(":", 1)
            self.invalidate(cache_type, key)


class PerformanceMonitor:
    """System performance monitoring"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts = []
        self.thresholds = {"cpu_percent": 80.0, "memory_percent": 85.0, "avg_response_time": 2.0, "error_rate": 0.05}

    def collect_metrics(
        self,
        active_connections: int = 0,
        queries_per_second: float = 0.0,
        avg_response_time: float = 0.0,
        error_rate: float = 0.0,
        cache_hit_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """Collect current system metrics"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            active_connections=active_connections,
            queries_per_second=queries_per_second,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
        )

        self.metrics_history.append(metrics)

        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

        # Check for alerts
        self._check_alerts(metrics)

        return metrics

    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check performance thresholds and generate alerts"""
        alerts = []

        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(
                {
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    "timestamp": metrics.timestamp,
                }
            )

        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(
                {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                    "timestamp": metrics.timestamp,
                }
            )

        if metrics.avg_response_time > self.thresholds["avg_response_time"]:
            alerts.append(
                {
                    "type": "slow_response",
                    "severity": "warning",
                    "message": f"Slow response time: {metrics.avg_response_time:.2f}s",
                    "timestamp": metrics.timestamp,
                }
            )

        if metrics.error_rate > self.thresholds["error_rate"]:
            alerts.append(
                {
                    "type": "high_error_rate",
                    "severity": "critical",
                    "message": f"High error rate: {metrics.error_rate:.2%}",
                    "timestamp": metrics.timestamp,
                }
            )

        self.alerts.extend(alerts)

        # Keep only recent alerts (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a["timestamp"] > cutoff]

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff]

        if not recent_metrics:
            return {}

        return {
            "time_period_hours": hours,
            "total_samples": len(recent_metrics),
            "cpu": {
                "avg": statistics.mean([m.cpu_percent for m in recent_metrics]),
                "max": max([m.cpu_percent for m in recent_metrics]),
                "min": min([m.cpu_percent for m in recent_metrics]),
            },
            "memory": {
                "avg": statistics.mean([m.memory_percent for m in recent_metrics]),
                "max": max([m.memory_percent for m in recent_metrics]),
                "peak_used_mb": max([m.memory_used_mb for m in recent_metrics]),
            },
            "response_time": {
                "avg": statistics.mean([m.avg_response_time for m in recent_metrics]),
                "max": max([m.avg_response_time for m in recent_metrics]),
                "p95": self._percentile([m.avg_response_time for m in recent_metrics], 95),
            },
            "error_rate": {
                "avg": statistics.mean([m.error_rate for m in recent_metrics]),
                "max": max([m.error_rate for m in recent_metrics]),
            },
            "cache_hit_rate": {"avg": statistics.mean([m.cache_hit_rate for m in recent_metrics])},
            "active_alerts": len([a for a in self.alerts if a["timestamp"] > cutoff]),
        }

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class ConnectionPoolOptimizer:
    """Database connection pool optimization"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.connection_stats = {
            "total_created": 0,
            "total_closed": 0,
            "current_active": 0,
            "max_active": 0,
            "wait_times": [],
        }

    def optimize_pool_size(self) -> Dict[str, Any]:
        """Analyze and suggest optimal pool size"""
        current_stats = self.get_current_stats()

        # Simple optimization logic
        recommendations = []

        if current_stats["utilization"] > 0.8:
            recommendations.append(
                {
                    "type": "increase_pool_size",
                    "current_size": current_stats["pool_size"],
                    "suggested_size": int(current_stats["pool_size"] * 1.5),
                    "reason": "High pool utilization detected",
                }
            )

        if current_stats["avg_wait_time"] > 0.1:
            recommendations.append(
                {
                    "type": "increase_pool_size",
                    "reason": "Long wait times for connections",
                    "avg_wait_time": current_stats["avg_wait_time"],
                }
            )

        if current_stats["utilization"] < 0.3:
            recommendations.append(
                {
                    "type": "decrease_pool_size",
                    "current_size": current_stats["pool_size"],
                    "suggested_size": max(5, int(current_stats["pool_size"] * 0.7)),
                    "reason": "Low pool utilization - reduce resource usage",
                }
            )

        return {"current_stats": current_stats, "recommendations": recommendations}

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current connection pool statistics"""
        # This would integrate with actual connection pool
        return {
            "pool_size": 20,  # Current pool size
            "active_connections": self.connection_stats["current_active"],
            "utilization": self.connection_stats["current_active"] / 20,
            "avg_wait_time": statistics.mean(self.connection_stats["wait_times"])
            if self.connection_stats["wait_times"]
            else 0.0,
            "total_created": self.connection_stats["total_created"],
            "total_closed": self.connection_stats["total_closed"],
        }


# Performance optimization coordinator
class PerformanceOptimizer:
    """Main performance optimization coordinator"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.query_optimizer = QueryOptimizer(db_manager)
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        self.connection_optimizer = ConnectionPoolOptimizer(db_manager)

        # Optimization settings
        self.auto_optimize = True
        self.optimization_interval = 300  # 5 minutes

    async def run_optimization_cycle(self):
        """Run complete optimization cycle"""
        logger.info("Starting performance optimization cycle...")

        # Collect current metrics
        metrics = self.performance_monitor.collect_metrics()

        # Clean up expired cache entries
        self.cache_manager.cleanup_expired()

        # Analyze queries and suggest optimizations
        slow_queries = self.query_optimizer.get_slow_queries(10)
        index_suggestions = self.query_optimizer.suggest_indexes()

        # Optimize connection pool
        pool_optimization = self.connection_optimizer.optimize_pool_size()

        # Generate optimization report
        report = {
            "timestamp": datetime.now(),
            "system_metrics": metrics,
            "slow_queries": [
                {"query": q.query_text, "avg_time": q.avg_time, "execution_count": q.execution_count}
                for q in slow_queries
            ],
            "index_suggestions": index_suggestions,
            "cache_hit_rates": {
                cache_type: self.cache_manager.get_hit_rate(cache_type)
                for cache_type in self.cache_manager.caches.keys()
            },
            "connection_pool": pool_optimization,
            "active_alerts": self.performance_monitor.alerts,
        }

        logger.info(
            f"Optimization cycle completed. Found {len(slow_queries)} slow queries, {len(index_suggestions)} index suggestions"
        )

        return report

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get comprehensive optimization recommendations"""
        recommendations = []

        # Query optimizations
        slow_queries = self.query_optimizer.get_slow_queries(5)
        for query in slow_queries:
            analysis = self.query_optimizer.analyze_query(query.query_text)
            if analysis["issues"]:
                recommendations.append(
                    {
                        "type": "query_optimization",
                        "priority": "high" if query.avg_time > 2.0 else "medium",
                        "title": f"Optimize slow query (avg: {query.avg_time:.2f}s)",
                        "description": f"Query executed {query.execution_count} times",
                        "issues": analysis["issues"],
                        "suggestions": analysis["suggestions"],
                    }
                )

        # Index suggestions
        index_suggestions = self.query_optimizer.suggest_indexes()
        for suggestion in index_suggestions:
            recommendations.append(
                {
                    "type": "database_index",
                    "priority": suggestion["priority"],
                    "title": f"Add {suggestion['type']} on {suggestion['table']}",
                    "description": suggestion["reason"],
                    "columns": suggestion["columns"],
                }
            )

        # Cache optimizations
        for cache_type in self.cache_manager.caches.keys():
            hit_rate = self.cache_manager.get_hit_rate(cache_type)
            if hit_rate < 0.7:  # Less than 70% hit rate
                recommendations.append(
                    {
                        "type": "cache_optimization",
                        "priority": "medium",
                        "title": f"Improve {cache_type} cache performance",
                        "description": f"Current hit rate: {hit_rate:.1%}",
                        "suggestions": [
                            "Increase cache TTL for stable data",
                            "Pre-warm cache for frequently accessed data",
                            "Review cache invalidation strategy",
                        ],
                    }
                )

        # System resource optimizations
        recent_metrics = self.performance_monitor.get_performance_summary(1)
        if recent_metrics and recent_metrics["cpu"]["avg"] > 70:
            recommendations.append(
                {
                    "type": "system_optimization",
                    "priority": "high",
                    "title": "High CPU usage detected",
                    "description": f"Average CPU: {recent_metrics['cpu']['avg']:.1f}%",
                    "suggestions": [
                        "Review and optimize CPU-intensive queries",
                        "Consider horizontal scaling",
                        "Implement query result caching",
                    ],
                }
            )

        return sorted(recommendations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)
