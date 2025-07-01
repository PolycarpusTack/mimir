"""Performance Optimization Module for Mimir Analytics.

This module provides memory optimization, caching strategies, and performance
tuning capabilities to reduce technical debt and improve system efficiency.
"""

import gc
import os
import sys
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import psutil

from .exceptions import PerformanceException
from .utils import format_bytes


class MemoryManager:
    """Memory management and optimization."""
    
    def __init__(self, memory_limit_mb: int = None, gc_threshold: float = 0.8):
        """Initialize memory manager."""
        self.memory_limit_mb = memory_limit_mb or self._detect_memory_limit()
        self.gc_threshold = gc_threshold
        
        # Configure garbage collection
        gc.set_threshold(700, 10, 10)
    
    def _detect_memory_limit(self) -> int:
        """Auto-detect reasonable memory limit."""
        total_memory = psutil.virtual_memory().total
        return int((total_memory * 0.8) / (1024 * 1024))
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': process.memory_percent(),
            'memory_limit_mb': self.memory_limit_mb,
            'gc_objects': len(gc.get_objects())
        }
    
    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        usage = self.check_memory_usage()
        return usage['process_memory_mb'] > self.memory_limit_mb * self.gc_threshold
    
    def force_garbage_collection(self) -> int:
        """Force garbage collection."""
        return gc.collect()
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_type = df[col].dtype
            
            if str(col_type)[:3] == 'int':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        return df


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.memory_manager = MemoryManager()
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete optimization cycle."""
        start_time = time.perf_counter()
        
        # Force garbage collection
        collected = self.memory_manager.force_garbage_collection()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'duration_ms': duration_ms,
            'objects_collected': collected
        }


# Global optimizer
_global_optimizer = None


def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def optimize_memory():
    """Decorator to optimize memory after function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            optimizer = get_optimizer()
            if optimizer.memory_manager.is_memory_pressure():
                optimizer.run_optimization_cycle()
            
            return result
        return wrapper
    return decorator


def analyze_memory_usage() -> Dict[str, Any]:
    """Analyze current memory usage."""
    optimizer = get_optimizer()
    usage = optimizer.memory_manager.check_memory_usage()
    
    recommendations = []
    
    if usage['process_memory_percent'] > 80:
        recommendations.append("High memory usage detected. Consider optimization.")
    
    if usage['gc_objects'] > 100000:
        recommendations.append("Large number of objects. Force GC recommended.")
    
    return {
        'memory_usage': usage,
        'recommendations': recommendations,
        'health': 'good' if len(recommendations) == 0 else 'needs_attention'
    }