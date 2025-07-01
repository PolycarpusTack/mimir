"""
Bulk operations utilities for Mimir Enterprise API

Provides efficient bulk processing for create, update, and delete operations.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BulkOperationType(Enum):
    """Types of bulk operations"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


class BulkOperationStatus(Enum):
    """Status of bulk operation items"""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BulkOperationItem:
    """Individual item in a bulk operation"""

    index: int
    data: Any
    status: BulkOperationStatus = BulkOperationStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class BulkResult:
    """Result of a bulk operation"""

    operation_type: BulkOperationType
    total_items: int
    successful: int
    failed: int
    skipped: int
    total_time_ms: float
    items: List[BulkOperationItem] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.successful / self.total_items) * 100


class BulkOperationHandler:
    """Handler for bulk operations with batching and error handling"""

    def __init__(self, batch_size: int = 100, max_workers: int = 4, max_items_per_request: int = 1000):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_items_per_request = max_items_per_request

    def validate_bulk_request(self, items: List[Any], operation_type: BulkOperationType) -> None:
        """
        Validate bulk operation request

        Args:
            items: List of items to process
            operation_type: Type of operation

        Raises:
            ValueError: If validation fails
        """
        if not items:
            raise ValueError("No items provided for bulk operation")

        if len(items) > self.max_items_per_request:
            raise ValueError(f"Too many items. Maximum {self.max_items_per_request} allowed")

        # Operation-specific validation
        if operation_type == BulkOperationType.CREATE:
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(f"Item {i}: Expected dictionary for create operation")

        elif operation_type in [BulkOperationType.UPDATE, BulkOperationType.DELETE]:
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    if "id" not in item:
                        raise ValueError(f"Item {i}: ID required for {operation_type.value} operation")
                elif not isinstance(item, (str, int)):
                    raise ValueError(f"Item {i}: Expected ID or object with ID for {operation_type.value}")

    def execute_bulk_operation(
        self, items: List[Any], operation_func: Callable, operation_type: BulkOperationType, **operation_kwargs
    ) -> BulkResult:
        """
        Execute bulk operation with batching and error handling

        Args:
            items: List of items to process
            operation_func: Function to execute for each item/batch
            operation_type: Type of operation
            **operation_kwargs: Additional arguments for operation function

        Returns:
            BulkResult with operation summary and details
        """
        start_time = time.time()

        # Validate request
        self.validate_bulk_request(items, operation_type)

        # Initialize result
        bulk_result = BulkResult(
            operation_type=operation_type, total_items=len(items), successful=0, failed=0, skipped=0, total_time_ms=0.0
        )

        # Create operation items
        operation_items = [BulkOperationItem(index=i, data=item) for i, item in enumerate(items)]

        # Process in batches
        batches = self._create_batches(operation_items, self.batch_size)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch jobs
            future_to_batch = {
                executor.submit(self._execute_batch, batch, operation_func, **operation_kwargs): batch
                for batch in batches
            }

            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    operation_items.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch execution failed: {e}")
                    # Mark all items in this batch as failed
                    for item in batch:
                        item.status = BulkOperationStatus.FAILED
                        item.error = str(e)

        # Calculate final results
        bulk_result.items = operation_items
        bulk_result.successful = sum(1 for item in operation_items if item.status == BulkOperationStatus.SUCCESS)
        bulk_result.failed = sum(1 for item in operation_items if item.status == BulkOperationStatus.FAILED)
        bulk_result.skipped = sum(1 for item in operation_items if item.status == BulkOperationStatus.SKIPPED)
        bulk_result.total_time_ms = (time.time() - start_time) * 1000

        # Generate summary
        bulk_result.summary = self._generate_summary(bulk_result)

        return bulk_result

    def _create_batches(self, items: List[T], batch_size: int) -> List[List[T]]:
        """Split items into batches"""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i : i + batch_size])
        return batches

    def _execute_batch(
        self, batch: List[BulkOperationItem], operation_func: Callable, **operation_kwargs
    ) -> List[BulkOperationItem]:
        """Execute a single batch of operations"""
        for item in batch:
            item_start_time = time.time()

            try:
                # Execute operation for this item
                result = operation_func(item.data, **operation_kwargs)

                # Mark as successful
                item.status = BulkOperationStatus.SUCCESS
                item.result = result

            except Exception as e:
                # Mark as failed
                item.status = BulkOperationStatus.FAILED
                item.error = str(e)
                logger.warning(f"Item {item.index} failed: {e}")

            finally:
                item.execution_time_ms = (time.time() - item_start_time) * 1000

        return batch

    def _generate_summary(self, bulk_result: BulkResult) -> Dict[str, Any]:
        """Generate operation summary"""
        avg_execution_time = 0.0
        if bulk_result.items:
            execution_times = [
                item.execution_time_ms for item in bulk_result.items if item.execution_time_ms is not None
            ]
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)

        return {
            "success_rate": bulk_result.success_rate,
            "average_execution_time_ms": round(avg_execution_time, 2),
            "throughput_items_per_second": round(
                bulk_result.total_items / (bulk_result.total_time_ms / 1000) if bulk_result.total_time_ms > 0 else 0, 2
            ),
            "errors": [
                {"index": item.index, "error": item.error}
                for item in bulk_result.items
                if item.status == BulkOperationStatus.FAILED
            ][
                :10
            ],  # Limit to first 10 errors
        }


class AsyncBulkOperationHandler:
    """Async version of bulk operation handler"""

    def __init__(self, batch_size: int = 100, concurrency_limit: int = 10):
        self.batch_size = batch_size
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def execute_bulk_operation_async(
        self, items: List[Any], operation_func: Callable, operation_type: BulkOperationType, **operation_kwargs
    ) -> BulkResult:
        """
        Execute bulk operation asynchronously

        Args:
            items: List of items to process
            operation_func: Async function to execute for each item
            operation_type: Type of operation
            **operation_kwargs: Additional arguments for operation function

        Returns:
            BulkResult with operation summary and details
        """
        start_time = time.time()

        # Initialize result
        bulk_result = BulkResult(
            operation_type=operation_type, total_items=len(items), successful=0, failed=0, skipped=0, total_time_ms=0.0
        )

        # Create operation items
        operation_items = [BulkOperationItem(index=i, data=item) for i, item in enumerate(items)]

        # Create tasks for each item
        tasks = [self._execute_item_async(item, operation_func, **operation_kwargs) for item in operation_items]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate final results
        bulk_result.items = operation_items
        bulk_result.successful = sum(1 for item in operation_items if item.status == BulkOperationStatus.SUCCESS)
        bulk_result.failed = sum(1 for item in operation_items if item.status == BulkOperationStatus.FAILED)
        bulk_result.skipped = sum(1 for item in operation_items if item.status == BulkOperationStatus.SKIPPED)
        bulk_result.total_time_ms = (time.time() - start_time) * 1000

        return bulk_result

    async def _execute_item_async(self, item: BulkOperationItem, operation_func: Callable, **operation_kwargs) -> None:
        """Execute operation for a single item asynchronously"""
        async with self.semaphore:
            item_start_time = time.time()

            try:
                # Execute async operation
                result = await operation_func(item.data, **operation_kwargs)

                # Mark as successful
                item.status = BulkOperationStatus.SUCCESS
                item.result = result

            except Exception as e:
                # Mark as failed
                item.status = BulkOperationStatus.FAILED
                item.error = str(e)
                logger.warning(f"Async item {item.index} failed: {e}")

            finally:
                item.execution_time_ms = (time.time() - item_start_time) * 1000


class BulkOperationValidator:
    """Validator for bulk operations with custom rules"""

    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}

    def add_validation_rule(self, operation_type: str, rule_func: Callable) -> None:
        """Add validation rule for specific operation type"""
        if operation_type not in self.validation_rules:
            self.validation_rules[operation_type] = []
        self.validation_rules[operation_type].append(rule_func)

    def validate_items(self, items: List[Any], operation_type: BulkOperationType) -> List[Dict[str, Any]]:
        """
        Validate all items for bulk operation

        Args:
            items: Items to validate
            operation_type: Type of operation

        Returns:
            List of validation errors
        """
        errors = []
        rules = self.validation_rules.get(operation_type.value, [])

        for i, item in enumerate(items):
            for rule in rules:
                try:
                    rule(item)
                except ValueError as e:
                    errors.append({"index": i, "item": item, "error": str(e), "rule": rule.__name__})

        return errors


# Global bulk operation handler
bulk_handler = BulkOperationHandler()
async_bulk_handler = AsyncBulkOperationHandler()
