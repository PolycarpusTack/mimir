"""
Custom exception hierarchy for Mimir AI/NLP modules.
Provides specific exception types for better error handling and debugging.
"""

from typing import Any, Dict, Optional


class MimirAIException(Exception):
    """Base exception for all Mimir AI-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadingError(MimirAIException):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize model loading error.

        Args:
            model_name: Name of the model that failed to load
            reason: Reason for the failure
            details: Optional additional details
        """
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, details)
        self.model_name = model_name
        self.reason = reason


class ModelNotAvailableError(MimirAIException):
    """Raised when a required model is not available."""

    def __init__(self, model_type: str, language: Optional[str] = None):
        """
        Initialize model not available error.

        Args:
            model_type: Type of model (e.g., 'NER', 'sentiment')
            language: Optional language code
        """
        if language:
            message = f"No {model_type} model available for language: {language}"
        else:
            message = f"No {model_type} model available"
        super().__init__(message)
        self.model_type = model_type
        self.language = language


class TextProcessingError(MimirAIException):
    """Raised when text processing fails."""

    def __init__(self, operation: str, reason: str, text_sample: Optional[str] = None):
        """
        Initialize text processing error.

        Args:
            operation: The operation that failed (e.g., 'tokenization', 'encoding')
            reason: Reason for the failure
            text_sample: Optional sample of the problematic text
        """
        message = f"Text processing failed during {operation}: {reason}"
        details = {}
        if text_sample:
            # Truncate text sample for security
            details["text_sample"] = text_sample[:100] + "..." if len(text_sample) > 100 else text_sample
        super().__init__(message, details)
        self.operation = operation
        self.reason = reason


class EntityExtractionError(MimirAIException):
    """Raised when entity extraction fails."""

    def __init__(self, entity_type: Optional[str] = None, reason: str = "Unknown error"):
        """
        Initialize entity extraction error.

        Args:
            entity_type: Type of entity being extracted
            reason: Reason for the failure
        """
        if entity_type:
            message = f"Failed to extract {entity_type} entities: {reason}"
        else:
            message = f"Entity extraction failed: {reason}"
        super().__init__(message)
        self.entity_type = entity_type


class ClassificationError(MimirAIException):
    """Raised when classification fails."""

    def __init__(self, classification_type: str, reason: str):
        """
        Initialize classification error.

        Args:
            classification_type: Type of classification (e.g., 'sentiment', 'industry')
            reason: Reason for the failure
        """
        message = f"{classification_type} classification failed: {reason}"
        super().__init__(message)
        self.classification_type = classification_type


class InvalidInputError(MimirAIException):
    """Raised when input validation fails."""

    def __init__(self, input_name: str, expected: str, received: Any):
        """
        Initialize invalid input error.

        Args:
            input_name: Name of the input parameter
            expected: Expected input format/type
            received: What was actually received
        """
        message = f"Invalid input for '{input_name}': expected {expected}, received {type(received).__name__}"
        details = {"input_name": input_name, "expected": expected, "received_type": type(received).__name__}
        super().__init__(message, details)
        self.input_name = input_name
        self.expected = expected
        self.received = received


class ConfigurationError(MimirAIException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, config_name: str, reason: str):
        """
        Initialize configuration error.

        Args:
            config_name: Name of the configuration item
            reason: Reason for the error
        """
        message = f"Configuration error for '{config_name}': {reason}"
        super().__init__(message)
        self.config_name = config_name


class ExternalServiceError(MimirAIException):
    """Raised when an external service call fails."""

    def __init__(
        self, service_name: str, operation: str, status_code: Optional[int] = None, reason: Optional[str] = None
    ):
        """
        Initialize external service error.

        Args:
            service_name: Name of the external service
            operation: Operation being performed
            status_code: Optional HTTP status code
            reason: Optional reason for failure
        """
        message = f"{service_name} {operation} failed"
        if status_code:
            message += f" (status: {status_code})"
        if reason:
            message += f": {reason}"

        details = {"service": service_name, "operation": operation}
        if status_code:
            details["status_code"] = status_code

        super().__init__(message, details)
        self.service_name = service_name
        self.operation = operation
        self.status_code = status_code


class ResourceNotFoundError(MimirAIException):
    """Raised when a required resource is not found."""

    def __init__(self, resource_type: str, resource_id: str):
        """
        Initialize resource not found error.

        Args:
            resource_type: Type of resource (e.g., 'model', 'config', 'file')
            resource_id: Identifier of the resource
        """
        message = f"{resource_type} not found: {resource_id}"
        super().__init__(message)
        self.resource_type = resource_type
        self.resource_id = resource_id


class TrainingError(MimirAIException):
    """Raised when model training fails."""

    def __init__(self, model_type: str, reason: str, epoch: Optional[int] = None):
        """
        Initialize training error.

        Args:
            model_type: Type of model being trained
            reason: Reason for the failure
            epoch: Optional epoch number where failure occurred
        """
        message = f"Training failed for {model_type}: {reason}"
        if epoch is not None:
            message += f" (at epoch {epoch})"
        super().__init__(message)
        self.model_type = model_type
        self.epoch = epoch


class DataValidationError(MimirAIException):
    """Raised when data validation fails."""

    def __init__(self, data_type: str, validation_rule: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize data validation error.

        Args:
            data_type: Type of data being validated
            validation_rule: Rule that failed
            details: Optional validation details
        """
        message = f"Data validation failed for {data_type}: {validation_rule}"
        super().__init__(message, details)
        self.data_type = data_type
        self.validation_rule = validation_rule


class MemoryError(MimirAIException):
    """Raised when memory limits are exceeded."""

    def __init__(self, operation: str, required_memory: Optional[str] = None, available_memory: Optional[str] = None):
        """
        Initialize memory error.

        Args:
            operation: Operation that failed due to memory
            required_memory: Optional required memory amount
            available_memory: Optional available memory amount
        """
        message = f"Insufficient memory for {operation}"
        details = {}
        if required_memory:
            message += f" (required: {required_memory}"
            details["required_memory"] = required_memory
        if available_memory:
            message += f", available: {available_memory}"
            details["available_memory"] = available_memory
        if required_memory or available_memory:
            message += ")"

        super().__init__(message, details)
        self.operation = operation


class TimeoutError(MimirAIException):
    """Raised when an operation times out."""

    def __init__(self, operation: str, timeout_seconds: float):
        """
        Initialize timeout error.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
        """
        message = f"{operation} timed out after {timeout_seconds} seconds"
        super().__init__(message)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# Utility functions for error handling


def handle_model_error(e: Exception, model_name: str, operation: str) -> None:
    """
    Convert generic exceptions to specific model errors.

    Args:
        e: Original exception
        model_name: Name of the model
        operation: Operation being performed

    Raises:
        ModelLoadingError or ModelNotAvailableError
    """
    if "not found" in str(e).lower() or "no such file" in str(e).lower():
        raise ResourceNotFoundError("model", model_name)
    elif "out of memory" in str(e).lower():
        raise MemoryError(f"loading {model_name}")
    else:
        raise ModelLoadingError(model_name, str(e))


def validate_text_input(text: Any, min_length: int = 1, max_length: Optional[int] = None) -> str:
    """
    Validate text input.

    Args:
        text: Input to validate
        min_length: Minimum text length
        max_length: Optional maximum text length

    Returns:
        Validated text string

    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(text, str):
        raise InvalidInputError("text", "string", text)

    if len(text) < min_length:
        raise InvalidInputError("text", f"string with length >= {min_length}", f"string of length {len(text)}")

    if max_length and len(text) > max_length:
        raise InvalidInputError("text", f"string with length <= {max_length}", f"string of length {len(text)}")

    return text
