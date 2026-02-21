"""Domain layer — enums, value objects, and custom exceptions."""

from app.domain.enums import ClassifierType, ConfidenceLevel, WorkflowRoute
from app.domain.errors import (
    BackendUnavailableError,
    ClassificationError,
    ConfigurationError,
    ValidationError,
)

__all__ = [
    "BackendUnavailableError",
    "ClassificationError",
    "ClassifierType",
    "ConfidenceLevel",
    "ConfigurationError",
    "ValidationError",
    "WorkflowRoute",
]
