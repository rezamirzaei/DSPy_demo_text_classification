"""Domain-specific exceptions for clean error handling."""


class ConfigurationError(RuntimeError):
    """Raised when runtime configuration is invalid or missing."""


class BackendUnavailableError(RuntimeError):
    """Raised when a required model backend cannot be reached."""


class ClassificationError(RuntimeError):
    """Raised when a classification pipeline step fails."""


class ValidationError(ValueError):
    """Raised when domain-level validation fails."""
