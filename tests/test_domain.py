"""Tests for app.domain.errors."""

from app.domain.errors import (
    BackendUnavailableError,
    ClassificationError,
    ConfigurationError,
    ValidationError,
)


class TestDomainErrors:
    def test_configuration_error_is_runtime(self):
        err = ConfigurationError("bad config")
        assert isinstance(err, RuntimeError)
        assert str(err) == "bad config"

    def test_backend_unavailable_is_runtime(self):
        err = BackendUnavailableError("no backend")
        assert isinstance(err, RuntimeError)

    def test_classification_error_is_runtime(self):
        err = ClassificationError("failed")
        assert isinstance(err, RuntimeError)

    def test_validation_error_is_value_error(self):
        err = ValidationError("invalid")
        assert isinstance(err, ValueError)
