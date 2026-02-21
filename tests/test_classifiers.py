"""Tests for app.models.classifier."""

import pytest
from unittest.mock import patch, MagicMock
from app.models.classifier import (
    ClassifierFactory,
    SentimentClassifier,
)


class TestClassifierFactory:
    def test_available_types(self):
        types = ClassifierFactory.available_types()
        assert "sentiment" in types
        assert "topic" in types
        assert "intent" in types
        assert "multi_label" in types
        assert "entity" in types
        assert "summarizer" in types

    def test_create_valid(self):
        with patch("dspy.ChainOfThought") as mock_cot:
            mock_cot.return_value = MagicMock()
            ClassifierFactory.create("sentiment")
            mock_cot.assert_called_once_with(SentimentClassifier)

    def test_create_invalid(self):
        with pytest.raises(ValueError, match="Unknown classifier type"):
            ClassifierFactory.create("nonexistent")

    def test_register_custom(self):
        class Custom:
            pass
        ClassifierFactory.register("custom", Custom)
        assert "custom" in ClassifierFactory.available_types()
        # Cleanup
        del ClassifierFactory._registry["custom"]


class TestSentimentClassifier:
    def test_normalize_confidence_high(self):
        assert SentimentClassifier.normalize_confidence("very high") == "high"

    def test_normalize_confidence_low(self):
        assert SentimentClassifier.normalize_confidence("low certainty") == "low"

    def test_normalize_confidence_medium(self):
        assert SentimentClassifier.normalize_confidence("moderate") == "medium"
