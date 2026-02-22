"""Tests for app.models.optimizer — DSPy BootstrapFewShot integration."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("PROVIDER", "rule_based")
os.environ.setdefault("ENVIRONMENT", "test")

from app.models.optimizer import (
    INTENT_EXAMPLES,
    SENTIMENT_EXAMPLES,
    TOPIC_EXAMPLES,
    DSPyOptimizer,
    _metric_exact_match,
)


class TestTrainingExamples:
    """Verify training example quality and structure."""

    def test_sentiment_examples_not_empty(self):
        assert len(SENTIMENT_EXAMPLES) >= 4

    def test_topic_examples_not_empty(self):
        assert len(TOPIC_EXAMPLES) >= 3

    def test_intent_examples_not_empty(self):
        assert len(INTENT_EXAMPLES) >= 3

    def test_sentiment_examples_have_inputs(self):
        for ex in SENTIMENT_EXAMPLES:
            assert hasattr(ex, "text")
            assert hasattr(ex, "sentiment")
            assert hasattr(ex, "confidence")

    def test_topic_examples_have_inputs(self):
        for ex in TOPIC_EXAMPLES:
            assert hasattr(ex, "text")
            assert hasattr(ex, "topic")

    def test_intent_examples_have_inputs(self):
        for ex in INTENT_EXAMPLES:
            assert hasattr(ex, "text")
            assert hasattr(ex, "intent")

    def test_sentiment_labels_valid(self):
        valid = {"positive", "negative", "neutral"}
        for ex in SENTIMENT_EXAMPLES:
            assert ex.sentiment in valid, f"Invalid sentiment: {ex.sentiment}"


class TestMetricExactMatch:
    def test_matching_sentiment(self):
        example = MagicMock(sentiment="positive", topic=None, intent=None)
        prediction = MagicMock(sentiment="positive", topic=None, intent=None)
        assert _metric_exact_match(example, prediction) is True

    def test_non_matching_sentiment(self):
        example = MagicMock(sentiment="positive", topic=None, intent=None)
        prediction = MagicMock(sentiment="negative", topic=None, intent=None)
        assert _metric_exact_match(example, prediction) is False

    def test_case_insensitive(self):
        example = MagicMock(sentiment="Positive", topic=None, intent=None)
        prediction = MagicMock(sentiment="positive", topic=None, intent=None)
        assert _metric_exact_match(example, prediction) is True

    def test_topic_match(self):
        example = MagicMock(sentiment=None, topic="Technology", intent=None)
        prediction = MagicMock(sentiment=None, topic="technology", intent=None)
        assert _metric_exact_match(example, prediction) is True

    def test_intent_match(self):
        example = MagicMock(sentiment=None, topic=None, intent="question")
        prediction = MagicMock(sentiment=None, topic=None, intent="Question")
        assert _metric_exact_match(example, prediction) is True

    def test_no_matching_fields(self):
        example = MagicMock(sentiment=None, topic=None, intent=None)
        prediction = MagicMock(sentiment=None, topic=None, intent=None)
        assert _metric_exact_match(example, prediction) is False


class TestDSPyOptimizer:
    @pytest.fixture
    def optimizer(self, tmp_path):
        return DSPyOptimizer(cache_dir=tmp_path)

    def test_init_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "dspy_cache"
        DSPyOptimizer(cache_dir=cache)
        assert cache.exists()

    def test_optimize_no_examples_returns_module(self, optimizer):
        module = MagicMock()
        result = optimizer.optimize("test", module, [])
        assert result is module

    def test_optimize_loads_from_cache(self, optimizer, tmp_path):
        cache_path = tmp_path / "test.json"
        cache_path.write_text("{}")

        module = MagicMock()
        module.load = MagicMock()

        optimizer.optimize("test", module, SENTIMENT_EXAMPLES)
        module.load.assert_called_once_with(str(cache_path))

    def test_optimize_handles_cache_load_failure(self, optimizer, tmp_path):
        """When cache exists but load fails, should attempt re-optimization."""
        cache_path = tmp_path / "test.json"
        cache_path.write_text("{}")

        module = MagicMock()
        module.load.side_effect = RuntimeError("corrupt cache")

        with patch("app.models.optimizer.dspy.BootstrapFewShot") as mock_bs:
            mock_bs.return_value.compile.return_value = module
            optimizer.optimize("test", module, SENTIMENT_EXAMPLES)
            mock_bs.return_value.compile.assert_called_once()

    def test_optimize_handles_optimization_failure(self, optimizer, tmp_path):
        module = MagicMock()

        with patch("app.models.optimizer.dspy.BootstrapFewShot") as mock_bs:
            mock_bs.return_value.compile.side_effect = RuntimeError("failed")
            result = optimizer.optimize("test_fail", module, SENTIMENT_EXAMPLES)
            assert result is module  # Falls back to original module

    def test_default_cache_dir(self):
        opt = DSPyOptimizer()
        assert opt._cache_dir == Path("data/dspy_optimized")


