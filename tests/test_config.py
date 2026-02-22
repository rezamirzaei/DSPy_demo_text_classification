"""Tests for app.domain.enums."""

import pytest

from app.domain.enums import (
    ClassifierType,
    ConfidenceLevel,
    SentimentType,
    WorkflowRoute,
)


class TestClassifierType:
    def test_values(self):
        assert ClassifierType.SENTIMENT == "sentiment"
        assert ClassifierType.TOPIC == "topic"
        assert ClassifierType.INTENT == "intent"
        assert ClassifierType.MULTI_LABEL == "multi_label"
        assert ClassifierType.ENTITY == "entity"
        assert ClassifierType.AGENT == "agent"

    def test_from_string_valid(self):
        assert ClassifierType.from_string("sentiment") == ClassifierType.SENTIMENT
        assert ClassifierType.from_string("  TOPIC  ") == ClassifierType.TOPIC

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid classifier type"):
            ClassifierType.from_string("nonexistent")


class TestConfidenceLevel:
    def test_values(self):
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"


class TestSentimentType:
    def test_values(self):
        assert SentimentType.POSITIVE == "positive"
        assert SentimentType.NEGATIVE == "negative"
        assert SentimentType.NEUTRAL == "neutral"


class TestWorkflowRoute:
    def test_values(self):
        assert WorkflowRoute.CLASSIFY == "classify"
        assert WorkflowRoute.RAG == "rag"
        assert WorkflowRoute.GRAPH_INFERENCE == "graph_inference"
        assert WorkflowRoute.UNKNOWN == "unknown"
