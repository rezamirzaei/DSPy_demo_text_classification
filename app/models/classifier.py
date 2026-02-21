"""DSPy Classifier Models (Model Layer - OOP)."""

import logging
from typing import Any, Dict, Type

import dspy

from app.domain.enums import ConfidenceLevel

logger = logging.getLogger(__name__)


# ── DSPy Signatures ──────────────────────────────────────

class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of the given text."""

    text = dspy.InputField(desc="Text to analyze")
    sentiment = dspy.OutputField(desc="One of: positive, negative, neutral")
    confidence = dspy.OutputField(desc="One of: high, medium, low")
    reasoning = dspy.OutputField(desc="Brief explanation")

    @staticmethod
    def normalize_confidence(raw: str) -> str:
        """Map free-form confidence strings to ConfidenceLevel values."""
        raw_lower = raw.lower()
        if "high" in raw_lower:
            return ConfidenceLevel.HIGH.value
        if "low" in raw_lower:
            return ConfidenceLevel.LOW.value
        return ConfidenceLevel.MEDIUM.value


class TopicClassifier(dspy.Signature):
    """Classify the topic/category of the given text."""

    text = dspy.InputField(desc="Text to classify")
    categories = dspy.InputField(
        desc="Available categories",
        default="Technology, Science, Business, Health, Sports, "
        "Politics, Entertainment, Education, Other",
    )
    topic = dspy.OutputField(desc="Most relevant category")
    confidence = dspy.OutputField(desc="One of: high, medium, low")
    reasoning = dspy.OutputField(desc="Brief explanation")


class IntentClassifier(dspy.Signature):
    """Detect the user's intent from the given text."""

    text = dspy.InputField(desc="Text to analyze")
    intents = dspy.InputField(
        desc="Available intents",
        default="question, request, complaint, feedback, "
        "greeting, farewell, information, other",
    )
    intent = dspy.OutputField(desc="Detected intent")
    confidence = dspy.OutputField(desc="One of: high, medium, low")
    entities = dspy.OutputField(desc="Key entities found")
    reasoning = dspy.OutputField(desc="Brief explanation")


class MultiLabelClassifier(dspy.Signature):
    """Assign multiple labels to the given text."""

    text = dspy.InputField(desc="Text to label")
    available_labels = dspy.InputField(
        desc="Available labels",
        default="informative, opinion, question, instructional, "
        "persuasive, narrative",
    )
    labels = dspy.OutputField(desc="Comma-separated applicable labels")
    confidence = dspy.OutputField(desc="One of: high, medium, low")
    reasoning = dspy.OutputField(desc="Brief explanation")


class EntityExtractor(dspy.Signature):
    """Extract named entities from text."""

    text = dspy.InputField(desc="Text to analyze")
    entities = dspy.OutputField(
        desc='JSON list of entities: '
        '[{"text": "...", "type": "ORG|PERSON|LOC|PRODUCT|CONCEPT"}]',
    )
    count = dspy.OutputField(desc="Number of entities found")


class AnalysisSummarizer(dspy.Signature):
    """Summarize a multi-step text analysis."""

    text = dspy.InputField(desc="Original text")
    sentiment = dspy.InputField(desc="Sentiment analysis result")
    topic = dspy.InputField(desc="Topic classification result")
    intent = dspy.InputField(desc="Intent detection result")
    entities = dspy.InputField(desc="Extracted entities")
    summary = dspy.OutputField(desc="Concise summary of all analysis results")


# ── Factory ──────────────────────────────────────────────

class ClassifierFactory:
    """Factory for creating DSPy classifier instances (OOP pattern)."""

    _registry: Dict[str, Type] = {
        "sentiment": SentimentClassifier,
        "topic": TopicClassifier,
        "intent": IntentClassifier,
        "multi_label": MultiLabelClassifier,
        "entity": EntityExtractor,
        "summarizer": AnalysisSummarizer,
    }

    @classmethod
    def create(cls, classifier_type: str) -> Any:
        """Instantiate a ChainOfThought module for the requested type."""
        if classifier_type not in cls._registry:
            raise ValueError(
                f"Unknown classifier type: {classifier_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return dspy.ChainOfThought(cls._registry[classifier_type])

    @classmethod
    def available_types(cls) -> list:
        """Return list of registered classifier type strings."""
        return list(cls._registry.keys())

    @classmethod
    def register(cls, name: str, classifier_class: Type) -> None:
        """Register a new classifier type at runtime."""
        cls._registry[name] = classifier_class
