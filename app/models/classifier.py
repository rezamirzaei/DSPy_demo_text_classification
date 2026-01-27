"""
DSPy Classification Models

This module contains DSPy-based classifiers for various text classification tasks:
- Sentiment Analysis
- Topic Classification
- Intent Detection
"""
import dspy
from typing import List, Optional
from dataclasses import dataclass


# ============================================================
# DSPy Signatures for Classification
# ============================================================

class SentimentSignature(dspy.Signature):
    """Classify the sentiment of text as positive, negative, or neutral."""

    text: str = dspy.InputField(desc="The text to analyze for sentiment")
    sentiment: str = dspy.OutputField(desc="The sentiment: 'positive', 'negative', or 'neutral'")
    confidence: str = dspy.OutputField(desc="Confidence level: 'high', 'medium', or 'low'")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")


class TopicSignature(dspy.Signature):
    """Classify the topic/category of the given text."""

    text: str = dspy.InputField(desc="The text to classify")
    categories: str = dspy.InputField(desc="Available categories to choose from")
    topic: str = dspy.OutputField(desc="The most appropriate topic/category")
    confidence: str = dspy.OutputField(desc="Confidence level: 'high', 'medium', or 'low'")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")


class IntentSignature(dspy.Signature):
    """Detect the user's intent from the given text."""

    text: str = dspy.InputField(desc="The user's input text")
    intents: str = dspy.InputField(desc="Available intents to detect")
    intent: str = dspy.OutputField(desc="The detected intent")
    confidence: str = dspy.OutputField(desc="Confidence level: 'high', 'medium', or 'low'")
    entities: str = dspy.OutputField(desc="Key entities extracted from the text (JSON format)")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")


class MultiLabelSignature(dspy.Signature):
    """Assign multiple labels to the given text."""

    text: str = dspy.InputField(desc="The text to classify")
    available_labels: str = dspy.InputField(desc="All available labels")
    labels: str = dspy.OutputField(desc="Comma-separated list of applicable labels")
    confidence: str = dspy.OutputField(desc="Confidence level: 'high', 'medium', or 'low'")
    reasoning: str = dspy.OutputField(desc="Brief explanation for each label assigned")


# ============================================================
# DSPy Classification Modules
# ============================================================

class TextClassifier(dspy.Module):
    """Base text classifier using Chain of Thought reasoning."""

    def __init__(self, signature_class):
        super().__init__()
        self.classifier = dspy.ChainOfThought(signature_class)

    def forward(self, **kwargs) -> dspy.Prediction:
        return self.classifier(**kwargs)


class SentimentClassifier(dspy.Module):
    """
    Sentiment Analysis Classifier

    Classifies text into positive, negative, or neutral sentiment
    with confidence scores and reasoning.
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(SentimentSignature)

    def forward(self, text: str) -> dspy.Prediction:
        result = self.classifier(text=text)

        # Normalize sentiment output
        sentiment = result.sentiment.lower().strip()
        if 'positive' in sentiment:
            sentiment = 'positive'
        elif 'negative' in sentiment:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return dspy.Prediction(
            sentiment=sentiment,
            confidence=result.confidence,
            reasoning=result.reasoning
        )


class TopicClassifier(dspy.Module):
    """
    Topic/Category Classifier

    Classifies text into predefined categories.
    """

    DEFAULT_CATEGORIES = [
        "Technology", "Business", "Science", "Health",
        "Sports", "Entertainment", "Politics", "Education", "Other"
    ]

    def __init__(self, categories: Optional[List[str]] = None):
        super().__init__()
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.classifier = dspy.ChainOfThought(TopicSignature)

    def forward(self, text: str, categories: Optional[List[str]] = None) -> dspy.Prediction:
        cats = categories or self.categories
        categories_str = ", ".join(cats)

        result = self.classifier(text=text, categories=categories_str)

        # Normalize topic to match one of the categories
        detected_topic = result.topic.strip()
        matched_topic = None
        for cat in cats:
            if cat.lower() in detected_topic.lower() or detected_topic.lower() in cat.lower():
                matched_topic = cat
                break

        return dspy.Prediction(
            topic=matched_topic or detected_topic,
            confidence=result.confidence,
            reasoning=result.reasoning,
            available_categories=cats
        )


class IntentClassifier(dspy.Module):
    """
    Intent Detection Classifier

    Detects user intent and extracts entities from text.
    """

    DEFAULT_INTENTS = [
        "question", "command", "greeting", "complaint",
        "feedback", "request", "information", "other"
    ]

    def __init__(self, intents: Optional[List[str]] = None):
        super().__init__()
        self.intents = intents or self.DEFAULT_INTENTS
        self.classifier = dspy.ChainOfThought(IntentSignature)

    def forward(self, text: str, intents: Optional[List[str]] = None) -> dspy.Prediction:
        intent_list = intents or self.intents
        intents_str = ", ".join(intent_list)

        result = self.classifier(text=text, intents=intents_str)

        return dspy.Prediction(
            intent=result.intent,
            confidence=result.confidence,
            entities=result.entities,
            reasoning=result.reasoning,
            available_intents=intent_list
        )


class MultiLabelClassifier(dspy.Module):
    """
    Multi-Label Classifier

    Assigns multiple labels to a single text.
    """

    def __init__(self, labels: List[str]):
        super().__init__()
        self.labels = labels
        self.classifier = dspy.ChainOfThought(MultiLabelSignature)

    def forward(self, text: str) -> dspy.Prediction:
        labels_str = ", ".join(self.labels)

        result = self.classifier(text=text, available_labels=labels_str)

        # Parse labels from comma-separated string
        assigned_labels = [l.strip() for l in result.labels.split(",")]

        return dspy.Prediction(
            labels=assigned_labels,
            confidence=result.confidence,
            reasoning=result.reasoning,
            available_labels=self.labels
        )


# ============================================================
# Classifier Factory
# ============================================================

class ClassifierFactory:
    """Factory for creating different types of classifiers."""

    _classifiers = {
        'sentiment': SentimentClassifier,
        'topic': TopicClassifier,
        'intent': IntentClassifier,
    }

    @classmethod
    def create(cls, classifier_type: str, **kwargs):
        """Create a classifier instance."""
        if classifier_type not in cls._classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}. "
                           f"Available: {list(cls._classifiers.keys())}")
        return cls._classifiers[classifier_type](**kwargs)

    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of available classifier types."""
        return list(cls._classifiers.keys())
