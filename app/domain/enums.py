"""Domain enums used across all layers."""

from enum import Enum


class ClassifierType(str, Enum):
    """Supported classification tasks."""

    SENTIMENT = "sentiment"
    TOPIC = "topic"
    INTENT = "intent"
    MULTI_LABEL = "multi_label"
    ENTITY = "entity"
    AGENT = "agent"

    @classmethod
    def from_string(cls, value: str) -> "ClassifierType":
        try:
            return cls(value.lower().strip())
        except ValueError:
            raise ValueError(
                f"Invalid classifier type '{value}'. "
                f"Choose from: {[e.value for e in cls]}"
            )


class ConfidenceLevel(str, Enum):
    """Normalized confidence buckets."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SentimentType(str, Enum):
    """Allowed sentiment labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class WorkflowRoute(str, Enum):
    """LangGraph agent routing destinations."""

    CLASSIFY = "classify"
    RAG = "rag"
    GRAPH_INFERENCE = "graph_inference"
    UNKNOWN = "unknown"
