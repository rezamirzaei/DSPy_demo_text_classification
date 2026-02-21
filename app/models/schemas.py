"""Pydantic request/response schemas (Model Layer)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.domain.enums import ClassifierType


class TimestampedModel(BaseModel):
    """Shared model base with auto-generated timestamp."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Requests ──────────────────────────────────────────────


class ClassificationRequest(BaseModel):
    """Single-text classification request."""

    text: str
    classifier_type: ClassifierType = ClassifierType.SENTIMENT
    categories: Optional[List[str]] = None
    intents: Optional[List[str]] = None
    labels: Optional[List[str]] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text must not be empty")
        return normalized


class BatchClassificationRequest(BaseModel):
    """Multi-text classification request."""

    texts: List[str]
    classifier_type: ClassifierType = ClassifierType.SENTIMENT
    categories: Optional[List[str]] = None

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, values: List[str]) -> List[str]:
        normalized = [value.strip() for value in values if value and value.strip()]
        if not normalized:
            raise ValueError("Texts list must not be empty")
        return normalized


class AgentRequest(BaseModel):
    """LangGraph agent analysis request."""

    text: str
    enable_knowledge_graph: bool = True

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text must not be empty")
        return normalized


class GraphInferenceRequest(BaseModel):
    """Graph inference request."""

    entity: str
    entity_type: Optional[str] = None
    max_depth: int = Field(default=2, ge=1, le=6)
    relation_filter: Optional[str] = None

    @field_validator("entity")
    @classmethod
    def entity_must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Entity must not be empty")
        return normalized

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None


# ── Responses ─────────────────────────────────────────────


class ClassificationResponse(TimestampedModel):
    """Single classification result."""

    text: str
    classifier_type: str = "sentiment"
    result: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class BatchClassificationResponse(BaseModel):
    """Batch classification results."""

    results: List[ClassificationResponse]
    total: int
    successful: int
    failed: int


class AgentResponse(TimestampedModel):
    """Full agent analysis result."""

    text: str = ""
    sentiment: Dict[str, Any] = Field(default_factory=dict)
    topic: Dict[str, Any] = Field(default_factory=dict)
    intent: Dict[str, Any] = Field(default_factory=dict)
    entities: List[Dict[str, str]] = Field(default_factory=list)
    knowledge_graph: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    success: bool = True
    error: Optional[str] = None
    steps: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    provider: str
    model: str
    initialized: bool = False
    classifiers_available: List[str] = Field(default_factory=list)
    version: str = "2.0.0"
