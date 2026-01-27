"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ClassifierType(str, Enum):
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    INTENT = "intent"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ClassificationRequest(BaseModel):
    """Request model for classification."""
    text: str = Field(..., min_length=1, description="Text to classify")
    classifier_type: ClassifierType = Field(
        default=ClassifierType.SENTIMENT,
        description="Type of classification to perform"
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Custom categories for topic classification"
    )
    intents: Optional[List[str]] = Field(
        None,
        description="Custom intents for intent classification"
    )


class ClassificationResponse(BaseModel):
    """Response model for classification results."""
    text: str = Field(..., description="Original input text")
    classifier_type: str = Field(..., description="Type of classifier used")
    result: Dict[str, Any] = Field(..., description="Classification result")
    success: bool = Field(default=True, description="Whether classification succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class SentimentResponse(BaseModel):
    """Detailed sentiment classification response."""
    sentiment: SentimentType
    confidence: ConfidenceLevel
    reasoning: str


class TopicResponse(BaseModel):
    """Detailed topic classification response."""
    topic: str
    confidence: ConfidenceLevel
    reasoning: str
    available_categories: List[str]


class IntentResponse(BaseModel):
    """Detailed intent classification response."""
    intent: str
    confidence: ConfidenceLevel
    entities: str
    reasoning: str
    available_intents: List[str]


class TrainingExample(BaseModel):
    """Training example for classifier optimization."""
    text: str = Field(..., description="Input text")
    label: str = Field(..., description="Expected label/classification")


class BatchClassificationRequest(BaseModel):
    """Request for batch classification."""
    texts: List[str] = Field(..., min_length=1, description="List of texts to classify")
    classifier_type: ClassifierType = Field(default=ClassifierType.SENTIMENT)
    categories: Optional[List[str]] = None
    intents: Optional[List[str]] = None


class BatchClassificationResponse(BaseModel):
    """Response for batch classification."""
    results: List[ClassificationResponse]
    total: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    classifiers_available: List[str]
