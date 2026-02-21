"""Pytest configuration & shared fixtures."""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables before any imports
os.environ.setdefault("PROVIDER", "rule_based")
os.environ.setdefault("HOST", "127.0.0.1")
os.environ.setdefault("PORT", "8000")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        "positive": "I absolutely love this amazing product!",
        "negative": "This is the worst terrible experience ever.",
        "neutral": "The meeting is scheduled for 3 PM.",
    }


@pytest.fixture
def app_client():
    """Create a Flask test client with mock controller."""
    from app.views.routes import create_app

    class MockController:
        is_initialized = True
        model = "test-model"
        provider = "test"

        def get_available_classifiers(self):
            return ["sentiment", "topic", "intent"]

        def classify(self, request):
            from app.models.schemas import ClassificationResponse
            return ClassificationResponse(
                text=request.text,
                classifier_type=request.classifier_type.value,
                result={"sentiment": "positive", "confidence": "high", "reasoning": "test"},
                success=True,
            )

        def batch_classify(self, request):
            from app.models.schemas import (
                BatchClassificationResponse,
                ClassificationResponse,
            )
            results = [
                ClassificationResponse(
                    text=t,
                    classifier_type=request.classifier_type.value,
                    result={"sentiment": "positive", "confidence": "high", "reasoning": "test"},
                    success=True,
                )
                for t in request.texts
            ]
            return BatchClassificationResponse(
                results=results,
                total=len(request.texts),
                successful=len(request.texts),
                failed=0,
            )

        def run_agent(self, request):
            from app.models.schemas import AgentResponse
            return AgentResponse(
                text=request.text,
                sentiment={"sentiment": "positive"},
                topic={"topic": "Technology"},
                intent={"intent": "question"},
                entities=[{"text": "test", "type": "CONCEPT"}],
                summary="Test summary",
                success=True,
                steps=["sentiment_analysis", "topic_classification"],
            )

        def get_knowledge_graph(self):
            return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

        def graph_infer(self, request):
            return {
                "query": {
                    "name": request.entity,
                    "type": request.entity_type,
                    "max_depth": request.max_depth,
                    "relation_filter": request.relation_filter,
                },
                "matches": [],
                "related": [],
                "predicted_links": [],
            }

    app = create_app(MockController())
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
