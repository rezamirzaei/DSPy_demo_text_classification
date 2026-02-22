"""Tests for app.models.schemas."""

import pytest
from app.domain.enums import ClassifierType
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationRequest,
    ClassificationResponse,
    GraphInferenceRequest,
    HealthResponse,
)


class TestClassificationRequest:
    def test_defaults(self):
        req = ClassificationRequest(text="hello")
        assert req.text == "hello"
        assert req.classifier_type == ClassifierType.SENTIMENT
        assert req.categories is None
        assert req.intents is None
        assert req.labels is None

    def test_custom_type(self):
        req = ClassificationRequest(text="hi", classifier_type=ClassifierType.TOPIC)
        assert req.classifier_type == ClassifierType.TOPIC

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            ClassificationRequest(text="   ")

    def test_with_categories(self):
        req = ClassificationRequest(text="hi", categories=["A", "B"])
        assert req.categories == ["A", "B"]


class TestClassificationResponse:
    def test_defaults(self):
        resp = ClassificationResponse(text="hi")
        assert resp.success is True
        assert resp.timestamp != ""

    def test_error_response(self):
        resp = ClassificationResponse(text="hi", success=False, error="boom")
        assert resp.error == "boom"
        assert resp.success is False


class TestBatchClassificationRequest:
    def test_valid(self):
        req = BatchClassificationRequest(texts=["a", "b"])
        assert len(req.texts) == 2

    def test_empty_texts_raises(self):
        with pytest.raises(Exception):
            BatchClassificationRequest(texts=[])


class TestBatchClassificationResponse:
    def test_model_dump(self):
        results = [ClassificationResponse(text="a", result={"x": "y"})]
        resp = BatchClassificationResponse(
            results=results, total=1, successful=1, failed=0
        )
        data = resp.model_dump()
        assert data["total"] == 1
        assert len(data["results"]) == 1


class TestAgentRequest:
    def test_defaults(self):
        req = AgentRequest(text="test me")
        assert req.enable_knowledge_graph is True

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            AgentRequest(text="  ")


class TestAgentResponse:
    def test_defaults(self):
        resp = AgentResponse(text="test")
        assert resp.success is True
        assert resp.steps == []
        assert resp.timestamp != ""

    def test_with_steps(self):
        resp = AgentResponse(text="x", steps=["a", "b"])
        assert len(resp.steps) == 2


class TestGraphInferenceRequest:
    def test_defaults(self):
        req = GraphInferenceRequest(entity="test")
        assert req.max_depth == 2
        assert req.relation_filter is None
        assert req.entity_type is None

    def test_entity_type_normalization(self):
        req = GraphInferenceRequest(entity="test", entity_type=" concept ")
        assert req.entity_type == "CONCEPT"


class TestHealthResponse:
    def test_fields(self):
        resp = HealthResponse(status="healthy", provider="test", model="m")
        assert resp.version == "2.1.0"
