"""Tests for app.models.schemas."""

import pytest

from app.domain.enums import ClassifierType
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    AnalysisResultModel,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationRequest,
    ClassificationResponse,
    EntityItem,
    GraphInferenceRequest,
    GraphInferenceResponse,
    HealthResponse,
    KGEnrichmentResult,
    KnowledgeGraphExport,
    QualityReport,
    ReseedResponse,
    ServiceHealthInfo,
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


class TestEntityItem:
    def test_defaults(self):
        entity = EntityItem(text="Python")
        assert entity.type == "CONCEPT"

    def test_custom_type(self):
        entity = EntityItem(text="Python", type="LANGUAGE")
        assert entity.type == "LANGUAGE"


class TestAnalysisResultModel:
    def test_empty(self):
        result = AnalysisResultModel()
        assert result.data == {}

    def test_with_data(self):
        result = AnalysisResultModel(data={"key": "value"})
        assert result.data["key"] == "value"

    def test_model_dump(self):
        result = AnalysisResultModel(data={"sentiment": "positive"})
        dumped = result.model_dump()
        assert dumped["data"]["sentiment"] == "positive"


class TestQualityReport:
    def test_valid(self):
        report = QualityReport(score=85, grade="B", issues=["low_confidence"])
        assert report.score == 85
        assert report.grade == "B"
        assert "low_confidence" in report.issues

    def test_score_boundaries(self):
        report = QualityReport(score=0, grade="F")
        assert report.score == 0
        report = QualityReport(score=100, grade="A")
        assert report.score == 100


class TestKGEnrichmentResult:
    def test_defaults(self):
        result = KGEnrichmentResult()
        assert result.entities_found_in_kg == 0
        assert result.entity_matches == []

    def test_with_error(self):
        result = KGEnrichmentResult(error="connection failed")
        assert result.error == "connection failed"


class TestServiceHealthInfo:
    def test_defaults(self):
        info = ServiceHealthInfo()
        assert info.initialized is False
        assert info.provider == "unknown"

    def test_custom(self):
        info = ServiceHealthInfo(initialized=True, provider="ollama", model="llama3.2")
        assert info.initialized is True
        assert info.model == "llama3.2"


class TestReseedResponse:
    def test_valid(self):
        resp = ReseedResponse(message="ok", node_count=10, edge_count=20)
        assert resp.message == "ok"
        assert resp.node_count == 10

    def test_model_dump(self):
        resp = ReseedResponse(message="ok", node_count=5, edge_count=3)
        dumped = resp.model_dump()
        assert dumped["edge_count"] == 3


class TestKnowledgeGraphExport:
    def test_defaults(self):
        export = KnowledgeGraphExport()
        assert export.nodes == []
        assert export.node_count == 0
        assert export.inferences is None

    def test_with_data(self):
        export = KnowledgeGraphExport(
            nodes=[{"name": "test", "type": "X"}],
            edges=[],
            node_count=1,
            edge_count=0,
        )
        assert export.node_count == 1


class TestGraphInferenceResponse:
    def test_valid(self):
        resp = GraphInferenceResponse(
            query={"name": "test", "type": None},
            matches=[{"name": "test", "type": "CONCEPT"}],
        )
        assert len(resp.matches) == 1

    def test_defaults(self):
        resp = GraphInferenceResponse(query={"name": "x"})
        assert resp.related == []
        assert resp.predicted_links == []


