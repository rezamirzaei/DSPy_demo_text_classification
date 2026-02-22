"""Tests for app.controllers.classification_controller."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.controllers.classification_controller import ClassificationController
from app.domain.enums import ClassifierType
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    BatchClassificationRequest,
    ClassificationRequest,
    GraphInferenceRequest,
)
from app.services.text_analysis import AnalysisResult


class TestClassificationController:
    @pytest.fixture
    def settings(self, tmp_path):
        return SimpleNamespace(graph_data_path=Path(tmp_path) / "graph.json")

    @pytest.fixture
    def controller(self, settings):
        """Controller with mocked DSPy service."""
        with patch("app.controllers.classification_controller.DSPyService") as mock_svc:
            instance = mock_svc.return_value
            instance.is_initialized = True
            instance.model_name = "test-model"
            instance.provider = "test"
            instance.initialize.return_value = True
            ctrl = ClassificationController(settings=settings)
            ctrl.initialize()
            yield ctrl

    def test_is_initialized(self, controller):
        assert controller.is_initialized is True

    def test_model_property(self, controller):
        assert controller.model == "test-model"

    def test_provider_property(self, controller):
        assert controller.provider == "test"

    def test_get_available_classifiers(self, controller):
        types = controller.get_available_classifiers()
        assert "sentiment" in types
        assert "topic" in types
        assert "intent" in types
        assert "entity" in types

    def test_classify_success(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.classify_sentiment.return_value = AnalysisResult(data={
            "sentiment": "positive",
            "confidence": "high",
            "reasoning": "test",
        })

        req = ClassificationRequest(
            text="I love this!",
            classifier_type=ClassifierType.SENTIMENT,
        )
        result = controller.classify(req)
        assert result.success is True
        assert result.result["sentiment"] == "positive"

    def test_classify_failure(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.classify_sentiment.side_effect = RuntimeError("boom")

        req = ClassificationRequest(text="test", classifier_type=ClassifierType.SENTIMENT)
        result = controller.classify(req)
        assert result.success is False
        assert "boom" in (result.error or "")

    def test_batch_classify(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.classify_sentiment.return_value = AnalysisResult(
            data={
                "sentiment": "positive",
                "confidence": "high",
                "reasoning": "test",
            }
        )

        req = BatchClassificationRequest(
            texts=["a", "b"],
            classifier_type=ClassifierType.SENTIMENT,
        )
        result = controller.batch_classify(req)
        assert result.total == 2
        assert result.successful == 2
        assert result.failed == 0

    def test_classify_topic(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.classify_topic.return_value = AnalysisResult(
            data={"topic": "Technology", "confidence": "high"}
        )
        req = ClassificationRequest(
            text="Python code",
            classifier_type=ClassifierType.TOPIC,
        )
        result = controller.classify(req)
        assert result.success is True
        assert result.result["topic"] == "Technology"

    def test_classify_intent(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.classify_intent.return_value = AnalysisResult(
            data={"intent": "question", "confidence": "high"}
        )
        req = ClassificationRequest(
            text="How are you?",
            classifier_type=ClassifierType.INTENT,
        )
        result = controller.classify(req)
        assert result.success is True
        assert result.result["intent"] == "question"

    def test_classify_multi_label(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.classify_multi_label.return_value = AnalysisResult(
            data={"labels": "question,informative", "confidence": "medium"}
        )
        req = ClassificationRequest(
            text="How to install Python 3?",
            classifier_type=ClassifierType.MULTI_LABEL,
        )
        result = controller.classify(req)
        assert result.success is True
        assert "question" in result.result["labels"]

    def test_classify_entity(self, controller):
        controller._analysis_engine = MagicMock()
        controller._analysis_engine.extract_entities.return_value = [
            {"text": "Python", "type": "CONCEPT"}
        ]
        req = ClassificationRequest(
            text="Python is great",
            classifier_type=ClassifierType.ENTITY,
        )
        result = controller.classify(req)
        assert result.success is True
        assert result.result["count"] == 1

    def test_get_knowledge_graph_default(self, controller):
        kg = controller.get_knowledge_graph()
        assert hasattr(kg, "nodes")
        assert hasattr(kg, "edges")
        assert hasattr(kg, "node_count")

    def test_graph_infer(self, controller):
        req = GraphInferenceRequest(entity="unknown")
        result = controller.graph_infer(req)
        assert result.query["name"] == "unknown"
        assert result.matches == []

    @patch("app.agents.classification_agent.ClassificationAgent")
    def test_run_agent_success(self, mock_agent_cls, controller):
        agent = mock_agent_cls.return_value
        agent.analyze.return_value = AgentResponse(text="x", success=True)
        req = AgentRequest(text="Analyze me")
        result = controller.run_agent(req)
        assert result.success is True
        agent.analyze.assert_called_once()

    @patch("app.agents.classification_agent.ClassificationAgent")
    def test_run_agent_failure(self, mock_agent_cls, controller):
        agent = mock_agent_cls.return_value
        agent.analyze.side_effect = RuntimeError("agent boom")
        req = AgentRequest(text="Analyze me")
        result = controller.run_agent(req)
        assert result.success is False
        assert "agent boom" in (result.error or "")

    def test_get_knowledge_graph_from_agent(self, controller):
        from app.models.schemas import KnowledgeGraphExport
        controller._agent = MagicMock()
        controller._agent.get_knowledge_graph.return_value = KnowledgeGraphExport(
            nodes=[{"name": "x", "type": "TEST"}], edges=[], node_count=1, edge_count=0,
        )
        result = controller.get_knowledge_graph()
        assert result.nodes == [{"name": "x", "type": "TEST"}]

    def test_reseed_knowledge_graph(self, controller):
        result = controller.reseed_knowledge_graph()
        assert result.message == "Knowledge graph seeded successfully"
        assert result.node_count > 0
        assert result.edge_count > 0

    def test_reseed_clears_and_rebuilds(self, controller):
        # First seed should have nodes
        kg_before = controller.get_knowledge_graph()
        initial_count = kg_before.node_count
        assert initial_count > 0

        # Reseed clears and rebuilds
        result = controller.reseed_knowledge_graph()
        assert result.node_count > 0
