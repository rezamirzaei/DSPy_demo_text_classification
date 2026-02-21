"""Tests for app.agents.classification_agent."""

from __future__ import annotations

from typing import Any

from app.agents.classification_agent import ClassificationAgent
from app.services.knowledge_graph import KnowledgeGraph
from app.services.text_analysis import AnalysisResult, TextAnalysisEngine


class FakeEngine(TextAnalysisEngine):
    def classify_sentiment(self, text: str) -> AnalysisResult:
        return AnalysisResult({"sentiment": "positive", "confidence": "high"})

    def classify_topic(self, text: str, categories=None) -> AnalysisResult:
        return AnalysisResult({"topic": "Technology", "confidence": "high"})

    def classify_intent(self, text: str, intents=None) -> AnalysisResult:
        return AnalysisResult({"intent": "question", "confidence": "high"})

    def classify_multi_label(self, text: str, labels=None) -> AnalysisResult:
        return AnalysisResult({"labels": "informative", "confidence": "medium"})

    def extract_entities(self, text: str):
        return [
            {"text": "Python", "type": "CONCEPT"},
            {"text": "LangGraph", "type": "CONCEPT"},
        ]

    def summarize(self, text: str, sentiment: dict[str, Any], topic: dict[str, Any], intent: dict[str, Any], entities):
        return "summary"


def test_agent_analyze_with_kg():
    agent = ClassificationAgent(
        analysis_engine=FakeEngine(),
        knowledge_graph=KnowledgeGraph(),
        enable_knowledge_graph=True,
    )

    result = agent.analyze("Tell me about Python")
    assert result.success is True
    assert "knowledge_graph_construction" in result.steps
    assert result.knowledge_graph["node_count"] >= 2


def test_agent_analyze_without_kg():
    agent = ClassificationAgent(
        analysis_engine=FakeEngine(),
        knowledge_graph=KnowledgeGraph(),
        enable_knowledge_graph=True,
    )

    result = agent.analyze("Tell me about Python", include_knowledge_graph=False)
    assert result.success is True
    assert "knowledge_graph_construction" not in result.steps


def test_agent_infer_entity():
    agent = ClassificationAgent(
        analysis_engine=FakeEngine(),
        knowledge_graph=KnowledgeGraph(),
        enable_knowledge_graph=True,
    )
    agent.analyze("Tell me about Python")

    inference = agent.infer_entity("Python")
    assert inference["query"]["name"] == "Python"
    assert isinstance(inference["related"], list)
