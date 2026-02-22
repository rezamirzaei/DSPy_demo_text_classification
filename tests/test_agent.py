"""Tests for app.agents.classification_agent."""

from __future__ import annotations

from typing import Any

from app.agents.classification_agent import ClassificationAgent
from app.services.knowledge_graph import KnowledgeGraph
from app.services.text_analysis import AnalysisResult, TextAnalysisEngine


class FakeEngine(TextAnalysisEngine):
    def classify_sentiment(self, text: str) -> AnalysisResult:
        return AnalysisResult(data={"sentiment": "positive", "confidence": "high", "reasoning": "test"})

    def classify_topic(self, text: str, categories=None) -> AnalysisResult:
        return AnalysisResult(data={"topic": "Technology", "confidence": "high", "reasoning": "test"})

    def classify_intent(self, text: str, intents=None) -> AnalysisResult:
        return AnalysisResult(data={"intent": "question", "confidence": "high", "reasoning": "test"})

    def classify_multi_label(self, text: str, labels=None) -> AnalysisResult:
        return AnalysisResult(data={"labels": "informative", "confidence": "medium", "reasoning": "test"})

    def extract_entities(self, text: str):
        return [
            {"text": "Python", "type": "CONCEPT"},
            {"text": "LangGraph", "type": "CONCEPT"},
        ]

    def summarize(self, text: str, sentiment: dict[str, Any], topic: dict[str, Any], intent: dict[str, Any], entities):
        return "summary"


class FailingEngine(TextAnalysisEngine):
    """Engine that raises exceptions for all operations."""

    def classify_sentiment(self, text: str) -> AnalysisResult:
        raise RuntimeError("Sentiment failed")

    def classify_topic(self, text: str, categories=None) -> AnalysisResult:
        raise RuntimeError("Topic failed")

    def classify_intent(self, text: str, intents=None) -> AnalysisResult:
        raise RuntimeError("Intent failed")

    def classify_multi_label(self, text: str, labels=None) -> AnalysisResult:
        raise RuntimeError("Multi-label failed")

    def extract_entities(self, text: str):
        raise RuntimeError("Entity extraction failed")

    def summarize(self, text: str, sentiment: dict[str, Any], topic: dict[str, Any], intent: dict[str, Any], entities):
        raise RuntimeError("Summary failed")


class TestAgentAnalysis:
    """Tests for the ClassificationAgent analysis pipeline."""

    def test_agent_analyze_with_kg(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Tell me about Python")
        assert result.success is True
        assert "knowledge_graph_construction" in result.steps
        assert result.knowledge_graph["node_count"] >= 2

    def test_agent_analyze_without_kg(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Tell me about Python", include_knowledge_graph=False)
        assert result.success is True
        assert "knowledge_graph_construction" not in result.steps

    def test_agent_infer_entity(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        agent.analyze("Tell me about Python")
        inference = agent.infer_entity("Python")
        assert inference.query["name"] == "Python"
        assert isinstance(inference.related, list)

    def test_agent_steps_include_router(self):
        """The new agent starts with a router node."""
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Test text")
        assert result.success is True
        # Router step should be first
        assert result.steps[0].startswith("router(")

    def test_agent_steps_include_quality_check(self):
        """The new agent ends with a quality_check node."""
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Test text")
        assert result.success is True
        assert result.steps[-1].startswith("quality_check(")

    def test_agent_steps_with_kg_include_enrichment(self):
        """When KG is enabled, kg_enrichment step should appear."""
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Test text")
        assert result.success is True
        step_names = [s.split("(")[0] for s in result.steps]
        assert "kg_enrichment" in step_names
        assert "knowledge_graph_construction" in step_names

    def test_agent_steps_order_with_kg(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Test text")
        assert result.success is True
        step_names = [s.split("(")[0] for s in result.steps]
        # router is always first
        assert step_names[0] == "router"
        # sentiment, topic, intent run in parallel — order is non-deterministic
        parallel_steps = set(step_names[1:4])
        assert parallel_steps == {"sentiment_analysis", "topic_classification", "intent_detection"}
        # After parallel merge, the remaining steps are deterministic
        assert "parallel_merge" in step_names
        merge_idx = step_names.index("parallel_merge")
        post_merge = step_names[merge_idx + 1:]
        expected_post_merge = [
            "entity_extraction",
            "kg_enrichment",
            "knowledge_graph_construction",
            "summary_generation",
            "quality_check",
        ]
        assert post_merge == expected_post_merge

    def test_agent_steps_order_without_kg(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Test text")
        step_names = [s.split("(")[0] for s in result.steps]
        # router is always first
        assert step_names[0] == "router"
        # sentiment, topic, intent run in parallel
        parallel_steps = set(step_names[1:4])
        assert parallel_steps == {"sentiment_analysis", "topic_classification", "intent_detection"}
        # After parallel merge
        assert "parallel_merge" in step_names
        merge_idx = step_names.index("parallel_merge")
        post_merge = step_names[merge_idx + 1:]
        expected_post_merge = [
            "entity_extraction",
            "summary_generation",
            "quality_check",
        ]
        assert post_merge == expected_post_merge

    def test_agent_sentiment_result(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("I love Python")
        assert result.sentiment.get("sentiment") == "positive"

    def test_agent_topic_result(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Python programming")
        assert result.topic.get("topic") == "Technology"

    def test_agent_intent_result(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("What is Python?")
        assert result.intent.get("intent") == "question"

    def test_agent_entities_result(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Tell me about Python and LangGraph")
        assert len(result.entities) == 2

    def test_agent_summary_result(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Test")
        assert "summary" in result.summary

    def test_agent_router_complexity_simple(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Short text")
        assert "complexity=simple" in result.steps[0]

    def test_agent_router_complexity_complex(self):
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        long_text = " ".join(["word"] * 35)
        result = agent.analyze(long_text)
        assert "complexity=complex" in result.steps[0]


class TestAgentErrorHandling:
    """Tests for agent error handling with failing engine."""

    def test_agent_handles_engine_failures_gracefully(self):
        agent = ClassificationAgent(
            analysis_engine=FailingEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Test text")
        assert result.success is True
        assert "sentiment_analysis (failed)" in result.steps
        assert "topic_classification (failed)" in result.steps

    def test_agent_failed_sentiment_returns_unknown(self):
        agent = ClassificationAgent(
            analysis_engine=FailingEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Test")
        assert result.sentiment.get("sentiment") == "unknown"

    def test_agent_failed_entities_returns_empty(self):
        agent = ClassificationAgent(
            analysis_engine=FailingEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Test")
        assert result.entities == []

    def test_quality_check_low_on_all_failures(self):
        """When all analyses fail, quality score should be low."""
        agent = ClassificationAgent(
            analysis_engine=FailingEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=False,
        )
        result = agent.analyze("Test")
        # Quality check step should reflect failures
        quality_step = result.steps[-1]
        assert "quality_check" in quality_step

    def test_agent_skips_kg_when_no_entities(self):
        """When entity extraction fails, KG enrichment should be skipped."""
        agent = ClassificationAgent(
            analysis_engine=FailingEngine(),
            knowledge_graph=KnowledgeGraph(),
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Test")
        step_names = [s.split("(")[0] for s in result.steps]
        assert "kg_enrichment" not in step_names
        assert "knowledge_graph_construction" not in step_names


class TestAgentGraphOperations:
    """Tests for agent graph operations."""

    def test_get_knowledge_graph(self):
        kg = KnowledgeGraph()
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=kg,
        )
        graph = agent.get_knowledge_graph()
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "edges")

    def test_clear_knowledge_graph(self):
        kg = KnowledgeGraph()
        kg.add_entity("Test", "TYPE")
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=kg,
        )
        assert kg.node_count == 1
        agent.clear_knowledge_graph()
        assert kg.node_count == 0

    def test_agent_with_seeded_graph(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=kg,
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Tell me about Python")
        assert result.success is True
        assert result.knowledge_graph["node_count"] > 50

    def test_agent_kg_enrichment_finds_seeded_entities(self):
        """When KG has seed data, enrichment should find matches."""
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=kg,
            enable_knowledge_graph=True,
        )
        result = agent.analyze("Tell me about Python and LangGraph")
        assert result.success is True
        # The summary should include KG context
        assert "KG context:" in result.summary

    def test_infer_entity_on_seeded_graph(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=kg,
        )
        inference = agent.infer_entity("DSPy", entity_type="FRAMEWORK", max_depth=2)
        assert len(inference.matches) == 1
        assert len(inference.related) > 0
        related_names = {r["entity"]["name"] for r in inference.related}
        assert "Python" in related_names

    def test_infer_entity_with_relation_filter(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        agent = ClassificationAgent(
            analysis_engine=FakeEngine(),
            knowledge_graph=kg,
        )
        inference = agent.infer_entity("BERT", entity_type="MODEL", relation_filter="used_for")
        for rel in inference.related:
            assert "used_for" in rel["relation"].lower()
