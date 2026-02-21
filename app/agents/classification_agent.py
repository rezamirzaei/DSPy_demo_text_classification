"""Classification Agent - LangGraph multi-step analysis pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.models.schemas import AgentResponse
from app.services import KnowledgeGraph, TextAnalysisEngine, build_analysis_engine

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed through the LangGraph nodes."""

    text: str
    include_knowledge_graph: bool
    sentiment: Dict[str, Any]
    topic: Dict[str, Any]
    intent: Dict[str, Any]
    entities: List[Dict[str, str]]
    knowledge_graph: Dict[str, Any]
    summary: str
    steps: List[str]
    error: Optional[str]


class ClassificationAgent:
    """LangGraph-based agent for comprehensive text analysis."""

    def __init__(
        self,
        analysis_engine: TextAnalysisEngine | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        enable_knowledge_graph: bool = True,
    ) -> None:
        self._analysis_engine = analysis_engine or build_analysis_engine(enable_dspy=False)
        self._kg = knowledge_graph or KnowledgeGraph()
        self._default_include_kg = enable_knowledge_graph
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("sentiment", self._do_sentiment)
        graph.add_node("topic", self._do_topic)
        graph.add_node("intent", self._do_intent)
        graph.add_node("entities", self._do_entities)
        graph.add_node("kg", self._do_kg)
        graph.add_node("summary", self._do_summary)

        graph.set_entry_point("sentiment")
        graph.add_edge("sentiment", "topic")
        graph.add_edge("topic", "intent")
        graph.add_edge("intent", "entities")
        graph.add_conditional_edges(
            "entities",
            self._route_after_entities,
            {
                "kg": "kg",
                "summary": "summary",
            },
        )
        graph.add_edge("kg", "summary")
        graph.add_edge("summary", END)

        return graph.compile()

    def _route_after_entities(self, state: AgentState) -> str:
        if state.get("include_knowledge_graph", self._default_include_kg):
            return "kg"
        return "summary"

    def _do_sentiment(self, state: AgentState) -> dict[str, Any]:
        try:
            result = self._analysis_engine.classify_sentiment(state["text"]).data
            return {
                "sentiment": result,
                "steps": state.get("steps", []) + ["sentiment_analysis"],
            }
        except Exception as exc:
            return {
                "sentiment": {
                    "sentiment": "unknown",
                    "confidence": "low",
                    "reasoning": str(exc),
                },
                "steps": state.get("steps", []) + ["sentiment_analysis (failed)"],
            }

    def _do_topic(self, state: AgentState) -> dict[str, Any]:
        try:
            result = self._analysis_engine.classify_topic(state["text"]).data
            return {
                "topic": result,
                "steps": state.get("steps", []) + ["topic_classification"],
            }
        except Exception as exc:
            return {
                "topic": {
                    "topic": "unknown",
                    "confidence": "low",
                    "reasoning": str(exc),
                },
                "steps": state.get("steps", []) + ["topic_classification (failed)"],
            }

    def _do_intent(self, state: AgentState) -> dict[str, Any]:
        try:
            result = self._analysis_engine.classify_intent(state["text"]).data
            return {
                "intent": result,
                "steps": state.get("steps", []) + ["intent_detection"],
            }
        except Exception as exc:
            return {
                "intent": {
                    "intent": "unknown",
                    "confidence": "low",
                    "reasoning": str(exc),
                },
                "steps": state.get("steps", []) + ["intent_detection (failed)"],
            }

    def _do_entities(self, state: AgentState) -> dict[str, Any]:
        try:
            entities = self._analysis_engine.extract_entities(state["text"])
            return {
                "entities": entities,
                "steps": state.get("steps", []) + ["entity_extraction"],
            }
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return {
                "entities": [],
                "steps": state.get("steps", []) + ["entity_extraction (failed)"],
            }

    def _do_kg(self, state: AgentState) -> dict[str, Any]:
        try:
            entities = state.get("entities", [])
            self._kg.build_from_entities(entities, source_text=state["text"])

            sentiment = str(state.get("sentiment", {}).get("sentiment", "unknown"))
            topic = str(state.get("topic", {}).get("topic", "unknown"))
            sentiment_entity = self._kg.add_entity(sentiment, "SENTIMENT")
            topic_entity = self._kg.add_entity(topic, "TOPIC")

            for entity_dict in entities:
                name = entity_dict.get("text", entity_dict.get("name", "")).strip()
                entity_type = entity_dict.get("type", "CONCEPT").strip().upper()
                if not name:
                    continue
                entity = self._kg.add_entity(name, entity_type)
                self._kg.add_relationship(entity, sentiment_entity, "has_sentiment")
                self._kg.add_relationship(entity, topic_entity, "belongs_to_topic")

            entity_objects = [
                self._kg.get_entity(
                    entity_dict.get("text", entity_dict.get("name", "")).strip(),
                    entity_dict.get("type", "CONCEPT").strip().upper(),
                )
                for entity_dict in entities
            ]
            compact_entities = [entity for entity in entity_objects if entity is not None]
            graph_data = self._kg.export_graph()
            graph_data["inferences"] = self._kg.infer_connections(compact_entities)

            return {
                "knowledge_graph": graph_data,
                "steps": state.get("steps", []) + ["knowledge_graph_construction"],
            }
        except Exception as exc:
            logger.exception("Knowledge graph construction failed: %s", exc)
            return {
                "knowledge_graph": {"error": str(exc)},
                "steps": state.get("steps", []) + ["knowledge_graph_construction (failed)"],
            }

    def _do_summary(self, state: AgentState) -> dict[str, Any]:
        try:
            summary = self._analysis_engine.summarize(
                text=state["text"],
                sentiment=state.get("sentiment", {}),
                topic=state.get("topic", {}),
                intent=state.get("intent", {}),
                entities=state.get("entities", []),
            )
            return {
                "summary": summary,
                "steps": state.get("steps", []) + ["summary_generation"],
            }
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            return {
                "summary": "Analysis completed with partial results.",
                "steps": state.get("steps", []) + ["summary_generation (failed)"],
            }

    def analyze(self, text: str, include_knowledge_graph: bool | None = None) -> AgentResponse:
        """Run the full analysis pipeline."""
        initial_state: AgentState = {
            "text": text,
            "include_knowledge_graph": (
                self._default_include_kg
                if include_knowledge_graph is None
                else include_knowledge_graph
            ),
            "sentiment": {},
            "topic": {},
            "intent": {},
            "entities": [],
            "knowledge_graph": {},
            "summary": "",
            "steps": [],
            "error": None,
        }

        try:
            result = self._graph.invoke(initial_state)
            return AgentResponse(
                text=text,
                sentiment=result.get("sentiment", {}),
                topic=result.get("topic", {}),
                intent=result.get("intent", {}),
                entities=result.get("entities", []),
                knowledge_graph=result.get("knowledge_graph", {}),
                summary=result.get("summary", ""),
                success=True,
                steps=result.get("steps", []),
            )
        except Exception as exc:
            logger.exception("Agent execution failed: %s", exc)
            return AgentResponse(
                text=text,
                success=False,
                error=str(exc),
                steps=["failed"],
            )

    def infer_entity(
        self,
        name: str,
        entity_type: str | None = None,
        max_depth: int = 2,
        relation_filter: str | None = None,
    ) -> dict[str, Any]:
        return self._kg.infer_for_entity(
            entity_name=name,
            entity_type=entity_type,
            max_depth=max_depth,
            relation_filter=relation_filter,
        )

    def get_knowledge_graph(self) -> dict[str, Any]:
        return self._kg.export_graph()

    def clear_knowledge_graph(self) -> None:
        self._kg.clear()
