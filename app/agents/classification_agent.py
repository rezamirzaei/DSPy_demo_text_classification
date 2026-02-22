"""Classification Agent — LangGraph multi-step analysis pipeline.

Uses LangGraph's StateGraph with conditional routing, knowledge-graph
enrichment, and quality validation.

Graph topology
--------------
         ┌──────────┐
         │  router   │
         └────┬──────┘
              │
         sentiment
              │
           topic
              │
          intent
              │
         entities
              │
     ┌────────┴────────┐
     │ (if KG enabled) │
     ▼                 ▼
 kg_enrich         (skip)
     │                 │
  kg_build             │
     └────────┬────────┘
              │
          summarise
              │
        quality_check
              │
             END
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.models.schemas import AgentResponse
from app.services import KnowledgeGraph, TextAnalysisEngine, build_analysis_engine

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────

class AgentState(TypedDict):
    """State that flows through every LangGraph node."""

    text: str
    include_knowledge_graph: bool

    # Analysis results
    sentiment: Dict[str, Any]
    topic: Dict[str, Any]
    intent: Dict[str, Any]
    entities: List[Dict[str, str]]

    # Knowledge-graph outputs
    knowledge_graph: Dict[str, Any]
    kg_enrichment: Dict[str, Any]

    # Final
    summary: str
    quality: Dict[str, Any]
    steps: List[str]
    error: Optional[str]


# ── Agent ────────────────────────────────────────────────

class ClassificationAgent:
    """LangGraph-based agent for comprehensive text analysis.

    The graph is compiled once at construction and reused for every
    ``analyze()`` call so there is no per-request overhead.
    """

    def __init__(
        self,
        analysis_engine: TextAnalysisEngine | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        enable_knowledge_graph: bool = True,
    ) -> None:
        self._engine = analysis_engine or build_analysis_engine(enable_dspy=False)
        self._kg = knowledge_graph or KnowledgeGraph()
        self._default_include_kg = enable_knowledge_graph
        self._compiled = self._build_graph()

    # ── graph construction ───────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        # Register nodes
        g.add_node("router", self._node_router)
        g.add_node("sentiment", self._node_sentiment)
        g.add_node("topic", self._node_topic)
        g.add_node("intent", self._node_intent)
        g.add_node("entities", self._node_entities)
        g.add_node("kg_enrich", self._node_kg_enrich)
        g.add_node("kg_build", self._node_kg_build)
        g.add_node("summarise", self._node_summarise)
        g.add_node("quality_check", self._node_quality_check)

        # Entry
        g.set_entry_point("router")

        # Router → sequential analyses
        g.add_edge("router", "sentiment")
        g.add_edge("sentiment", "topic")
        g.add_edge("topic", "intent")
        g.add_edge("intent", "entities")

        # After entities, decide whether to do KG work
        g.add_conditional_edges(
            "entities",
            self._route_after_entities,
            {
                "kg_enrich": "kg_enrich",
                "summarise": "summarise",
            },
        )

        # KG enrichment → KG build → summarise
        g.add_edge("kg_enrich", "kg_build")
        g.add_edge("kg_build", "summarise")

        # Summarise → quality check → END
        g.add_edge("summarise", "quality_check")
        g.add_edge("quality_check", END)

        return g.compile()

    # ── routing logic ────────────────────────────────────

    @staticmethod
    def _route_after_entities(state: AgentState) -> str:
        """Decide whether KG enrichment is worthwhile."""
        if not state.get("include_knowledge_graph", True):
            return "summarise"
        entities = state.get("entities", [])
        if not entities:
            return "summarise"
        return "kg_enrich"

    # ── node implementations ─────────────────────────────

    def _node_router(self, state: AgentState) -> dict[str, Any]:
        """Initial routing — inspect text and prepare metadata."""
        text = state["text"]
        word_count = len(text.split())
        complexity = (
            "complex" if word_count > 30
            else "moderate" if word_count > 15
            else "simple"
        )
        return {
            "steps": state.get("steps", []) + [
                f"router(words={word_count}, complexity={complexity})"
            ],
        }

    def _node_sentiment(self, state: AgentState) -> dict[str, Any]:
        try:
            result = self._engine.classify_sentiment(state["text"]).data
            return {
                "sentiment": result,
                "steps": state.get("steps", []) + ["sentiment_analysis"],
            }
        except Exception as exc:
            logger.warning("Sentiment analysis failed: %s", exc)
            return {
                "sentiment": {
                    "sentiment": "unknown",
                    "confidence": "low",
                    "reasoning": str(exc),
                },
                "steps": state.get("steps", []) + ["sentiment_analysis (failed)"],
            }

    def _node_topic(self, state: AgentState) -> dict[str, Any]:
        try:
            result = self._engine.classify_topic(state["text"]).data
            return {
                "topic": result,
                "steps": state.get("steps", []) + ["topic_classification"],
            }
        except Exception as exc:
            logger.warning("Topic classification failed: %s", exc)
            return {
                "topic": {
                    "topic": "unknown",
                    "confidence": "low",
                    "reasoning": str(exc),
                },
                "steps": state.get("steps", []) + ["topic_classification (failed)"],
            }

    def _node_intent(self, state: AgentState) -> dict[str, Any]:
        try:
            result = self._engine.classify_intent(state["text"]).data
            return {
                "intent": result,
                "steps": state.get("steps", []) + ["intent_detection"],
            }
        except Exception as exc:
            logger.warning("Intent detection failed: %s", exc)
            return {
                "intent": {
                    "intent": "unknown",
                    "confidence": "low",
                    "reasoning": str(exc),
                },
                "steps": state.get("steps", []) + ["intent_detection (failed)"],
            }

    def _node_entities(self, state: AgentState) -> dict[str, Any]:
        try:
            entities = self._engine.extract_entities(state["text"])
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

    def _node_kg_enrich(self, state: AgentState) -> dict[str, Any]:
        """Query the existing knowledge graph for related context.

        For each extracted entity we look up neighbours in the KG so that
        the downstream summary can reference real-world knowledge.
        """
        try:
            entities = state.get("entities", [])
            enrichment: dict[str, Any] = {
                "entity_matches": [],
                "related_facts": [],
                "predicted_links": [],
            }

            for entity_dict in entities:
                name = entity_dict.get("text", entity_dict.get("name", "")).strip()
                etype = entity_dict.get("type", "").strip().upper() or None
                if not name:
                    continue

                inference = self._kg.infer_for_entity(
                    entity_name=name,
                    entity_type=etype,
                    max_depth=2,
                )

                # If typed search found nothing, try name-only
                if not inference["matches"] and etype:
                    inference = self._kg.infer_for_entity(
                        entity_name=name,
                        entity_type=None,
                        max_depth=2,
                    )

                if inference["matches"]:
                    enrichment["entity_matches"].extend(inference["matches"])
                    for rel in inference.get("related", [])[:5]:
                        enrichment["related_facts"].append({
                            "entity": name,
                            "related_to": rel["entity"]["name"],
                            "relation": rel["relation"],
                            "weight": rel["weight"],
                        })
                    for pred in inference.get("predicted_links", [])[:3]:
                        enrichment["predicted_links"].append({
                            "from": name,
                            "to": pred["target"]["name"],
                            "via": pred["via"]["name"],
                            "confidence": pred["confidence"],
                        })

            enrichment["entities_found_in_kg"] = len(enrichment["entity_matches"])
            enrichment["total_related_facts"] = len(enrichment["related_facts"])

            return {
                "kg_enrichment": enrichment,
                "steps": state.get("steps", []) + [
                    f"kg_enrichment(found={enrichment['entities_found_in_kg']})"
                ],
            }
        except Exception as exc:
            logger.warning("KG enrichment failed: %s", exc)
            return {
                "kg_enrichment": {"error": str(exc)},
                "steps": state.get("steps", []) + ["kg_enrichment (failed)"],
            }

    def _node_kg_build(self, state: AgentState) -> dict[str, Any]:
        """Add extracted entities and analysis results to the knowledge graph."""
        try:
            entities = state.get("entities", [])
            self._kg.build_from_entities(entities, source_text=state["text"])

            sentiment_label = str(
                state.get("sentiment", {}).get("sentiment", "unknown")
            )
            topic_label = str(
                state.get("topic", {}).get("topic", "unknown")
            )

            sentiment_entity = self._kg.add_entity(sentiment_label, "SENTIMENT")
            topic_entity = self._kg.add_entity(topic_label, "TOPIC")

            for entity_dict in entities:
                name = entity_dict.get("text", entity_dict.get("name", "")).strip()
                etype = entity_dict.get("type", "CONCEPT").strip().upper()
                if not name:
                    continue
                entity = self._kg.add_entity(name, etype)
                self._kg.add_relationship(
                    entity, sentiment_entity, "has_sentiment",
                    context=state["text"][:100],
                )
                self._kg.add_relationship(
                    entity, topic_entity, "belongs_to_topic",
                    context=state["text"][:100],
                )

            entity_objects = [
                self._kg.get_entity(
                    e.get("text", e.get("name", "")).strip(),
                    e.get("type", "CONCEPT").strip().upper(),
                )
                for e in entities
            ]
            compact = [e for e in entity_objects if e is not None]
            graph_data = self._kg.export_graph()
            graph_data["inferences"] = self._kg.infer_connections(compact)

            return {
                "knowledge_graph": graph_data,
                "steps": state.get("steps", []) + ["knowledge_graph_construction"],
            }
        except Exception as exc:
            logger.exception("Knowledge graph construction failed: %s", exc)
            return {
                "knowledge_graph": {"error": str(exc)},
                "steps": state.get("steps", []) + [
                    "knowledge_graph_construction (failed)"
                ],
            }

    def _node_summarise(self, state: AgentState) -> dict[str, Any]:
        """Generate a human-readable summary of all analysis results."""
        try:
            summary = self._engine.summarize(
                text=state["text"],
                sentiment=state.get("sentiment", {}),
                topic=state.get("topic", {}),
                intent=state.get("intent", {}),
                entities=state.get("entities", []),
            )

            # Append KG enrichment info if available
            enrichment = state.get("kg_enrichment", {})
            facts = enrichment.get("related_facts", [])
            if facts:
                fact_strs = [
                    f"{f['entity']} {f['relation']} {f['related_to']}"
                    for f in facts[:3]
                ]
                summary += " KG context: " + "; ".join(fact_strs) + "."

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

    def _node_quality_check(self, state: AgentState) -> dict[str, Any]:
        """Validate the analysis quality and flag issues."""
        issues: list[str] = []
        score = 100

        sentiment = state.get("sentiment", {})
        topic = state.get("topic", {})
        intent = state.get("intent", {})
        entities = state.get("entities", [])

        if sentiment.get("sentiment") == "unknown":
            issues.append("sentiment_unavailable")
            score -= 20
        elif sentiment.get("confidence") == "low":
            issues.append("low_sentiment_confidence")
            score -= 10

        if topic.get("topic") == "unknown":
            issues.append("topic_unavailable")
            score -= 20
        elif topic.get("confidence") == "low":
            issues.append("low_topic_confidence")
            score -= 10

        if intent.get("intent") == "unknown":
            issues.append("intent_unavailable")
            score -= 20
        elif intent.get("confidence") == "low":
            issues.append("low_intent_confidence")
            score -= 10

        if not entities:
            issues.append("no_entities_found")
            score -= 15

        enrichment = state.get("kg_enrichment", {})
        if enrichment.get("entities_found_in_kg", 0) > 0:
            score += 5

        quality = {
            "score": max(0, min(100, score)),
            "grade": (
                "A" if score >= 90
                else "B" if score >= 75
                else "C" if score >= 60
                else "D" if score >= 40
                else "F"
            ),
            "issues": issues,
            "entities_in_kg": enrichment.get("entities_found_in_kg", 0),
        }

        return {
            "quality": quality,
            "steps": state.get("steps", []) + [
                f"quality_check(score={quality['score']}, grade={quality['grade']})"
            ],
        }

    # ── public API ───────────────────────────────────────

    def analyze(
        self,
        text: str,
        include_knowledge_graph: bool | None = None,
    ) -> AgentResponse:
        """Run the full analysis pipeline via LangGraph."""
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
            "kg_enrichment": {},
            "summary": "",
            "quality": {},
            "steps": [],
            "error": None,
        }

        try:
            result = self._compiled.invoke(initial_state)
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
        """Run entity-centric inference on the knowledge graph."""
        return self._kg.infer_for_entity(
            entity_name=name,
            entity_type=entity_type,
            max_depth=max_depth,
            relation_filter=relation_filter,
        )

    def get_knowledge_graph(self) -> dict[str, Any]:
        """Export the current knowledge graph state."""
        return self._kg.export_graph()

    def clear_knowledge_graph(self) -> None:
        """Clear all entities and relationships."""
        self._kg.clear()
