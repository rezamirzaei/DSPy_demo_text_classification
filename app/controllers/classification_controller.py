"""Classification Controller (Controller Layer)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.domain.enums import ClassifierType
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationRequest,
    ClassificationResponse,
    GraphInferenceRequest,
    GraphInferenceResponse,
    KnowledgeGraphExport,
    ReseedResponse,
)
from app.services import KnowledgeGraph, build_analysis_engine
from app.services.dspy_service import DSPyService

logger = logging.getLogger(__name__)


class ClassificationController:
    """Central controller managing classification pipeline.

    Orchestrates DSPy initialization, analysis engine construction,
    knowledge-graph lifecycle, and (optionally) DSPy prompt optimization
    via BootstrapFewShot.
    """

    def __init__(self, settings: Any = None) -> None:
        if settings is None:
            from config import get_settings

            settings = get_settings()

        self._settings = settings
        self._dspy_service = DSPyService(settings)
        self._analysis_engine = build_analysis_engine(enable_dspy=False, settings=settings)
        self._knowledge_graph = KnowledgeGraph(
            persist_path=settings.graph_data_path,
            auto_persist=True,
        )
        # Seed graph with real-world AI/ML data if empty and no JSON exists
        if self._knowledge_graph.node_count == 0:
            self._knowledge_graph.seed_default_graph()
        self._agent: Any = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        # The application remains functional with deterministic fallback.
        return self._initialized or self.provider == "rule_based"

    @property
    def model(self) -> str:
        return self._dspy_service.model_name

    @property
    def provider(self) -> str:
        return self._dspy_service.provider

    @property
    def backend_ready(self) -> bool:
        return self._dspy_service.is_initialized

    def initialize(self) -> bool:
        """Initialize primary backend and build hybrid engine.

        When an LLM-backed provider is active (not rule_based), the
        controller also attempts DSPy prompt optimization via
        ``BootstrapFewShot`` to improve classification quality.
        """
        self._initialized = self._dspy_service.initialize()
        use_dspy = self._dspy_service.is_initialized and self._dspy_service.provider != "rule_based"
        self._analysis_engine = build_analysis_engine(enable_dspy=use_dspy, settings=self._settings)

        # Run DSPy optimizer when an LLM backend is available
        if use_dspy:
            self._run_optimizer()

        return self.is_initialized

    def _run_optimizer(self) -> None:
        """Attempt to optimize DSPy classifiers with BootstrapFewShot."""
        try:
            from app.models.optimizer import DSPyOptimizer

            cache_dir = self._settings.data_dir / "dspy_optimized"
            optimizer = DSPyOptimizer(cache_dir=cache_dir)
            optimized = optimizer.optimize_all_classifiers()
            logger.info(
                "DSPy optimizer completed: %d modules optimized",
                len(optimized),
            )
        except Exception as exc:
            logger.warning("DSPy optimization skipped: %s", exc)

    def get_available_classifiers(self) -> List[str]:
        """Return registered classifier type names."""
        return [classifier.value for classifier in ClassifierType if classifier != ClassifierType.AGENT]

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """Run a single classification."""
        try:
            payload = self._classify_payload(request)
            return ClassificationResponse(
                text=request.text,
                classifier_type=request.classifier_type.value,
                result=payload,
                success=True,
            )
        except Exception as exc:
            logger.exception("Classification failed: %s", exc)
            return ClassificationResponse(
                text=request.text,
                classifier_type=request.classifier_type.value,
                result={},
                success=False,
                error=str(exc),
            )

    def _classify_payload(self, request: ClassificationRequest) -> Dict[str, Any]:
        if request.classifier_type == ClassifierType.SENTIMENT:
            return self._analysis_engine.classify_sentiment(request.text).data

        if request.classifier_type == ClassifierType.TOPIC:
            return self._analysis_engine.classify_topic(
                request.text,
                categories=request.categories,
            ).data

        if request.classifier_type == ClassifierType.INTENT:
            return self._analysis_engine.classify_intent(
                request.text,
                intents=request.intents,
            ).data

        if request.classifier_type == ClassifierType.MULTI_LABEL:
            return self._analysis_engine.classify_multi_label(
                request.text,
                labels=request.labels,
            ).data

        if request.classifier_type == ClassifierType.ENTITY:
            entities = self._analysis_engine.extract_entities(request.text)
            return {
                "entities": entities,
                "count": len(entities),
            }

        raise ValueError(f"Unsupported classifier type: {request.classifier_type.value}")

    def batch_classify(
        self,
        request: BatchClassificationRequest,
    ) -> BatchClassificationResponse:
        """Classify a batch of texts."""
        results: List[ClassificationResponse] = []

        for text in request.texts:
            single = ClassificationRequest(
                text=text,
                classifier_type=request.classifier_type,
                categories=request.categories,
            )
            results.append(self.classify(single))

        successful = sum(1 for response in results if response.success)
        return BatchClassificationResponse(
            results=results,
            total=len(results),
            successful=successful,
            failed=len(results) - successful,
        )

    def run_agent(self, request: AgentRequest) -> AgentResponse:
        """Run the LangGraph agent pipeline."""
        try:
            if self._agent is None:
                from app.agents.classification_agent import ClassificationAgent

                self._agent = ClassificationAgent(
                    analysis_engine=self._analysis_engine,
                    knowledge_graph=self._knowledge_graph,
                    enable_knowledge_graph=request.enable_knowledge_graph,
                )

            result: AgentResponse = self._agent.analyze(
                request.text,
                include_knowledge_graph=request.enable_knowledge_graph,
            )
            return result
        except Exception as exc:
            logger.exception("Agent analysis failed: %s", exc)
            return AgentResponse(text=request.text, success=False, error=str(exc))

    def graph_infer(self, request: GraphInferenceRequest) -> GraphInferenceResponse:
        """Run graph-centric inference for an entity."""
        return self._knowledge_graph.infer_for_entity(
            entity_name=request.entity,
            entity_type=request.entity_type,
            max_depth=request.max_depth,
            relation_filter=request.relation_filter,
        )

    def get_knowledge_graph(self) -> KnowledgeGraphExport:
        """Export the current knowledge graph."""
        if self._agent is not None:
            result: KnowledgeGraphExport = self._agent.get_knowledge_graph()
            return result
        return self._knowledge_graph.export_graph()

    def reseed_knowledge_graph(self) -> ReseedResponse:
        """Clear and re-seed the knowledge graph with curated data."""
        self._knowledge_graph.clear()
        self._knowledge_graph.seed_default_graph()
        return ReseedResponse(
            message="Knowledge graph seeded successfully",
            node_count=self._knowledge_graph.node_count,
            edge_count=self._knowledge_graph.edge_count,
        )

