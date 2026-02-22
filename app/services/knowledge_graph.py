"""Knowledge Graph Service - Entity relationship inference and persistence."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Set

from app.models.schemas import GraphInferenceResponse, KnowledgeGraphExport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entity:
    """An entity node in the knowledge graph."""

    name: str
    entity_type: str

    def __hash__(self) -> int:
        return hash((self.name.lower(), self.entity_type.upper()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return (
            self.name.lower() == other.name.lower()
            and self.entity_type.upper() == other.entity_type.upper()
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.entity_type}


@dataclass
class Relationship:
    """A directed edge between two entities."""

    source: Entity
    target: Entity
    relation_type: str
    weight: float = 1.0
    context: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation": self.relation_type,
            "weight": self.weight,
            "context": self.context,
        }


class KnowledgeGraph:
    """In-memory knowledge graph for entity relationship inference.

    Optional persistence can be enabled by providing a JSON file path.
    """

    def __init__(
        self,
        persist_path: Optional[str | Path] = None,
        auto_persist: bool = False,
    ) -> None:
        self._entities: Dict[str, Entity] = {}
        self._adjacency: Dict[str, List[Relationship]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[Relationship]] = defaultdict(list)
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)
        self._edge_index: Dict[str, Relationship] = {}
        self._lock = RLock()

        self._persist_path = Path(persist_path) if persist_path else None
        self._auto_persist = auto_persist
        if self._persist_path and self._persist_path.exists():
            self.load()

    @staticmethod
    def _entity_key(entity: Entity) -> str:
        return f"{entity.entity_type.upper()}:{entity.name.lower()}"

    @staticmethod
    def _edge_key(src_key: str, tgt_key: str, relation_type: str) -> str:
        return f"{src_key}->{relation_type.lower()}->{tgt_key}"

    def _persist_if_needed(self) -> None:
        if self._auto_persist:
            self.save()

    def _find_by_name(self, name: str) -> list[Entity]:
        target = name.strip().lower()
        return [e for e in self._entities.values() if e.name.lower() == target]

    def add_entity(self, name: str, entity_type: str) -> Entity:
        with self._lock:
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError("Entity name must not be empty")

            normalized_type = entity_type.upper().strip() or "CONCEPT"
            entity = Entity(name=normalized_name, entity_type=normalized_type)
            key = self._entity_key(entity)
            if key not in self._entities:
                self._entities[key] = entity
                self._entity_index[entity.entity_type].add(key)
                self._persist_if_needed()
            return self._entities[key]

    def add_relationship(
        self,
        source: Entity,
        target: Entity,
        relation_type: str,
        weight: float = 1.0,
        context: str = "",
    ) -> Relationship:
        with self._lock:
            src_key = self._entity_key(source)
            tgt_key = self._entity_key(target)
            if src_key not in self._entities:
                self.add_entity(source.name, source.entity_type)
            if tgt_key not in self._entities:
                self.add_entity(target.name, target.entity_type)

            edge_key = self._edge_key(src_key, tgt_key, relation_type)
            if edge_key in self._edge_index:
                existing = self._edge_index[edge_key]
                existing.weight = max(existing.weight, weight)
                if context:
                    existing.context = context
                self._persist_if_needed()
                return existing

            rel = Relationship(
                source=self._entities[src_key],
                target=self._entities[tgt_key],
                relation_type=relation_type,
                weight=weight,
                context=context,
            )
            self._adjacency[src_key].append(rel)
            self._reverse_adjacency[tgt_key].append(rel)
            self._edge_index[edge_key] = rel
            self._persist_if_needed()
            return rel

    def get_entity(self, name: str, entity_type: str) -> Entity | None:
        key = self._entity_key(Entity(name=name, entity_type=entity_type))
        return self._entities.get(key)

    def get_neighbors(self, entity: Entity) -> list[tuple[Entity, str, float]]:
        key = self._entity_key(entity)
        return [
            (relationship.target, relationship.relation_type, relationship.weight)
            for relationship in self._adjacency.get(key, [])
        ]

    def get_incoming(self, entity: Entity) -> list[tuple[Entity, str, float]]:
        key = self._entity_key(entity)
        return [
            (relationship.source, relationship.relation_type, relationship.weight)
            for relationship in self._reverse_adjacency.get(key, [])
        ]

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        keys = self._entity_index.get(entity_type.upper(), set())
        return [self._entities[key] for key in keys if key in self._entities]

    def find_related(self, entity: Entity, max_depth: int = 2) -> list[dict[str, object]]:
        key = self._entity_key(entity)
        if key not in self._entities:
            return []

        visited = {key}
        queue = deque([(key, 0)])
        related: list[dict[str, object]] = []

        while queue:
            current_key, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for rel in self._adjacency.get(current_key, []):
                target_key = self._entity_key(rel.target)
                if target_key in visited:
                    continue
                visited.add(target_key)
                related.append(
                    {
                        "entity": rel.target.to_dict(),
                        "relation": rel.relation_type,
                        "depth": depth + 1,
                        "weight": rel.weight,
                        "direction": "outgoing",
                    }
                )
                queue.append((target_key, depth + 1))

            for rel in self._reverse_adjacency.get(current_key, []):
                source_key = self._entity_key(rel.source)
                if source_key in visited:
                    continue
                visited.add(source_key)
                related.append(
                    {
                        "entity": rel.source.to_dict(),
                        "relation": f"inverse_{rel.relation_type}",
                        "depth": depth + 1,
                        "weight": rel.weight,
                        "direction": "incoming",
                    }
                )
                queue.append((source_key, depth + 1))

        return related

    def infer_connections(self, entities: Iterable[Entity]) -> list[dict[str, object]]:
        """Infer latent links by shared outgoing neighbors."""
        inferences: list[dict[str, object]] = []
        entity_keys = [
            self._entity_key(entity)
            for entity in entities
            if self._entity_key(entity) in self._entities
        ]

        for index, key_a in enumerate(entity_keys):
            neighbors_a = {
                self._entity_key(relationship.target)
                for relationship in self._adjacency.get(key_a, [])
            }
            for key_b in entity_keys[index + 1:]:
                neighbors_b = {
                    self._entity_key(relationship.target)
                    for relationship in self._adjacency.get(key_b, [])
                }
                shared = neighbors_a & neighbors_b
                if not shared:
                    continue

                shared_entities = [
                    self._entities[key].to_dict() for key in shared if key in self._entities
                ]
                inferences.append(
                    {
                        "entity_a": self._entities[key_a].to_dict(),
                        "entity_b": self._entities[key_b].to_dict(),
                        "shared_connections": shared_entities,
                        "inference_strength": len(shared)
                        / max(len(neighbors_a), len(neighbors_b), 1),
                    }
                )

        return inferences

    def infer_for_entity(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        max_depth: int = 2,
        relation_filter: Optional[str] = None,
    ) -> GraphInferenceResponse:
        """Run entity-centric graph inference.

        If `entity_type` is not provided, all entities with matching names are considered.
        Returns a validated ``GraphInferenceResponse`` Pydantic model.
        """
        query_info: Dict[str, Any] = {
            "name": entity_name,
            "type": entity_type,
            "max_depth": max_depth,
            "relation_filter": relation_filter,
        }
        relation_filter_norm = relation_filter.lower().strip() if relation_filter else None
        matches: list[Entity]

        if entity_type:
            entity = self.get_entity(entity_name, entity_type)
            matches = [entity] if entity else []
        else:
            matches = self._find_by_name(entity_name)

        if not matches:
            return GraphInferenceResponse(
                query=query_info,
                matches=[],
                related=[],
                predicted_links=[],
            )

        related: list[dict[str, object]] = []
        predicted_links: list[dict[str, object]] = []

        for entity in matches:
            for item in self.find_related(entity, max_depth=max_depth):
                relation_name = str(item["relation"]).lower()
                if relation_filter_norm and relation_filter_norm not in relation_name:
                    continue
                item["source"] = entity.to_dict()
                related.append(item)

            predicted_links.extend(self._predict_links(entity))

        return GraphInferenceResponse(
            query=query_info,
            matches=[entity.to_dict() for entity in matches],
            related=related,  # type: ignore[arg-type]
            predicted_links=predicted_links,  # type: ignore[arg-type]
        )

    def _predict_links(self, entity: Entity) -> list[dict[str, object]]:
        """Infer likely links by two-hop traversal."""
        entity_key = self._entity_key(entity)
        direct_targets = {
            self._entity_key(relationship.target)
            for relationship in self._adjacency.get(entity_key, [])
        }

        predictions: list[dict[str, object]] = []
        for first_hop in self._adjacency.get(entity_key, []):
            first_hop_key = self._entity_key(first_hop.target)
            for second_hop in self._adjacency.get(first_hop_key, []):
                candidate_key = self._entity_key(second_hop.target)
                if candidate_key == entity_key or candidate_key in direct_targets:
                    continue
                score = round((first_hop.weight + second_hop.weight) / 2.0, 3)
                predictions.append(
                    {
                        "source": entity.to_dict(),
                        "target": second_hop.target.to_dict(),
                        "via": first_hop.target.to_dict(),
                        "suggested_relation": "possibly_related",
                        "confidence": score,
                    }
                )

        # Deduplicate by target name/type, keep highest score
        dedup: dict[str, dict[str, object]] = {}
        for prediction in predictions:
            target_dict: dict[str, str] = prediction["target"]  # type: ignore[assignment]
            key = f"{target_dict['type']}:{target_dict['name'].lower()}"
            if key not in dedup or prediction["confidence"] > dedup[key]["confidence"]:  # type: ignore[operator]
                dedup[key] = prediction
        return list(dedup.values())

    def build_from_entities(
        self,
        entity_dicts: list[dict[str, str]],
        source_text: str = "",
    ) -> None:
        entities: list[Entity] = []
        for entity_dict in entity_dicts:
            name = entity_dict.get("text", entity_dict.get("name", "")).strip()
            entity_type = entity_dict.get("type", "CONCEPT").strip().upper()
            if not name:
                continue
            entity = self.add_entity(name, entity_type)
            entities.append(entity)

        for index, first_entity in enumerate(entities):
            for second_entity in entities[index + 1:]:
                context = source_text[:200] if source_text else ""
                self.add_relationship(
                    first_entity,
                    second_entity,
                    "co_occurs_with",
                    1.0,
                    context,
                )

        self._persist_if_needed()

    def export_graph(self) -> KnowledgeGraphExport:
        """Export the full graph as a validated Pydantic model."""
        nodes = [entity.to_dict() for entity in self._entities.values()]
        edges: list[dict[str, object]] = []
        for relationships in self._adjacency.values():
            for relationship in relationships:
                edges.append(relationship.to_dict())

        return KnowledgeGraphExport(
            nodes=nodes,  # type: ignore[arg-type]
            edges=edges,  # type: ignore[arg-type]
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def save(self, path: Optional[str | Path] = None) -> None:
        target = Path(path) if path else self._persist_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.export_graph().model_dump(), handle, ensure_ascii=True, indent=2)

    def load(self, path: Optional[str | Path] = None) -> None:
        source = Path(path) if path else self._persist_path
        if not source or not source.exists():
            return

        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.clear()

        for node in payload.get("nodes", []):
            name = str(node.get("name", "")).strip()
            if not name:
                continue
            self.add_entity(name, str(node.get("type", "CONCEPT")))

        for edge in payload.get("edges", []):
            source_entity_data = edge.get("source", {})
            target_entity_data = edge.get("target", {})

            source_name, source_type = self._parse_legacy_entity(source_entity_data)
            target_name, target_type = self._parse_legacy_entity(target_entity_data)
            if not source_name or not target_name:
                continue

            source_entity = self.add_entity(source_name, source_type)
            target_entity = self.add_entity(target_name, target_type)
            self.add_relationship(
                source_entity,
                target_entity,
                str(edge.get("relation", "related_to")),
                float(edge.get("weight", 1.0)),
                str(edge.get("context", "")),
            )

    @staticmethod
    def _parse_legacy_entity(raw: object) -> tuple[str, str]:
        if isinstance(raw, dict):
            name = str(raw.get("name", "")).strip()
            entity_type = str(raw.get("type", "CONCEPT")).strip().upper() or "CONCEPT"
            return name, entity_type

        if isinstance(raw, str):
            value = raw.strip()
            if not value:
                return "", "CONCEPT"
            if ":" in value:
                entity_type, name = value.split(":", 1)
                return name.strip(), entity_type.strip().upper() or "CONCEPT"
            return value, "CONCEPT"

        return "", "CONCEPT"

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()
            self._adjacency.clear()
            self._reverse_adjacency.clear()
            self._entity_index.clear()
            self._edge_index.clear()
            self._persist_if_needed()

    def seed_default_graph(self) -> None:
        """Populate the graph with a curated AI/ML knowledge base.

        Covers models, frameworks, organisations, researchers, and concepts
        with rich typed relationships so inference works out-of-the-box.
        """
        if self.node_count > 0:
            logger.info("Graph already has %d nodes – skipping seed.", self.node_count)
            return

        # ── Languages ─────────────────────────────────────────
        python = self.add_entity("Python", "LANGUAGE")
        self.add_entity("JavaScript", "LANGUAGE")
        self.add_entity("TypeScript", "LANGUAGE")
        self.add_entity("Rust", "LANGUAGE")
        self.add_entity("Go", "LANGUAGE")
        self.add_entity("Java", "LANGUAGE")
        self.add_entity("SQL", "LANGUAGE")

        # ── Frameworks & Libraries ────────────────────────────
        dspy = self.add_entity("DSPy", "FRAMEWORK")
        langchain = self.add_entity("LangChain", "FRAMEWORK")
        langgraph = self.add_entity("LangGraph", "FRAMEWORK")
        flask = self.add_entity("Flask", "FRAMEWORK")
        django = self.add_entity("Django", "FRAMEWORK")
        fastapi = self.add_entity("FastAPI", "FRAMEWORK")
        self.add_entity("React", "FRAMEWORK")
        pytorch = self.add_entity("PyTorch", "FRAMEWORK")
        tensorflow = self.add_entity("TensorFlow", "FRAMEWORK")
        hf_transformers = self.add_entity("Hugging Face Transformers", "FRAMEWORK")
        sklearn = self.add_entity("scikit-learn", "FRAMEWORK")
        pandas = self.add_entity("Pandas", "LIBRARY")
        numpy = self.add_entity("NumPy", "LIBRARY")
        pydantic = self.add_entity("Pydantic", "LIBRARY")

        # ── Organisations ─────────────────────────────────────
        openai = self.add_entity("OpenAI", "ORG")
        google = self.add_entity("Google", "ORG")
        meta = self.add_entity("Meta", "ORG")
        microsoft = self.add_entity("Microsoft", "ORG")
        anthropic = self.add_entity("Anthropic", "ORG")
        stanford = self.add_entity("Stanford University", "ORG")
        deepmind = self.add_entity("DeepMind", "ORG")
        nvidia = self.add_entity("NVIDIA", "ORG")
        hugging_face = self.add_entity("Hugging Face", "ORG")

        # ── Models ────────────────────────────────────────────
        gpt4 = self.add_entity("GPT-4", "MODEL")
        gemini = self.add_entity("Gemini", "MODEL")
        claude = self.add_entity("Claude", "MODEL")
        llama3 = self.add_entity("Llama 3", "MODEL")
        self.add_entity("Mistral", "MODEL")
        bert = self.add_entity("BERT", "MODEL")
        t5 = self.add_entity("T5", "MODEL")
        whisper = self.add_entity("Whisper", "MODEL")
        phi3 = self.add_entity("Phi-3", "MODEL")

        # ── Fields ────────────────────────────────────────────
        nlp = self.add_entity("Natural Language Processing", "FIELD")
        cv = self.add_entity("Computer Vision", "FIELD")
        ml = self.add_entity("Machine Learning", "FIELD")
        dl = self.add_entity("Deep Learning", "FIELD")
        rl = self.add_entity("Reinforcement Learning", "FIELD")
        gen_ai = self.add_entity("Generative AI", "FIELD")
        ir = self.add_entity("Information Retrieval", "FIELD")
        kg_field = self.add_entity("Knowledge Graphs", "FIELD")
        ds = self.add_entity("Data Science", "FIELD")
        mlops = self.add_entity("MLOps", "FIELD")

        # ── Concepts ──────────────────────────────────────────
        transformer = self.add_entity("Transformer", "CONCEPT")
        attention = self.add_entity("Attention Mechanism", "CONCEPT")
        rag = self.add_entity("Retrieval-Augmented Generation", "CONCEPT")
        fine_tuning = self.add_entity("Fine-Tuning", "CONCEPT")
        prompt_eng = self.add_entity("Prompt Engineering", "CONCEPT")
        embeddings = self.add_entity("Embeddings", "CONCEPT")
        vector_db = self.add_entity("Vector Database", "CONCEPT")
        cot = self.add_entity("Chain-of-Thought", "CONCEPT")
        few_shot = self.add_entity("Few-Shot Learning", "CONCEPT")
        transfer_learn = self.add_entity("Transfer Learning", "CONCEPT")
        nn = self.add_entity("Neural Network", "CONCEPT")
        backprop = self.add_entity("Backpropagation", "CONCEPT")
        agents_concept = self.add_entity("Autonomous Agents", "CONCEPT")
        tool_use = self.add_entity("Tool Use", "CONCEPT")
        rlhf = self.add_entity("RLHF", "CONCEPT")

        # ── Tasks ─────────────────────────────────────────────
        text_cls = self.add_entity("Text Classification", "TASK")
        sentiment = self.add_entity("Sentiment Analysis", "TASK")
        ner = self.add_entity("Named Entity Recognition", "TASK")
        qa = self.add_entity("Question Answering", "TASK")
        summarization = self.add_entity("Summarization", "TASK")
        code_gen = self.add_entity("Code Generation", "TASK")
        intent_det = self.add_entity("Intent Detection", "TASK")

        # ── Technologies ──────────────────────────────────────
        docker = self.add_entity("Docker", "TECHNOLOGY")
        k8s = self.add_entity("Kubernetes", "TECHNOLOGY")
        chromadb = self.add_entity("ChromaDB", "TECHNOLOGY")

        # ── People ────────────────────────────────────────────
        hinton = self.add_entity("Geoffrey Hinton", "PERSON")
        lecun = self.add_entity("Yann LeCun", "PERSON")
        ng = self.add_entity("Andrew Ng", "PERSON")
        khattab = self.add_entity("Omar Khattab", "PERSON")
        altman = self.add_entity("Sam Altman", "PERSON")
        hassabis = self.add_entity("Demis Hassabis", "PERSON")

        # ── DSPy Relationships ────────────────────────────────
        self.add_relationship(dspy, python, "implemented_in", 1.0, "DSPy is a Python framework")
        self.add_relationship(dspy, stanford, "developed_by", 1.0, "DSPy from Stanford NLP")
        self.add_relationship(dspy, khattab, "created_by", 1.0, "Omar Khattab created DSPy")
        self.add_relationship(dspy, prompt_eng, "automates", 0.9, "DSPy automates prompt engineering")
        self.add_relationship(dspy, text_cls, "used_for", 0.9, "DSPy excels at text classification")
        self.add_relationship(dspy, rag, "supports", 0.85, "DSPy has built-in RAG support")
        self.add_relationship(dspy, few_shot, "optimizes", 0.9, "DSPy optimizes few-shot examples")
        self.add_relationship(dspy, cot, "implements", 0.9, "DSPy ChainOfThought module")
        self.add_relationship(dspy, qa, "used_for", 0.85, "DSPy for QA pipelines")
        self.add_relationship(dspy, langchain, "alternative_to", 0.7, "DSPy vs LangChain")

        # ── LangGraph Relationships ───────────────────────────
        self.add_relationship(langgraph, python, "implemented_in", 1.0, "LangGraph is Python")
        self.add_relationship(langgraph, langchain, "extends", 0.9, "LangGraph extends LangChain")
        self.add_relationship(langgraph, agents_concept, "enables", 0.95, "LangGraph enables agents")
        self.add_relationship(langgraph, tool_use, "enables", 0.85, "LangGraph enables tool use")

        # ── Web Frameworks ────────────────────────────────────
        self.add_relationship(flask, python, "implemented_in", 1.0, "Flask is a Python web framework")
        self.add_relationship(django, python, "implemented_in", 1.0, "Django is a Python web framework")
        self.add_relationship(fastapi, python, "implemented_in", 1.0, "FastAPI is modern Python")
        self.add_relationship(fastapi, pydantic, "depends_on", 0.95, "FastAPI uses Pydantic")

        # ── Model Relationships ───────────────────────────────
        self.add_relationship(gpt4, openai, "developed_by", 1.0, "GPT-4 by OpenAI")
        self.add_relationship(gpt4, transformer, "based_on", 1.0, "GPT-4 uses transformers")
        self.add_relationship(gpt4, gen_ai, "belongs_to", 0.95, "GPT-4 is gen AI")
        self.add_relationship(gpt4, nlp, "applies_to", 0.95, "GPT-4 is frontier NLP")
        self.add_relationship(gpt4, rlhf, "trained_with", 0.9, "GPT-4 uses RLHF")
        self.add_relationship(gpt4, code_gen, "used_for", 0.9, "GPT-4 for code generation")

        self.add_relationship(gemini, google, "developed_by", 1.0, "Gemini by Google DeepMind")
        self.add_relationship(gemini, deepmind, "developed_by", 0.9, "Gemini co-developed by DeepMind")
        self.add_relationship(gemini, transformer, "based_on", 1.0, "Gemini uses transformers")
        self.add_relationship(gemini, gen_ai, "belongs_to", 0.95, "Gemini is gen AI")

        self.add_relationship(claude, anthropic, "developed_by", 1.0, "Claude by Anthropic")
        self.add_relationship(claude, rlhf, "trained_with", 0.9, "Claude uses RLHF")

        self.add_relationship(llama3, meta, "developed_by", 1.0, "Llama 3 by Meta")
        self.add_relationship(llama3, transformer, "based_on", 1.0, "Llama 3 uses transformers")
        self.add_relationship(llama3, gen_ai, "belongs_to", 0.95, "Llama 3 is open-weight gen AI")

        self.add_relationship(bert, google, "developed_by", 1.0, "BERT by Google")
        self.add_relationship(bert, transformer, "based_on", 1.0, "BERT uses transformer encoder")
        self.add_relationship(bert, nlp, "applies_to", 1.0, "BERT revolutionized NLP")
        self.add_relationship(bert, text_cls, "used_for", 0.95, "BERT for text classification")
        self.add_relationship(bert, ner, "used_for", 0.9, "BERT for NER")
        self.add_relationship(bert, sentiment, "used_for", 0.9, "BERT for sentiment analysis")
        self.add_relationship(bert, qa, "used_for", 0.9, "BERT for QA")
        self.add_relationship(bert, transfer_learn, "demonstrates", 0.9, "BERT pioneered transfer learning")

        self.add_relationship(t5, google, "developed_by", 1.0, "T5 by Google Research")
        self.add_relationship(t5, transformer, "based_on", 1.0, "T5 encoder-decoder transformer")
        self.add_relationship(t5, summarization, "used_for", 0.9, "T5 for summarization")

        self.add_relationship(whisper, openai, "developed_by", 1.0, "Whisper by OpenAI")
        self.add_relationship(phi3, microsoft, "developed_by", 1.0, "Phi-3 by Microsoft")

        # ── Architecture Relationships ────────────────────────
        self.add_relationship(transformer, attention, "built_on", 1.0, "Transformers built on attention")
        self.add_relationship(transformer, dl, "belongs_to", 0.95, "Core DL architecture")
        self.add_relationship(transformer, nlp, "revolutionized", 0.95, "Transformers revolutionized NLP")
        self.add_relationship(attention, nn, "component_of", 0.9, "Attention is NN component")
        self.add_relationship(nn, dl, "foundation_of", 1.0, "NNs are foundation of DL")
        self.add_relationship(nn, backprop, "trained_with", 1.0, "NNs trained with backprop")

        # ── RAG / Embeddings ──────────────────────────────────
        self.add_relationship(rag, vector_db, "relies_on", 0.9, "RAG uses vector databases")
        self.add_relationship(rag, embeddings, "uses", 0.9, "RAG uses embeddings")
        self.add_relationship(rag, ir, "combines_with", 0.85, "RAG combines IR with generation")
        self.add_relationship(vector_db, chromadb, "implemented_by", 0.8, "ChromaDB is vector DB")

        # ── Field Hierarchy ───────────────────────────────────
        self.add_relationship(dl, ml, "subfield_of", 1.0, "DL is subfield of ML")
        self.add_relationship(nlp, ml, "subfield_of", 0.9, "NLP relies on ML")
        self.add_relationship(cv, ml, "subfield_of", 0.9, "CV is ML application area")
        self.add_relationship(gen_ai, dl, "subfield_of", 0.95, "Gen AI built on DL")
        self.add_relationship(rl, ml, "subfield_of", 1.0, "RL is branch of ML")
        self.add_relationship(kg_field, ir, "enhances", 0.8, "KGs enhance IR")

        # ── Task Relationships ────────────────────────────────
        self.add_relationship(sentiment, text_cls, "subtype_of", 0.95, "Sentiment is text classification")
        self.add_relationship(intent_det, text_cls, "subtype_of", 0.9, "Intent detection is classification")
        self.add_relationship(text_cls, nlp, "belongs_to", 1.0, "Text classification is NLP")
        self.add_relationship(qa, nlp, "belongs_to", 0.95, "QA is NLP task")
        self.add_relationship(code_gen, gen_ai, "belongs_to", 0.9, "Code gen is gen AI")
        self.add_relationship(ner, nlp, "belongs_to", 0.95, "NER is NLP task")

        # ── DL Frameworks ─────────────────────────────────────
        self.add_relationship(pytorch, python, "implemented_in", 1.0, "PyTorch in Python")
        self.add_relationship(pytorch, meta, "developed_by", 0.9, "PyTorch by Meta")
        self.add_relationship(pytorch, dl, "used_for", 1.0, "PyTorch for DL")
        self.add_relationship(tensorflow, python, "implemented_in", 1.0, "TF in Python")
        self.add_relationship(tensorflow, google, "developed_by", 1.0, "TF by Google")
        self.add_relationship(tensorflow, dl, "used_for", 1.0, "TF for DL")
        self.add_relationship(hf_transformers, hugging_face, "developed_by", 1.0, "HF Transformers by HF")
        self.add_relationship(hf_transformers, transfer_learn, "enables", 0.9, "HF enables transfer learning")
        self.add_relationship(hf_transformers, fine_tuning, "supports", 0.9, "HF supports fine-tuning")
        self.add_relationship(sklearn, ml, "used_for", 1.0, "sklearn for ML")

        # ── Data Libraries ────────────────────────────────────
        self.add_relationship(pandas, ds, "used_for", 0.95, "Pandas for data science")
        self.add_relationship(numpy, ml, "used_for", 0.9, "NumPy for numerical computing")

        # ── People ────────────────────────────────────────────
        self.add_relationship(hinton, dl, "pioneer_of", 1.0, "Hinton is godfather of DL")
        self.add_relationship(hinton, backprop, "popularized", 0.95, "Hinton popularized backprop")
        self.add_relationship(lecun, meta, "works_at", 0.9, "LeCun is at Meta")
        self.add_relationship(ng, ml, "educator_in", 0.95, "Ng is ML educator")
        self.add_relationship(ng, stanford, "affiliated_with", 0.9, "Ng at Stanford")
        self.add_relationship(altman, openai, "leads", 1.0, "Altman leads OpenAI")
        self.add_relationship(hassabis, deepmind, "leads", 1.0, "Hassabis leads DeepMind")

        # ── Org Relationships ─────────────────────────────────
        self.add_relationship(deepmind, google, "subsidiary_of", 0.95, "DeepMind is Google subsidiary")
        self.add_relationship(microsoft, openai, "investor_in", 0.9, "Microsoft invests in OpenAI")
        self.add_relationship(nvidia, dl, "enables", 0.95, "NVIDIA GPUs enable DL")
        self.add_relationship(stanford, ml, "research_hub_for", 0.95, "Stanford is ML research hub")

        # ── Concept Relationships ─────────────────────────────
        self.add_relationship(cot, prompt_eng, "technique_of", 0.9, "CoT is prompting technique")
        self.add_relationship(few_shot, prompt_eng, "technique_of", 0.85, "Few-shot is prompting")
        self.add_relationship(fine_tuning, transfer_learn, "method_of", 0.95, "Fine-tuning is transfer learning")
        self.add_relationship(rlhf, rl, "applies", 0.9, "RLHF applies RL")
        self.add_relationship(agents_concept, tool_use, "requires", 0.85, "Agents require tool use")

        # ── Infrastructure ────────────────────────────────────
        self.add_relationship(docker, k8s, "orchestrated_by", 0.85, "Docker orchestrated by K8s")
        self.add_relationship(docker, mlops, "used_in", 0.8, "Docker in MLOps")

        self._persist_if_needed()
        logger.info(
            "Seeded knowledge graph: %d nodes, %d edges",
            self.node_count,
            self.edge_count,
        )

    @property
    def node_count(self) -> int:
        return len(self._entities)

    @property
    def edge_count(self) -> int:
        return len(self._edge_index)
