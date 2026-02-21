"""Knowledge Graph Service - Entity relationship inference and persistence."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Set

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
        persist_path: str | Path | None = None,
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
        entity_type: str | None = None,
        max_depth: int = 2,
        relation_filter: str | None = None,
    ) -> dict[str, object]:
        """Run entity-centric graph inference.

        If `entity_type` is not provided, all entities with matching names are considered.
        """
        relation_filter_norm = relation_filter.lower().strip() if relation_filter else None
        matches: list[Entity]

        if entity_type:
            entity = self.get_entity(entity_name, entity_type)
            matches = [entity] if entity else []
        else:
            matches = self._find_by_name(entity_name)

        if not matches:
            return {
                "query": {
                    "name": entity_name,
                    "type": entity_type,
                    "max_depth": max_depth,
                    "relation_filter": relation_filter,
                },
                "matches": [],
                "related": [],
                "predicted_links": [],
            }

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

        return {
            "query": {
                "name": entity_name,
                "type": entity_type,
                "max_depth": max_depth,
                "relation_filter": relation_filter,
            },
            "matches": [entity.to_dict() for entity in matches],
            "related": related,
            "predicted_links": predicted_links,
        }

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
            target = prediction["target"]
            key = f"{target['type']}:{target['name'].lower()}"
            if key not in dedup or prediction["confidence"] > dedup[key]["confidence"]:
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

    def export_graph(self) -> dict[str, object]:
        nodes = [entity.to_dict() for entity in self._entities.values()]
        edges: list[dict[str, object]] = []
        for relationships in self._adjacency.values():
            for relationship in relationships:
                edges.append(relationship.to_dict())

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._persist_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.export_graph(), handle, ensure_ascii=True, indent=2)

    def load(self, path: str | Path | None = None) -> None:
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

    @property
    def node_count(self) -> int:
        return len(self._entities)

    @property
    def edge_count(self) -> int:
        return len(self._edge_index)
