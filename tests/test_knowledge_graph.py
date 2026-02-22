"""Tests for app.services.knowledge_graph."""

import json

import pytest
from app.services.knowledge_graph import Entity, KnowledgeGraph, Relationship


class TestEntity:
    def test_creation(self):
        e = Entity(name="Python", entity_type="LANGUAGE")
        assert e.name == "Python"
        assert e.entity_type == "LANGUAGE"

    def test_equality_case_insensitive(self):
        e1 = Entity(name="python", entity_type="language")
        e2 = Entity(name="Python", entity_type="LANGUAGE")
        assert e1 == e2

    def test_hash_case_insensitive(self):
        e1 = Entity(name="python", entity_type="language")
        e2 = Entity(name="Python", entity_type="LANGUAGE")
        assert hash(e1) == hash(e2)

    def test_to_dict(self):
        e = Entity(name="Go", entity_type="LANG")
        assert e.to_dict() == {"name": "Go", "type": "LANG"}

    def test_not_equal_to_non_entity(self):
        e = Entity(name="x", entity_type="Y")
        assert e != "not an entity"


class TestRelationship:
    def test_creation(self):
        src = Entity(name="A", entity_type="X")
        tgt = Entity(name="B", entity_type="Y")
        rel = Relationship(source=src, target=tgt, relation_type="uses")
        assert rel.weight == 1.0
        assert rel.context == ""

    def test_to_dict(self):
        src = Entity(name="A", entity_type="X")
        tgt = Entity(name="B", entity_type="Y")
        rel = Relationship(source=src, target=tgt, relation_type="likes", weight=0.5)
        d = rel.to_dict()
        assert d["relation"] == "likes"
        assert d["weight"] == 0.5


class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self):
        return KnowledgeGraph()

    def test_add_entity(self, kg):
        e = kg.add_entity("Python", "LANGUAGE")
        assert e.name == "Python"
        assert kg.node_count == 1

    def test_add_duplicate_entity(self, kg):
        kg.add_entity("Python", "LANGUAGE")
        kg.add_entity("python", "language")
        assert kg.node_count == 1

    def test_add_relationship(self, kg):
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "Y")
        rel = kg.add_relationship(a, b, "uses")
        assert rel.relation_type == "uses"
        assert kg.edge_count == 1

    def test_get_entity(self, kg):
        kg.add_entity("Python", "LANGUAGE")
        found = kg.get_entity("Python", "LANGUAGE")
        assert found is not None
        assert found.name == "Python"

    def test_get_entity_not_found(self, kg):
        assert kg.get_entity("Missing", "X") is None

    def test_get_neighbors(self, kg):
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "Y")
        kg.add_relationship(a, b, "knows")
        neighbors = kg.get_neighbors(a)
        assert len(neighbors) == 1
        assert neighbors[0][0].name == "B"

    def test_get_incoming(self, kg):
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "Y")
        kg.add_relationship(a, b, "knows")
        incoming = kg.get_incoming(b)
        assert len(incoming) == 1
        assert incoming[0][0].name == "A"

    def test_get_entities_by_type(self, kg):
        kg.add_entity("A", "LANG")
        kg.add_entity("B", "LANG")
        kg.add_entity("C", "OTHER")
        result = kg.get_entities_by_type("LANG")
        assert len(result) == 2

    def test_find_related(self, kg):
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "Y")
        c = kg.add_entity("C", "Z")
        kg.add_relationship(a, b, "knows")
        kg.add_relationship(b, c, "knows")
        related = kg.find_related(a, max_depth=2)
        names = {r["entity"]["name"] for r in related}
        assert "B" in names
        assert "C" in names

    def test_find_related_unknown_entity(self, kg):
        unknown = Entity(name="ghost", entity_type="X")
        assert kg.find_related(unknown) == []

    def test_infer_connections(self, kg):
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "X")
        shared = kg.add_entity("S", "Y")
        kg.add_relationship(a, shared, "uses")
        kg.add_relationship(b, shared, "uses")
        inferences = kg.infer_connections([a, b])
        assert len(inferences) == 1
        assert inferences[0]["inference_strength"] > 0

    def test_build_from_entities(self, kg):
        dicts = [
            {"text": "Python", "type": "LANG"},
            {"text": "DSPy", "type": "FRAMEWORK"},
        ]
        kg.build_from_entities(dicts, source_text="test")
        assert kg.node_count >= 2
        assert kg.edge_count >= 1

    def test_export_graph(self, kg):
        kg.add_entity("X", "T")
        data = kg.export_graph()
        assert "nodes" in data
        assert "edges" in data
        assert data["node_count"] == 1

    def test_clear(self, kg):
        kg.add_entity("X", "T")
        kg.clear()
        assert kg.node_count == 0
        assert kg.edge_count == 0

    def test_add_entity_empty_name_raises(self, kg):
        with pytest.raises(ValueError, match="must not be empty"):
            kg.add_entity("", "TYPE")

    def test_infer_for_entity(self, kg):
        a = kg.add_entity("Python", "CONCEPT")
        b = kg.add_entity("LangGraph", "CONCEPT")
        kg.add_relationship(a, b, "related_to")

        result = kg.infer_for_entity("Python", entity_type="CONCEPT", max_depth=2)
        assert result["query"]["name"] == "Python"
        assert len(result["matches"]) == 1
        assert isinstance(result["related"], list)

    def test_load_legacy_string_edges(self, tmp_path):
        graph_path = tmp_path / "legacy_graph.json"
        graph_path.write_text(
            json.dumps(
                {
                    "nodes": [],
                    "edges": [
                        {
                            "source": "CONCEPT:Python",
                            "target": "CONCEPT:LangGraph",
                            "relation": "related_to",
                            "weight": 1.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        kg = KnowledgeGraph(persist_path=graph_path)
        assert kg.node_count == 2
        assert kg.edge_count == 1


class TestKnowledgeGraphSeed:
    """Tests for the seed_default_graph method."""

    def test_seed_populates_graph(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        assert kg.node_count > 50
        assert kg.edge_count > 80

    def test_seed_skips_if_non_empty(self):
        kg = KnowledgeGraph()
        kg.add_entity("TestEntity", "TEST")
        kg.seed_default_graph()
        # Should not add the seed data because graph was non-empty
        assert kg.node_count == 1

    def test_seed_contains_dspy_entity(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        entity = kg.get_entity("DSPy", "FRAMEWORK")
        assert entity is not None
        assert entity.name == "DSPy"

    def test_seed_contains_langgraph_entity(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        entity = kg.get_entity("LangGraph", "FRAMEWORK")
        assert entity is not None

    def test_seed_dspy_relationships(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        dspy = kg.get_entity("DSPy", "FRAMEWORK")
        neighbors = kg.get_neighbors(dspy)
        neighbor_names = {n[0].name for n in neighbors}
        assert "Python" in neighbor_names
        assert "Stanford University" in neighbor_names
        assert "Omar Khattab" in neighbor_names

    def test_seed_inference_works(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        result = kg.infer_for_entity("Python", entity_type="LANGUAGE", max_depth=2)
        assert len(result["matches"]) == 1
        assert len(result["related"]) > 0

    def test_seed_predict_links(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        dspy = kg.get_entity("DSPy", "FRAMEWORK")
        predictions = kg._predict_links(dspy)
        # DSPy connects to Python, Stanford, etc. which connect onwards
        assert len(predictions) > 0

    def test_seed_infer_connections(self):
        kg = KnowledgeGraph()
        kg.seed_default_graph()
        dspy = kg.get_entity("DSPy", "FRAMEWORK")
        bert = kg.get_entity("BERT", "MODEL")
        inferences = kg.infer_connections([dspy, bert])
        # Both connect to similar entities (NLP tasks, etc.)
        assert isinstance(inferences, list)


class TestKnowledgeGraphPersistence:
    """Tests for save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test_graph.json"
        kg1 = KnowledgeGraph(persist_path=path)
        a = kg1.add_entity("Alpha", "TYPE_A")
        b = kg1.add_entity("Beta", "TYPE_B")
        kg1.add_relationship(a, b, "connects_to", 0.75, "test context")
        kg1.save()

        kg2 = KnowledgeGraph(persist_path=path)
        assert kg2.node_count == 2
        assert kg2.edge_count == 1
        entity = kg2.get_entity("Alpha", "TYPE_A")
        assert entity is not None

    def test_auto_persist(self, tmp_path):
        path = tmp_path / "auto_graph.json"
        kg = KnowledgeGraph(persist_path=path, auto_persist=True)
        kg.add_entity("TestEntity", "TEST")
        assert path.exists()

    def test_seed_then_save_load(self, tmp_path):
        path = tmp_path / "seed_graph.json"
        kg1 = KnowledgeGraph(persist_path=path)
        kg1.seed_default_graph()
        kg1.save()

        kg2 = KnowledgeGraph(persist_path=path)
        assert kg2.node_count == kg1.node_count
        assert kg2.edge_count == kg1.edge_count


class TestKnowledgeGraphEdgeCases:
    """Edge-case tests."""

    def test_duplicate_relationship_updates_weight(self):
        kg = KnowledgeGraph()
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "Y")
        kg.add_relationship(a, b, "knows", 0.5)
        kg.add_relationship(a, b, "knows", 0.9)
        assert kg.edge_count == 1
        neighbors = kg.get_neighbors(a)
        assert neighbors[0][2] == 0.9  # weight should be max

    def test_find_related_max_depth_1(self):
        kg = KnowledgeGraph()
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "X")
        c = kg.add_entity("C", "X")
        kg.add_relationship(a, b, "r1")
        kg.add_relationship(b, c, "r2")
        related = kg.find_related(a, max_depth=1)
        names = {r["entity"]["name"] for r in related}
        assert "B" in names
        assert "C" not in names

    def test_infer_for_entity_no_match(self):
        kg = KnowledgeGraph()
        result = kg.infer_for_entity("NonExistent", entity_type="FAKE")
        assert len(result["matches"]) == 0
        assert len(result["related"]) == 0

    def test_infer_for_entity_no_type(self):
        kg = KnowledgeGraph()
        kg.add_entity("Shared", "TYPE_A")
        kg.add_entity("Shared", "TYPE_B")
        result = kg.infer_for_entity("Shared")
        assert len(result["matches"]) == 2

    def test_relation_filter(self):
        kg = KnowledgeGraph()
        a = kg.add_entity("A", "X")
        b = kg.add_entity("B", "Y")
        c = kg.add_entity("C", "Z")
        kg.add_relationship(a, b, "uses")
        kg.add_relationship(a, c, "develops")
        result = kg.infer_for_entity("A", entity_type="X", relation_filter="uses")
        relations = [r["relation"] for r in result["related"]]
        assert all("uses" in r.lower() for r in relations)

    def test_export_graph_structure(self):
        kg = KnowledgeGraph()
        kg.add_entity("Test", "TYPE")
        data = kg.export_graph()
        assert "nodes" in data
        assert "edges" in data
        assert "node_count" in data
        assert "edge_count" in data
        assert data["node_count"] == 1
        assert data["edge_count"] == 0

    def test_build_from_entities_with_empty_names(self):
        kg = KnowledgeGraph()
        dicts = [
            {"text": "", "type": "LANG"},
            {"text": "Python", "type": "LANG"},
            {"name": "DSPy", "type": "FRAMEWORK"},
        ]
        kg.build_from_entities(dicts, source_text="testing")
        assert kg.node_count == 2  # empty name should be skipped

    def test_parse_legacy_entity_string(self):
        name, entity_type = KnowledgeGraph._parse_legacy_entity("CONCEPT:Python")
        assert name == "Python"
        assert entity_type == "CONCEPT"

    def test_parse_legacy_entity_empty_string(self):
        name, entity_type = KnowledgeGraph._parse_legacy_entity("")
        assert name == ""
        assert entity_type == "CONCEPT"

    def test_parse_legacy_entity_non_string_non_dict(self):
        name, entity_type = KnowledgeGraph._parse_legacy_entity(42)
        assert name == ""
        assert entity_type == "CONCEPT"


