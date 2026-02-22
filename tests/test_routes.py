"""Tests for app.views.routes — Flask HTTP endpoints."""

import json


class TestHealthEndpoint:
    def test_health_returns_200(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["initialized"] is True

    def test_health_has_provider(self, app_client):
        resp = app_client.get("/health")
        data = resp.get_json()
        assert "provider" in data
        assert "model" in data


class TestIndexEndpoint:
    def test_index_returns_html(self, app_client):
        resp = app_client.get("/")
        assert resp.status_code == 200
        assert b"DSPy Classification Studio" in resp.data


class TestClassifiersEndpoint:
    def test_list_classifiers(self, app_client):
        resp = app_client.get("/api/classifiers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sentiment" in data["available"]
        assert "agent" in data["available"]


class TestClassifyEndpoint:
    def test_classify_success(self, app_client):
        resp = app_client.post(
            "/api/classify",
            data=json.dumps({"text": "I love this!", "classifier_type": "sentiment"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["result"]["sentiment"] == "positive"

    def test_classify_empty_text(self, app_client):
        resp = app_client.post(
            "/api/classify",
            data=json.dumps({"text": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_classify_invalid_type(self, app_client):
        resp = app_client.post(
            "/api/classify",
            data=json.dumps({"text": "hello", "classifier_type": "bogus"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_classify_no_json(self, app_client):
        resp = app_client.post("/api/classify")
        assert resp.status_code == 400


class TestBatchClassifyEndpoint:
    def test_batch_success(self, app_client):
        resp = app_client.post(
            "/api/classify/batch",
            data=json.dumps({"texts": ["a", "b"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 2
        assert data["successful"] == 2

    def test_batch_empty_texts(self, app_client):
        resp = app_client.post(
            "/api/classify/batch",
            data=json.dumps({"texts": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestAgentEndpoint:
    def test_agent_analyze(self, app_client):
        resp = app_client.post(
            "/api/agent/analyze",
            data=json.dumps({"text": "Analyze this please"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "steps" in data

    def test_agent_empty_text(self, app_client):
        resp = app_client.post(
            "/api/agent/analyze",
            data=json.dumps({"text": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestKnowledgeGraphEndpoint:
    def test_knowledge_graph(self, app_client):
        resp = app_client.get("/api/knowledge-graph")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data
        assert "edges" in data


class TestGraphInferEndpoint:
    def test_graph_infer(self, app_client):
        resp = app_client.post(
            "/api/graph/infer",
            data=json.dumps({"entity": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["query"]["name"] == "test"
        assert "related" in data
        assert "predicted_links" in data

    def test_graph_infer_empty(self, app_client):
        resp = app_client.post(
            "/api/graph/infer",
            data=json.dumps({"entity": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestSeedEndpoint:
    def test_seed_knowledge_graph(self, app_client):
        resp = app_client.post("/api/knowledge-graph/seed")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "message" in data
        assert data["node_count"] >= 0


class TestErrorHandlers:
    def test_404_returns_json(self, app_client):
        resp = app_client.get("/nonexistent-path")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["error"] == "Not found"

    def test_405_returns_json(self, app_client):
        resp = app_client.delete("/api/classify")
        assert resp.status_code == 405
        data = resp.get_json()
        assert data["error"] == "Method not allowed"


