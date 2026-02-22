"""Flask Routes (View Layer) — all HTTP endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError

from app.models.schemas import (
    AgentRequest,
    BatchClassificationRequest,
    ClassificationRequest,
    GraphInferenceRequest,
    HealthResponse,
)

logger = logging.getLogger(__name__)


def _validation_error_response(exc: ValidationError):
    details = json.loads(exc.json())
    return (
        jsonify(
            {
                "error": "Invalid request payload",
                "details": details,
            }
        ),
        400,
    )


def create_app(controller: Any = None) -> Flask:
    """Flask application factory (MVC pattern)."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    if controller is None:
        from app.controllers.classification_controller import ClassificationController

        controller = ClassificationController()
        controller.initialize()

    # ── Error handlers ───────────────────────────────
    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Not found", "status": 404}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "Method not allowed", "status": 405}), 405

    @app.errorhandler(500)
    def internal_error(_error):
        logger.exception("Internal server error")
        return jsonify({"error": "Internal server error", "status": 500}), 500

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        response = HealthResponse(
            status="healthy",
            provider=controller.provider,
            model=controller.model,
            initialized=controller.is_initialized,
            classifiers_available=controller.get_available_classifiers(),
        )
        return jsonify(response.model_dump())

    @app.route("/api/classifiers")
    def list_classifiers():
        available = controller.get_available_classifiers()
        details = {
            classifier_type: {
                "name": classifier_type.replace("_", " ").title(),
                "type": classifier_type,
            }
            for classifier_type in available
        }
        details["agent"] = {
            "name": "AI Agent",
            "type": "agent",
            "description": "LangGraph multi-step analysis",
        }
        return jsonify({"available": available + ["agent"], "details": details})

    @app.route("/api/classify", methods=["POST"])
    def classify():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "JSON body is required"}), 400

        try:
            schema = ClassificationRequest(
                text=data.get("text", ""),
                classifier_type=data.get("classifier_type", "sentiment"),
                categories=data.get("categories"),
                intents=data.get("intents"),
                labels=data.get("labels"),
            )
        except ValidationError as exc:
            return _validation_error_response(exc)

        result = controller.classify(schema)
        return jsonify(result.model_dump())

    @app.route("/api/classify/batch", methods=["POST"])
    def batch_classify():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "JSON body is required"}), 400

        try:
            schema = BatchClassificationRequest(
                texts=data.get("texts", []),
                classifier_type=data.get("classifier_type", "sentiment"),
                categories=data.get("categories"),
            )
        except ValidationError as exc:
            return _validation_error_response(exc)

        result = controller.batch_classify(schema)
        return jsonify(result.model_dump())

    @app.route("/api/agent/analyze", methods=["POST"])
    def agent_analyze():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "JSON body is required"}), 400

        try:
            schema = AgentRequest(
                text=data.get("text", ""),
                enable_knowledge_graph=data.get("enable_knowledge_graph", True),
            )
        except ValidationError as exc:
            return _validation_error_response(exc)

        result = controller.run_agent(schema)
        return jsonify(result.model_dump())

    @app.route("/api/knowledge-graph")
    def knowledge_graph():
        return jsonify(controller.get_knowledge_graph())

    @app.route("/api/knowledge-graph/seed", methods=["POST"])
    def seed_knowledge_graph():
        """Seed the knowledge graph with curated AI/ML data."""
        result = controller.reseed_knowledge_graph()
        return jsonify(result)

    @app.route("/api/graph/infer", methods=["POST"])
    def graph_infer():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "JSON body is required"}), 400

        try:
            schema = GraphInferenceRequest(
                entity=data.get("entity", ""),
                entity_type=data.get("entity_type"),
                max_depth=data.get("max_depth", 2),
                relation_filter=data.get("relation_filter"),
            )
        except ValidationError as exc:
            return _validation_error_response(exc)

        inference = controller.graph_infer(schema)
        return jsonify(inference)

    return app
