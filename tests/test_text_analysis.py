"""Tests for app.services.text_analysis."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.text_analysis import (
    DSPyTextAnalysisEngine,
    HybridTextAnalysisEngine,
    OllamaTextAnalysisEngine,
    RuleBasedTextAnalysisEngine,
    build_analysis_engine,
)


class TestRuleBasedTextAnalysisEngine:
    def test_classify_sentiment_positive(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_sentiment("Amazing and excellent product").data
        assert result["sentiment"] == "positive"

    def test_classify_sentiment_negative(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_sentiment("Worst broken awful issue").data
        assert result["sentiment"] == "negative"

    def test_classify_topic(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_topic("Python API and cloud service", ["Technology", "Health"]).data
        assert result["topic"] == "Technology"

    def test_classify_intent_question(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_intent("How can I deploy this?").data
        assert result["intent"] == "question"

    def test_classify_intent_respects_allowed_set(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_intent("How can I deploy this?", intents=["feedback"]).data
        assert result["intent"] == "feedback"

    def test_classify_multi_label(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_multi_label("How to install Python 3?").data
        assert "question" in result["labels"] or "instructional" in result["labels"]

    def test_classify_multi_label_allowed(self):
        engine = RuleBasedTextAnalysisEngine()
        result = engine.classify_multi_label("I think this is useful", labels=["opinion", "narrative"]).data
        assert result["labels"] in {"opinion", "narrative"}

    def test_extract_entities(self):
        engine = RuleBasedTextAnalysisEngine()
        entities = engine.extract_entities("Email me at test@example.com about Python 3")
        assert any(entity["type"] == "EMAIL" for entity in entities)
        assert any(entity["text"] == "Python" for entity in entities)

    def test_summarize(self):
        engine = RuleBasedTextAnalysisEngine()
        summary = engine.summarize(
            text="x",
            sentiment={"sentiment": "positive"},
            topic={"topic": "Technology"},
            intent={"intent": "question"},
            entities=[{"text": "Python", "type": "CONCEPT"}],
        )
        assert "Sentiment=positive" in summary


class TestDSPyTextAnalysisEngine:
    @patch("app.services.text_analysis.ClassifierFactory")
    def test_dspy_engine_methods(self, mock_factory):
        def _mk_return(name):
            if name == "entity":
                return lambda **_: {"entities": '[{"text": "Python", "type": "ORG"}]'}
            if name == "summarizer":
                return lambda **_: {"summary": "done"}
            if name == "intent":
                return lambda **_: {
                    "intent": "question",
                    "confidence": "high",
                    "entities": '[{"text": "API", "type": "CONCEPT"}]',
                }
            return lambda **_: {
                "sentiment": "positive", "confidence": "high",
                "topic": "Technology", "labels": "informative",
            }

        mock_factory.create.side_effect = _mk_return

        engine = DSPyTextAnalysisEngine()
        sentiment = engine.classify_sentiment("x").data
        topic = engine.classify_topic("x").data
        intent = engine.classify_intent("x").data
        labels = engine.classify_multi_label("x").data
        entities = engine.extract_entities("x")
        summary = engine.summarize("x", {}, {}, {}, [])

        assert sentiment["sentiment"] == "positive"
        assert topic["topic"] == "Technology"
        assert intent["intent"] == "question"
        assert labels["labels"] == "informative"
        assert entities[0]["text"] == "Python"
        assert summary == "done"

    @patch("app.services.text_analysis.ClassifierFactory")
    def test_dspy_engine_entity_parse_fallback(self, mock_factory):
        def _mk_return(name):
            if name == "entity":
                return lambda **_: {"entities": "not-json"}
            return lambda **_: {"summary": "x"}

        mock_factory.create.side_effect = _mk_return
        engine = DSPyTextAnalysisEngine()
        entities = engine.extract_entities("x")
        assert entities == []


class TestHybridTextAnalysisEngine:
    def test_falls_back_on_primary_error(self):
        fallback = RuleBasedTextAnalysisEngine()
        primary = MagicMock()
        primary.classify_sentiment.side_effect = RuntimeError("boom")
        hybrid = HybridTextAnalysisEngine(primary=primary, fallback=fallback)

        result = hybrid.classify_sentiment("great").data
        assert result["sentiment"] == "positive"

    def test_uses_primary_when_available(self):
        fallback = RuleBasedTextAnalysisEngine()
        primary = MagicMock()
        primary.classify_sentiment.return_value = MagicMock(data={"sentiment": "neutral"})
        hybrid = HybridTextAnalysisEngine(primary=primary, fallback=fallback)
        result = hybrid.classify_sentiment("great").data
        assert result["sentiment"] == "neutral"
        assert hybrid.has_primary is True

    def test_falls_back_on_primary_timeout(self):
        fallback = RuleBasedTextAnalysisEngine()
        primary = MagicMock()

        def _slow(*args, **kwargs):
            time.sleep(0.2)
            return MagicMock(data={"sentiment": "neutral"})

        primary.classify_sentiment.side_effect = _slow
        hybrid = HybridTextAnalysisEngine(
            primary=primary,
            fallback=fallback,
            primary_timeout_seconds=1,
        )
        hybrid._primary_timeout_seconds = 0.01
        result = hybrid.classify_sentiment("great").data
        assert result["sentiment"] == "positive"


class _FakeHTTPResponse:
    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


class TestOllamaTextAnalysisEngine:
    @patch("app.services.text_analysis.url_request.urlopen")
    def test_sentiment_json(self, mock_urlopen):
        mock_urlopen.return_value = _FakeHTTPResponse(
            '{"response":"{\\"sentiment\\":\\"positive\\",\\"confidence\\":\\"high\\",\\"reasoning\\":\\"great\\"}"}'
        )
        settings = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2:3b",
            lm_timeout_seconds=30,
            lm_max_tokens=64,
            ollama_keep_alive="30m",
        )
        engine = OllamaTextAnalysisEngine(settings)
        result = engine.classify_sentiment("I love this").data
        assert result["sentiment"] == "positive"
        assert result["confidence"] == "high"

    @patch("app.services.text_analysis.url_request.urlopen")
    def test_entities_json_array(self, mock_urlopen):
        mock_urlopen.return_value = _FakeHTTPResponse(
            '{"response":"[{\\\"text\\\":\\\"Python\\\",\\\"type\\\":\\\"CONCEPT\\\"}]"}'
        )
        settings = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2:3b",
            lm_timeout_seconds=30,
            lm_max_tokens=64,
            ollama_keep_alive="30m",
        )
        engine = OllamaTextAnalysisEngine(settings)
        entities = engine.extract_entities("Python")
        assert entities == [{"text": "Python", "type": "CONCEPT"}]


class TestBuildAnalysisEngine:
    def test_build_without_dspy(self):
        engine = build_analysis_engine(enable_dspy=False)
        assert engine.has_primary is False

    @patch("app.services.text_analysis.DSPyTextAnalysisEngine")
    def test_build_with_dspy_failure(self, mock_engine):
        mock_engine.side_effect = RuntimeError("fail")
        engine = build_analysis_engine(enable_dspy=True)
        assert engine.has_primary is False

    @patch("app.services.text_analysis.DSPyTextAnalysisEngine")
    def test_build_with_dspy_success(self, mock_engine):
        mock_engine.return_value = MagicMock()
        engine = build_analysis_engine(enable_dspy=True)
        assert engine.has_primary is True

    @patch("app.services.text_analysis._is_ollama_reachable", return_value=True)
    @patch("app.services.text_analysis.OllamaTextAnalysisEngine")
    def test_build_with_ollama_provider(self, mock_engine, _mock_reachable):
        settings = SimpleNamespace(provider="ollama", ollama_base_url="http://localhost:11434")
        mock_engine.return_value = MagicMock()
        engine = build_analysis_engine(enable_dspy=True, settings=settings)
        assert engine.has_primary is True

    @patch("app.services.text_analysis._is_ollama_reachable", return_value=False)
    def test_build_with_ollama_unreachable_skips_primary(self, _mock_reachable):
        settings = SimpleNamespace(
            provider="ollama",
            ollama_base_url="http://localhost:11434",
            primary_engine_timeout_seconds=40,
        )
        engine = build_analysis_engine(enable_dspy=True, settings=settings)
        assert engine.has_primary is False
