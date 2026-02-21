"""Reusable text analysis engines (DSPy + rule-based fallback)."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence

from app.domain.enums import ConfidenceLevel
from app.models.classifier import ClassifierFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Normalized classifier payload."""

    data: dict[str, Any]


class TextAnalysisEngine(ABC):
    """Interface for text analysis engines."""

    @abstractmethod
    def classify_sentiment(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def classify_topic(self, text: str, categories: Sequence[str] | None = None) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def classify_intent(self, text: str, intents: Sequence[str] | None = None) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def classify_multi_label(self, text: str, labels: Sequence[str] | None = None) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def extract_entities(self, text: str) -> List[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def summarize(
        self,
        text: str,
        sentiment: dict[str, Any],
        topic: dict[str, Any],
        intent: dict[str, Any],
        entities: list[dict[str, str]],
    ) -> str:
        raise NotImplementedError


class RuleBasedTextAnalysisEngine(TextAnalysisEngine):
    """Deterministic fallback engine with no external dependency."""

    _POSITIVE = {
        "good",
        "great",
        "excellent",
        "amazing",
        "awesome",
        "love",
        "helpful",
        "fast",
        "happy",
        "best",
    }
    _NEGATIVE = {
        "bad",
        "terrible",
        "awful",
        "worst",
        "hate",
        "broken",
        "slow",
        "bug",
        "issue",
        "problem",
    }

    _TOPIC_KEYWORDS = {
        "Technology": {"python", "ai", "software", "code", "langgraph", "cloud", "api"},
        "Science": {"research", "experiment", "physics", "biology", "chemistry", "theory"},
        "Business": {"market", "revenue", "startup", "sales", "profit", "customer"},
        "Health": {"health", "medical", "doctor", "disease", "wellness", "fitness"},
        "Sports": {"game", "team", "score", "league", "player", "match"},
        "Politics": {"election", "policy", "government", "senate", "minister", "vote"},
        "Entertainment": {"movie", "music", "actor", "show", "series", "festival"},
        "Education": {"school", "student", "teacher", "course", "learning", "university"},
    }

    _QUESTION_STARTS = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "do",
        "does",
        "did",
    }

    _GREETING_WORDS = {"hello", "hi", "hey", "greetings"}
    _FAREWELL_WORDS = {"bye", "goodbye", "see you", "farewell"}

    def classify_sentiment(self, text: str) -> AnalysisResult:
        tokens = self._tokenize(text)
        pos = sum(token in self._POSITIVE for token in tokens)
        neg = sum(token in self._NEGATIVE for token in tokens)

        if pos > neg:
            sentiment = "positive"
        elif neg > pos:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        distance = abs(pos - neg)
        confidence = (
            ConfidenceLevel.HIGH.value
            if distance >= 2
            else ConfidenceLevel.MEDIUM.value
        )

        return AnalysisResult(
            {
                "sentiment": sentiment,
                "confidence": confidence,
                "reasoning": f"rule_based(pos={pos}, neg={neg})",
            }
        )

    def classify_topic(
        self,
        text: str,
        categories: Sequence[str] | None = None,
    ) -> AnalysisResult:
        tokens = set(self._tokenize(text))
        allowed_categories = [category.strip() for category in categories or [] if category.strip()]
        scores: dict[str, int] = {}

        candidate_map = self._TOPIC_KEYWORDS
        if allowed_categories:
            normalized = {category.lower(): category for category in allowed_categories}
            candidate_map = {
                category: keywords
                for category, keywords in self._TOPIC_KEYWORDS.items()
                if category.lower() in normalized
            }
            if not candidate_map:
                candidate_map = {allowed_categories[0]: set()}

        for category, keywords in candidate_map.items():
            scores[category] = len(tokens.intersection(keywords))

        best_category = max(scores, key=scores.get)
        confidence = (
            ConfidenceLevel.HIGH.value if scores[best_category] >= 2 else ConfidenceLevel.MEDIUM.value
        )

        return AnalysisResult(
            {
                "topic": best_category,
                "confidence": confidence,
                "reasoning": f"rule_based(scores={scores})",
            }
        )

    def classify_intent(
        self,
        text: str,
        intents: Sequence[str] | None = None,
    ) -> AnalysisResult:
        text_norm = text.strip().lower()
        tokens = self._tokenize(text)
        intent = "information"
        confidence = ConfidenceLevel.MEDIUM.value

        if any(text_norm.startswith(prefix) for prefix in self._QUESTION_STARTS) or text_norm.endswith("?"):
            intent = "question"
            confidence = ConfidenceLevel.HIGH.value
        elif "please" in tokens or "can you" in text_norm or "could you" in text_norm:
            intent = "request"
            confidence = ConfidenceLevel.HIGH.value
        elif any(word in tokens for word in ["issue", "problem", "broken", "hate", "worst"]):
            intent = "complaint"
        elif any(word in tokens for word in self._GREETING_WORDS):
            intent = "greeting"
        elif any(word in text_norm for word in self._FAREWELL_WORDS):
            intent = "farewell"

        allowed_intents = {intent_name.strip().lower() for intent_name in intents or [] if intent_name.strip()}
        if allowed_intents and intent not in allowed_intents:
            intent = next(iter(allowed_intents))
            confidence = ConfidenceLevel.LOW.value

        return AnalysisResult(
            {
                "intent": intent,
                "confidence": confidence,
                "entities": self.extract_entities(text),
                "reasoning": f"rule_based(intent={intent})",
            }
        )

    def classify_multi_label(
        self,
        text: str,
        labels: Sequence[str] | None = None,
    ) -> AnalysisResult:
        text_norm = text.strip().lower()
        inferred: list[str] = []

        if text.strip().endswith("?"):
            inferred.append("question")
        if "how to" in text_norm or text_norm.startswith("steps"):
            inferred.append("instructional")
        if any(word in text_norm for word in ["i think", "i believe", "in my opinion"]):
            inferred.append("opinion")
        if any(word in text_norm for word in ["should", "must", "need to", "buy"]):
            inferred.append("persuasive")
        if any(char.isdigit() for char in text):
            inferred.append("informative")

        if not inferred:
            inferred = ["informative"]

        allowed = [label.strip() for label in labels or [] if label.strip()]
        if allowed:
            allowed_set = {label.lower() for label in allowed}
            filtered = [label for label in inferred if label.lower() in allowed_set]
            inferred = filtered or [allowed[0]]

        return AnalysisResult(
            {
                "labels": ", ".join(dict.fromkeys(inferred)),
                "confidence": ConfidenceLevel.MEDIUM.value,
                "reasoning": "rule_based(label_heuristics)",
            }
        )

    def extract_entities(self, text: str) -> List[dict[str, str]]:
        found: list[dict[str, str]] = []

        for url in re.findall(r"https?://\S+", text):
            found.append({"text": url, "type": "URL"})

        for email in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
            found.append({"text": email, "type": "EMAIL"})

        for number in re.findall(r"\b\d+(?:\.\d+)?\b", text):
            found.append({"text": number, "type": "NUMBER"})

        proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9_]{1,}\b", text)
        for token in proper_nouns:
            label = "ORG" if token.isupper() else "CONCEPT"
            found.append({"text": token, "type": label})

        dedup: dict[str, dict[str, str]] = {}
        for entity in found:
            key = f"{entity['type']}:{entity['text'].lower()}"
            dedup[key] = entity

        return list(dedup.values())

    def summarize(
        self,
        text: str,
        sentiment: dict[str, Any],
        topic: dict[str, Any],
        intent: dict[str, Any],
        entities: list[dict[str, str]],
    ) -> str:
        entity_names = ", ".join(entity["text"] for entity in entities[:5]) or "none"
        return (
            f"Sentiment={sentiment.get('sentiment', 'unknown')}, "
            f"Topic={topic.get('topic', 'unknown')}, "
            f"Intent={intent.get('intent', 'unknown')}, "
            f"Entities={entity_names}."
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())


class DSPyTextAnalysisEngine(TextAnalysisEngine):
    """DSPy-backed analysis engine."""

    def __init__(self) -> None:
        self._sentiment = ClassifierFactory.create("sentiment")
        self._topic = ClassifierFactory.create("topic")
        self._intent = ClassifierFactory.create("intent")
        self._multi_label = ClassifierFactory.create("multi_label")
        self._entity = ClassifierFactory.create("entity")
        self._summarizer = ClassifierFactory.create("summarizer")

    def classify_sentiment(self, text: str) -> AnalysisResult:
        response = self._sentiment(text=text)
        return AnalysisResult(self._to_payload(response))

    def classify_topic(
        self,
        text: str,
        categories: Sequence[str] | None = None,
    ) -> AnalysisResult:
        kwargs: dict[str, Any] = {"text": text}
        if categories:
            kwargs["categories"] = ", ".join(categories)
        response = self._topic(**kwargs)
        return AnalysisResult(self._to_payload(response))

    def classify_intent(
        self,
        text: str,
        intents: Sequence[str] | None = None,
    ) -> AnalysisResult:
        kwargs: dict[str, Any] = {"text": text}
        if intents:
            kwargs["intents"] = ", ".join(intents)
        response = self._intent(**kwargs)
        payload = self._to_payload(response)
        payload["entities"] = self._normalize_entities(payload.get("entities", ""))
        return AnalysisResult(payload)

    def classify_multi_label(
        self,
        text: str,
        labels: Sequence[str] | None = None,
    ) -> AnalysisResult:
        kwargs: dict[str, Any] = {"text": text}
        if labels:
            kwargs["available_labels"] = ", ".join(labels)
        response = self._multi_label(**kwargs)
        return AnalysisResult(self._to_payload(response))

    def extract_entities(self, text: str) -> List[dict[str, str]]:
        response = self._entity(text=text)
        payload = self._to_payload(response)
        return self._normalize_entities(payload.get("entities", ""))

    def summarize(
        self,
        text: str,
        sentiment: dict[str, Any],
        topic: dict[str, Any],
        intent: dict[str, Any],
        entities: list[dict[str, str]],
    ) -> str:
        response = self._summarizer(
            text=text,
            sentiment=json.dumps(sentiment),
            topic=json.dumps(topic),
            intent=json.dumps(intent),
            entities=json.dumps(entities),
        )
        payload = self._to_payload(response)
        return str(payload.get("summary", "")).strip()

    @staticmethod
    def _to_payload(response: Any) -> dict[str, Any]:
        if hasattr(response, "items"):
            return {k: str(v) for k, v in response.items() if not k.startswith("_")}

        # DSPy Prediction objects expose attributes for fields.
        payload: dict[str, Any] = {}
        for attribute in dir(response):
            if attribute.startswith("_"):
                continue
            value = getattr(response, attribute)
            if callable(value):
                continue
            payload[attribute] = value
        return payload

    @staticmethod
    def _normalize_entities(raw: Any) -> list[dict[str, str]]:
        if isinstance(raw, list):
            entities = raw
        else:
            try:
                entities = json.loads(str(raw))
            except (json.JSONDecodeError, TypeError):
                entities = []

        normalized: list[dict[str, str]] = []
        for entity in entities if isinstance(entities, list) else []:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("text", entity.get("name", ""))).strip()
            if not name:
                continue
            entity_type = str(entity.get("type", "CONCEPT")).strip().upper()
            normalized.append({"text": name, "type": entity_type})

        return normalized


class HybridTextAnalysisEngine(TextAnalysisEngine):
    """Primary engine with deterministic fallback."""

    def __init__(
        self,
        fallback: TextAnalysisEngine | None = None,
        primary: TextAnalysisEngine | None = None,
    ) -> None:
        self._fallback = fallback or RuleBasedTextAnalysisEngine()
        self._primary = primary

    @property
    def has_primary(self) -> bool:
        return self._primary is not None

    def _execute(self, operation: str, primary_call: Any, fallback_call: Any) -> Any:
        if self._primary is not None:
            try:
                return primary_call()
            except Exception as exc:  # pragma: no cover - hard to deterministically trigger all DSPy failures
                logger.warning(
                    "Primary analysis engine failed (%s). Falling back to rule-based: %s",
                    operation,
                    exc,
                )
        return fallback_call()

    def classify_sentiment(self, text: str) -> AnalysisResult:
        return self._execute(
            "sentiment",
            lambda: self._primary.classify_sentiment(text),
            lambda: self._fallback.classify_sentiment(text),
        )

    def classify_topic(self, text: str, categories: Sequence[str] | None = None) -> AnalysisResult:
        return self._execute(
            "topic",
            lambda: self._primary.classify_topic(text, categories),
            lambda: self._fallback.classify_topic(text, categories),
        )

    def classify_intent(self, text: str, intents: Sequence[str] | None = None) -> AnalysisResult:
        return self._execute(
            "intent",
            lambda: self._primary.classify_intent(text, intents),
            lambda: self._fallback.classify_intent(text, intents),
        )

    def classify_multi_label(self, text: str, labels: Sequence[str] | None = None) -> AnalysisResult:
        return self._execute(
            "multi_label",
            lambda: self._primary.classify_multi_label(text, labels),
            lambda: self._fallback.classify_multi_label(text, labels),
        )

    def extract_entities(self, text: str) -> List[dict[str, str]]:
        return self._execute(
            "entity",
            lambda: self._primary.extract_entities(text),
            lambda: self._fallback.extract_entities(text),
        )

    def summarize(
        self,
        text: str,
        sentiment: dict[str, Any],
        topic: dict[str, Any],
        intent: dict[str, Any],
        entities: list[dict[str, str]],
    ) -> str:
        return self._execute(
            "summary",
            lambda: self._primary.summarize(text, sentiment, topic, intent, entities),
            lambda: self._fallback.summarize(text, sentiment, topic, intent, entities),
        )


def build_analysis_engine(enable_dspy: bool) -> HybridTextAnalysisEngine:
    """Build a hybrid engine with optional DSPy primary backend."""
    primary: TextAnalysisEngine | None = None
    if enable_dspy:
        try:
            primary = DSPyTextAnalysisEngine()
        except Exception as exc:
            logger.warning("Unable to initialize DSPy analysis engine: %s", exc)
    return HybridTextAnalysisEngine(primary=primary)
