"""DSPy Optimizer — BootstrapFewShot prompt optimization.

This module implements DSPy's unique value proposition: automatic prompt
optimization via labelled examples.  It provides a reusable ``DSPyOptimizer``
class that can optimize any DSPy module against a training set and persist
the result so that subsequent cold starts are instant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import dspy

logger = logging.getLogger(__name__)


# ── Training examples ────────────────────────────────────

SENTIMENT_EXAMPLES: list[dspy.Example] = [
    dspy.Example(
        text="I absolutely love this product! Best purchase I've made.",
        sentiment="positive",
        confidence="high",
        reasoning="Strong positive language: love, best.",
    ).with_inputs("text"),
    dspy.Example(
        text="This is terrible. Completely broken and waste of money.",
        sentiment="negative",
        confidence="high",
        reasoning="Strong negative language: terrible, broken, waste.",
    ).with_inputs("text"),
    dspy.Example(
        text="The meeting is scheduled for 3 PM tomorrow.",
        sentiment="neutral",
        confidence="high",
        reasoning="Factual statement with no emotional valence.",
    ).with_inputs("text"),
    dspy.Example(
        text="The service was okay, nothing special but nothing bad.",
        sentiment="neutral",
        confidence="medium",
        reasoning="Mixed or indifferent language.",
    ).with_inputs("text"),
    dspy.Example(
        text="I'm so frustrated with the constant bugs and crashes.",
        sentiment="negative",
        confidence="high",
        reasoning="Frustrated, bugs, crashes indicate negative sentiment.",
    ).with_inputs("text"),
    dspy.Example(
        text="Great framework for building AI applications quickly.",
        sentiment="positive",
        confidence="high",
        reasoning="Positive descriptor: great, quickly.",
    ).with_inputs("text"),
]

TOPIC_EXAMPLES: list[dspy.Example] = [
    dspy.Example(
        text="Python 3.12 introduces new pattern matching features.",
        categories="Technology, Science, Business, Health, Sports, Politics, Entertainment, Education, Other",
        topic="Technology",
        confidence="high",
        reasoning="Discusses programming language features.",
    ).with_inputs("text", "categories"),
    dspy.Example(
        text="The stock market rallied after the Fed's announcement.",
        categories="Technology, Science, Business, Health, Sports, Politics, Entertainment, Education, Other",
        topic="Business",
        confidence="high",
        reasoning="Financial markets and central bank policy.",
    ).with_inputs("text", "categories"),
    dspy.Example(
        text="A new vaccine shows 95% efficacy in clinical trials.",
        categories="Technology, Science, Business, Health, Sports, Politics, Entertainment, Education, Other",
        topic="Health",
        confidence="high",
        reasoning="Medical trial results about a vaccine.",
    ).with_inputs("text", "categories"),
    dspy.Example(
        text="The team won the championship in overtime.",
        categories="Technology, Science, Business, Health, Sports, Politics, Entertainment, Education, Other",
        topic="Sports",
        confidence="high",
        reasoning="Competition, team, championship are sports terms.",
    ).with_inputs("text", "categories"),
]

INTENT_EXAMPLES: list[dspy.Example] = [
    dspy.Example(
        text="What is the capital of France?",
        intents="question, request, complaint, feedback, greeting, farewell, information, other",
        intent="question",
        confidence="high",
        entities="[]",
        reasoning="Interrogative sentence seeking factual information.",
    ).with_inputs("text", "intents"),
    dspy.Example(
        text="Please send me the report by Friday.",
        intents="question, request, complaint, feedback, greeting, farewell, information, other",
        intent="request",
        confidence="high",
        entities='[{"text": "Friday", "type": "DATE"}]',
        reasoning="Polite imperative requesting an action.",
    ).with_inputs("text", "intents"),
    dspy.Example(
        text="Your product broke after one day. I want a refund.",
        intents="question, request, complaint, feedback, greeting, farewell, information, other",
        intent="complaint",
        confidence="high",
        entities="[]",
        reasoning="Expresses dissatisfaction and demands resolution.",
    ).with_inputs("text", "intents"),
]


def _metric_exact_match(example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> bool:
    """Metric for BootstrapFewShot: check if the primary output matches."""
    # Try multiple field names since different signatures use different keys
    for field in ("sentiment", "topic", "intent"):
        expected = getattr(example, field, None)
        predicted = getattr(prediction, field, None)
        if expected is not None and predicted is not None:
            return str(predicted).strip().lower() == str(expected).strip().lower()
    return False


class DSPyOptimizer:
    """Optimize DSPy modules using BootstrapFewShot.

    Optimized modules are serialised to ``{cache_dir}/{name}.json`` so that
    the expensive bootstrap runs only once per training set.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/dspy_optimized")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def optimize(
        self,
        name: str,
        module: dspy.Module,
        train_examples: Sequence[dspy.Example],
        metric: Any = None,
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 4,
    ) -> dspy.Module:
        """Return an optimized module, loading from cache if available."""
        cache_path = self._cache_dir / f"{name}.json"

        if cache_path.exists():
            try:
                module.load(str(cache_path))
                logger.info("Loaded optimized module '%s' from cache", name)
                return module
            except Exception as exc:
                logger.warning("Cache load failed for '%s': %s — re-optimizing", name, exc)

        if not train_examples:
            logger.warning("No training examples for '%s' — returning unoptimized module", name)
            return module

        metric_fn = metric or _metric_exact_match

        try:
            optimizer = dspy.BootstrapFewShot(
                metric=metric_fn,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
            optimized = optimizer.compile(module, trainset=list(train_examples))

            try:
                optimized.save(str(cache_path))
                logger.info("Optimized module '%s' saved to %s", name, cache_path)
            except Exception as exc:
                logger.warning("Failed to persist optimized module '%s': %s", name, exc)

            return optimized

        except Exception as exc:
            logger.warning("Optimization failed for '%s': %s — returning base module", name, exc)
            return module

    def optimize_all_classifiers(self) -> dict[str, dspy.Module]:
        """Optimize all standard classifiers and return them keyed by name."""
        from app.models.classifier import (
            IntentClassifier,
            SentimentClassifier,
            TopicClassifier,
        )

        results: dict[str, dspy.Module] = {}

        results["sentiment"] = self.optimize(
            "sentiment",
            dspy.Predict(SentimentClassifier),
            SENTIMENT_EXAMPLES,
        )
        results["topic"] = self.optimize(
            "topic",
            dspy.Predict(TopicClassifier),
            TOPIC_EXAMPLES,
        )
        results["intent"] = self.optimize(
            "intent",
            dspy.Predict(IntentClassifier),
            INTENT_EXAMPLES,
        )

        return results
