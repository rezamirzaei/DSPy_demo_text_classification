"""DSPy Service — manages LM lifecycle and configuration."""

from __future__ import annotations

import logging
import socket
from typing import Any, Optional
from urllib.parse import urlparse

import dspy

from app.models.schemas import ServiceHealthInfo

logger = logging.getLogger(__name__)


class DSPyService:
    """Manages DSPy language-model initialization."""

    def __init__(self, settings: Any = None) -> None:
        if settings is None:
            from config import get_settings
            settings = get_settings()
        self._settings = settings
        self._lm: Optional[Any] = None
        self._model_name: str = "unknown"
        self._provider: str = "unknown"
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider

    def initialize(self) -> bool:
        """Initialize DSPy with the configured LLM provider."""
        config = self._settings.get_lm_config()
        self._provider = config["provider"]
        self._model_name = config["model"]

        if self._provider == "rule_based":
            logger.info("Using rule-based backend — no LM needed")
            self._initialized = True
            return True

        logger.info(
            "Initializing DSPy: provider=%s, model=%s",
            self._provider,
            self._model_name,
        )

        # Avoid noisy per-request connection failures when Ollama is unreachable.
        if self._provider == "ollama":
            api_base = str(config.get("api_base", ""))
            if not self._is_ollama_reachable(api_base):
                logger.warning(
                    "Ollama is unreachable at %s. Falling back to rule-based analysis engine.",
                    api_base or "<empty>",
                )
                self._initialized = False
                return False

        # Validate API key for cloud providers
        if self._provider in ("openai", "google", "huggingface"):
            api_key = str(config.get("api_key", "")).strip()
            if not api_key:
                logger.warning(
                    "No API key configured for %s. Falling back to rule-based analysis engine.",
                    self._provider,
                )
                self._initialized = False
                return False

        try:
            lm_kwargs: dict[str, Any] = {"model": config["model"]}
            if config.get("api_key"):
                lm_kwargs["api_key"] = config["api_key"]
            if config.get("api_base"):
                lm_kwargs["api_base"] = config["api_base"]
            if config.get("keep_alive"):
                lm_kwargs["keep_alive"] = config["keep_alive"]
            lm_kwargs["timeout"] = int(getattr(self._settings, "lm_timeout_seconds", 45))
            lm_kwargs["max_tokens"] = int(getattr(self._settings, "lm_max_tokens", 256))
            lm_kwargs["num_retries"] = int(getattr(self._settings, "lm_num_retries", 1))
            self._lm = dspy.LM(**lm_kwargs)
            dspy.configure(lm=self._lm)
            self._initialized = True
            logger.info("DSPy initialized successfully")
            return True
        except Exception as exc:
            logger.error("Failed to initialize DSPy: %s", exc)
            self._initialized = False
            return False

    def health_check(self) -> ServiceHealthInfo:
        """Return LM health info as a validated Pydantic model."""
        return ServiceHealthInfo(
            initialized=self._initialized,
            provider=self._provider,
            model=self._model_name,
        )

    @staticmethod
    def _is_ollama_reachable(api_base: str) -> bool:
        if not api_base:
            return False

        parsed = urlparse(api_base)
        host = parsed.hostname
        if not host:
            return False

        if parsed.port:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80

        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            return False
