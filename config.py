"""Application-wide configuration (pydantic-settings singleton)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────
    app_name: str = "DSPy Classification Studio"
    environment: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_key: str = ""

    # ── Server ───────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Provider ─────────────────────────────────
    provider: Literal[
        "rule_based", "ollama", "gemini", "openai", "huggingface"
    ] = "rule_based"

    # ── Ollama ───────────────────────────────────
    ollama_model: str = "phi3:mini"
    ollama_base_url: str = "http://localhost:11434"

    # ── Gemini ───────────────────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini/gemini-2.0-flash"

    # ── OpenAI ───────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── HuggingFace ──────────────────────────────
    hf_token: str = ""
    hf_model: str = "meta-llama/Llama-2-7b-chat-hf"

    # ── Data paths ───────────────────────────────
    data_dir: Path = Path("./data")

    # ── Classification defaults ──────────────────
    default_categories: tuple[str, ...] = (
        "Technology", "Business", "Science", "Health",
        "Sports", "Entertainment", "Politics", "Education", "Other",
    )
    default_intents: tuple[str, ...] = (
        "question", "command", "greeting", "complaint",
        "feedback", "request", "information", "other",
    )

    # ── CORS ─────────────────────────────────────
    cors_origins: str = "*"

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper().strip()

    # ── Derived helpers ──────────────────────────

    @property
    def chroma_persist_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def graph_data_path(self) -> Path:
        return self.data_dir / "graph" / "knowledge_graph.json"

    def get_lm_config(self) -> Dict[str, Any]:
        """Return provider-specific DSPy LM kwargs."""
        configs: Dict[str, Dict[str, Any]] = {
            "ollama": {
                "provider": "ollama",
                "model": f"ollama/{self.ollama_model}",
                "api_base": self.ollama_base_url,
                "api_key": "ollama",
            },
            "gemini": {
                "provider": "gemini",
                "model": self.gemini_model,
                "api_key": self.google_api_key,
                "api_base": "",
            },
            "openai": {
                "provider": "openai",
                "model": self.openai_model,
                "api_key": self.openai_api_key,
                "api_base": "",
            },
            "huggingface": {
                "provider": "huggingface",
                "model": f"huggingface/{self.hf_model}",
                "api_key": self.hf_token,
                "api_base": "",
            },
            "rule_based": {
                "provider": "rule_based",
                "model": "rule_based",
                "api_key": "",
                "api_base": "",
            },
        }
        return configs[self.provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide singleton settings, creating data dirs."""
    settings = Settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.graph_data_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


def reset_settings() -> None:
    """Clear the singleton (for testing)."""
    get_settings.cache_clear()
