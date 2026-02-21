"""Tests for app.services.dspy_service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.dspy_service import DSPyService


def _settings(config):
    return SimpleNamespace(get_lm_config=lambda: config)


def test_initialize_rule_based():
    service = DSPyService(_settings({"provider": "rule_based", "model": "rule_based"}))
    assert service.initialize() is True
    assert service.is_initialized is True
    assert service.provider == "rule_based"


@patch("app.services.dspy_service.dspy")
def test_initialize_success(mock_dspy):
    mock_dspy.LM.return_value = object()
    service = DSPyService(
        _settings(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "k",
                "api_base": "",
            }
        )
    )

    assert service.initialize() is True
    mock_dspy.configure.assert_called_once()


@patch("app.services.dspy_service.dspy")
def test_initialize_failure(mock_dspy):
    mock_dspy.LM.side_effect = RuntimeError("init failed")
    service = DSPyService(
        _settings(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "k",
                "api_base": "",
            }
        )
    )

    assert service.initialize() is False
    assert service.is_initialized is False


def test_health_check():
    service = DSPyService(_settings({"provider": "rule_based", "model": "rule_based"}))
    service.initialize()
    health = service.health_check()
    assert health["initialized"] is True
    assert health["provider"] == "rule_based"


@patch("app.services.dspy_service.dspy")
@patch("app.services.dspy_service.socket.create_connection")
def test_initialize_ollama_unreachable(mock_conn, mock_dspy):
    mock_conn.side_effect = OSError("refused")
    service = DSPyService(
        _settings(
            {
                "provider": "ollama",
                "model": "ollama/phi3:mini",
                "api_key": "ollama",
                "api_base": "http://localhost:11434",
            }
        )
    )

    assert service.initialize() is False
    mock_dspy.LM.assert_not_called()


@patch("app.services.dspy_service.dspy")
@patch("app.services.dspy_service.socket.create_connection")
def test_initialize_ollama_reachable(mock_conn, mock_dspy):
    mock_conn.return_value = MagicMock()
    mock_dspy.LM.return_value = object()
    service = DSPyService(
        _settings(
            {
                "provider": "ollama",
                "model": "ollama/phi3:mini",
                "api_key": "ollama",
                "api_base": "http://localhost:11434",
            }
        )
    )

    assert service.initialize() is True
    mock_dspy.LM.assert_called_once()
