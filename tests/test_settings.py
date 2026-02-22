"""Tests for config.py — Settings."""

from config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings(provider="rule_based")
        assert s.app_name == "DSPy Classification Studio"
        assert s.port == 8000
        assert s.debug is False

    def test_log_level_normalized(self):
        s = Settings(log_level="  info  ", provider="rule_based")
        assert s.log_level == "INFO"

    def test_provider_options(self):
        for p in ["rule_based", "ollama", "openai", "huggingface"]:
            s = Settings(provider=p)
            assert s.provider == p

    def test_get_lm_config_ollama(self):
        s = Settings(provider="ollama", ollama_model="phi3:mini")
        config = s.get_lm_config()
        assert config["provider"] == "ollama"
        assert "ollama/phi3:mini" in config["model"]

    def test_get_lm_config_openai(self):
        s = Settings(provider="openai", openai_api_key="test-key")
        config = s.get_lm_config()
        assert config["provider"] == "openai"
        assert config["api_key"] == "test-key"

    def test_ollama_keep_alive(self):
        s = Settings(provider="ollama", ollama_keep_alive="45m")
        config = s.get_lm_config()
        assert config["keep_alive"] == "45m"

    def test_get_lm_config_rule_based(self):
        s = Settings(provider="rule_based")
        config = s.get_lm_config()
        assert config["provider"] == "rule_based"

    def test_chroma_persist_dir(self):
        s = Settings(provider="rule_based")
        assert str(s.chroma_persist_dir).endswith("chroma")

    def test_graph_data_path(self):
        s = Settings(provider="rule_based")
        assert str(s.graph_data_path).endswith("knowledge_graph.json")

    def test_default_categories(self):
        s = Settings(provider="rule_based")
        assert "Technology" in s.default_categories
        assert len(s.default_categories) == 9

    def test_default_intents(self):
        s = Settings(provider="rule_based")
        assert "question" in s.default_intents
        assert len(s.default_intents) == 8
