from news_sentinel.llm.gemini_summary import GeminiSummaryConfig


def test_gemini_summary_config_defaults_disabled(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_SUMMARY_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = GeminiSummaryConfig.from_env()
    assert config.enabled is False
    assert config.api_key == ""
    assert config.model == "gemini-1.5-flash"


def test_gemini_summary_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_SUMMARY_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_SUMMARY_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GEMINI_SUMMARY_TEMPERATURE", "0.3")

    config = GeminiSummaryConfig.from_env()
    assert config.enabled is True
    assert config.api_key == "test-key"
    assert config.model == "gemini-2.0-flash"
    assert config.temperature == 0.3
