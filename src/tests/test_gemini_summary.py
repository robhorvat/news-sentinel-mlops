from news_sentinel.llm.gemini_summary import (
    GeminiSummaryConfig,
    _coerce_to_complete_summary,
    _has_required_sections,
    build_local_incident_summary,
)


def test_gemini_summary_config_defaults_disabled(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_SUMMARY_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_SUMMARY_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_SUMMARY_FALLBACK_MODELS", raising=False)

    config = GeminiSummaryConfig.from_env()
    assert config.enabled is False
    assert config.api_key == ""
    assert config.model == "gemini-2.5-flash"
    assert config.fallback_models == ()


def test_gemini_summary_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_SUMMARY_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_SUMMARY_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GEMINI_SUMMARY_FALLBACK_MODELS", "gemini-1.5-flash")
    monkeypatch.setenv("GEMINI_SUMMARY_TEMPERATURE", "0.3")

    config = GeminiSummaryConfig.from_env()
    assert config.enabled is True
    assert config.api_key == "test-key"
    assert config.model == "gemini-2.0-flash"
    assert config.fallback_models == ("gemini-1.5-flash",)
    assert config.temperature == 0.3


def test_build_local_incident_summary_contains_core_fields() -> None:
    text = build_local_incident_summary(
        headline="Oil prices climb after supply chain disruptions",
        predicted_label="Business",
        model_used="baseline",
        confidence=0.78,
        class_scores={"0": 0.07, "1": 0.06, "2": 0.78, "3": 0.09},
        failure_note="Gemini API error (401): invalid key",
    )
    assert "Situation" in text
    assert "Business" in text
    assert "Evidence" in text
    assert "Next Action" in text


def test_has_required_sections() -> None:
    good = (
        "- Situation: x\n"
        "- Evidence: y\n"
        "- Risk: z\n"
        "- Next Action: n\n"
    )
    bad = "- Situation: x"
    assert _has_required_sections(good) is True
    assert _has_required_sections(bad) is False


def test_coerce_to_complete_summary_with_partial_text() -> None:
    partial = "- Situation: The classifier predicts Sports for this headline."
    completed = _coerce_to_complete_summary(
        source_text=partial,
        predicted_label="Sports",
        model_used="baseline",
        confidence=0.80,
        class_scores={"0": 0.06, "1": 0.80, "2": 0.07, "3": 0.07},
    )
    assert "Situation" in completed
    assert "Evidence" in completed
    assert "Risk" in completed
    assert "Next Action" in completed
