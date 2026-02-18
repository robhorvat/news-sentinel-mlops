from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


class GeminiSummaryUnavailableError(RuntimeError):
    """Raised when Gemini summary is requested but unavailable."""


@dataclass(frozen=True)
class GeminiSummaryConfig:
    enabled: bool
    api_key: str
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> "GeminiSummaryConfig":
        enabled_raw = os.getenv("GEMINI_SUMMARY_ENABLED", "0").strip().lower()
        enabled = enabled_raw in {"1", "true", "yes", "on"}

        return cls(
            enabled=enabled,
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            model=os.getenv("GEMINI_SUMMARY_MODEL", "gemini-1.5-flash").strip(),
            temperature=float(os.getenv("GEMINI_SUMMARY_TEMPERATURE", "0.2")),
        )


class GeminiIncidentSummarizer:
    def __init__(self, config: GeminiSummaryConfig):
        if not config.enabled:
            raise GeminiSummaryUnavailableError(
                "Gemini summary disabled. Set GEMINI_SUMMARY_ENABLED=1 to enable."
            )

        if not config.api_key:
            raise GeminiSummaryUnavailableError(
                "Missing GEMINI_API_KEY. Set it before using /incident-summary."
            )

        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise GeminiSummaryUnavailableError(
                "google-generativeai not installed. Run: make install-gemini"
            ) from exc

        genai.configure(api_key=config.api_key)
        self._model_name = config.model
        self._temperature = config.temperature
        self._model = genai.GenerativeModel(config.model)

    @classmethod
    def from_env(cls) -> "GeminiIncidentSummarizer":
        return cls(GeminiSummaryConfig.from_env())

    @property
    def model_name(self) -> str:
        return self._model_name

    def summarize(
        self,
        headline: str,
        predicted_label: str,
        model_used: str,
        confidence: float,
        class_scores: Dict[str, float],
    ) -> str:
        score_text = ", ".join(
            f"label_{k}={v:.3f}" for k, v in sorted(class_scores.items(), key=lambda item: item[0])
        )

        prompt = (
            "You are assisting an ML operations analyst. "
            "Write a concise incident-style summary in 4 bullet points with these headers: "
            "Situation, Evidence, Risk, Next Action. "
            "Keep under 120 words total.\n\n"
            f"Headline: {headline}\n"
            f"Predicted label: {predicted_label}\n"
            f"Model used: {model_used}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Class scores: {score_text}\n"
        )

        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": self._temperature, "max_output_tokens": 220},
        )

        text = (getattr(response, "text", "") or "").strip()
        if text:
            return text

        return (
            "- Situation: The headline was classified by the News Sentinel pipeline.\n"
            f"- Evidence: Model={model_used}, label={predicted_label}, confidence={confidence:.2%}.\n"
            "- Risk: Prediction may be uncertain without additional context.\n"
            "- Next Action: Review supporting articles before making operational decisions."
        )
