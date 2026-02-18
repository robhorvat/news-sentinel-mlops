from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable


class GeminiSummaryUnavailableError(RuntimeError):
    """Raised when Gemini summary is requested but unavailable."""


class GeminiSummaryRuntimeError(RuntimeError):
    """Raised when Gemini call fails at request-time."""


@dataclass(frozen=True)
class GeminiSummaryConfig:
    enabled: bool
    api_key: str
    model: str
    fallback_models: tuple[str, ...]
    temperature: float

    @classmethod
    def from_env(cls) -> "GeminiSummaryConfig":
        enabled_raw = os.getenv("GEMINI_SUMMARY_ENABLED", "0").strip().lower()
        enabled = enabled_raw in {"1", "true", "yes", "on"}
        primary_model = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-2.5-flash").strip()
        fallback_raw = os.getenv(
            "GEMINI_SUMMARY_FALLBACK_MODELS",
            "",
        )
        fallback_models = tuple(
            model
            for model in (item.strip() for item in fallback_raw.split(","))
            if model and model != primary_model
        )

        return cls(
            enabled=enabled,
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            model=primary_model,
            fallback_models=fallback_models,
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
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types as genai_types
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise GeminiSummaryUnavailableError(
                "google-genai not installed. Run: make install-gemini"
            ) from exc

        self._client = genai.Client(api_key=config.api_key)
        self._genai_errors = genai_errors
        self._genai_types = genai_types
        self._model_name = config.model
        self._fallback_models = config.fallback_models
        self._temperature = config.temperature

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
        prompt = self._build_prompt(
            headline=headline,
            predicted_label=predicted_label,
            model_used=model_used,
            confidence=confidence,
            class_scores=class_scores,
        )

        models_to_try = tuple(dict.fromkeys((self._model_name, *self._fallback_models)))
        last_error = ""
        primary_error = ""

        for model_name in models_to_try:
            max_attempts = 2 if model_name == self._model_name else 1
            for _attempt in range(max_attempts):
                try:
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=self._genai_types.GenerateContentConfig(
                            temperature=self._temperature,
                            max_output_tokens=260,
                        ),
                    )
                    text = (getattr(response, "text", "") or "").strip()
                    if text and _has_required_sections(text):
                        return text
                    if text:
                        msg = (
                            f"incomplete Gemini summary from {model_name}: "
                            "missing one or more required sections"
                        )
                    else:
                        msg = f"empty Gemini response from model {model_name}"

                    if model_name == self._model_name:
                        primary_error = msg
                    last_error = msg
                    # Retry the primary model once before moving on.
                    continue
                except self._genai_errors.APIError as exc:
                    code = _normalize_error_code(getattr(exc, "status", "unknown"))
                    message = (getattr(exc, "message", "") or str(exc)).strip()
                    msg = f"Gemini API error ({code}): {message}"

                    # Do not let unavailable fallback model names hide primary model behavior.
                    if code == "NOT_FOUND" and model_name != self._model_name:
                        if not last_error:
                            last_error = msg
                        break

                    if model_name == self._model_name:
                        primary_error = msg
                    last_error = msg

                    # Auth/quota/permission/input errors should stop immediately.
                    if _is_terminal_api_error(code):
                        raise GeminiSummaryRuntimeError(last_error)
                    break
                except Exception as exc:
                    msg = f"Gemini request failed: {type(exc).__name__}: {exc}"
                    if model_name == self._model_name:
                        primary_error = msg
                    last_error = msg
                    break

        if primary_error:
            raise GeminiSummaryRuntimeError(primary_error)

        raise GeminiSummaryRuntimeError(last_error or "Unknown Gemini summary failure.")

    def _build_prompt(
        self,
        headline: str,
        predicted_label: str,
        model_used: str,
        confidence: float,
        class_scores: Dict[str, float],
    ) -> str:
        score_text = self._format_scores(class_scores)
        return (
            "You are an MLOps incident analyst.\n"
            "Return exactly 4 concise bullets, each starting with these labels in order:\n"
            "- Situation:\n"
            "- Evidence:\n"
            "- Risk:\n"
            "- Next Action:\n\n"
            "Rules:\n"
            "- Keep total length under 110 words.\n"
            "- Be specific to the headline and model evidence.\n"
            "- Do not mention policy or disclaimers.\n\n"
            f"Headline: {headline}\n"
            f"Predicted label: {predicted_label}\n"
            f"Model used: {model_used}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Class scores: {score_text}\n"
        )

    @staticmethod
    def _format_scores(class_scores: Dict[str, float]) -> str:
        label_names = {
            "0": "World",
            "1": "Sports",
            "2": "Business",
            "3": "Sci/Tech",
        }
        sorted_scores = sorted(
            class_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_items = sorted_scores[:4]
        parts = []
        for label_id, score in top_items:
            name = label_names.get(str(label_id), str(label_id))
            parts.append(f"{name}={score:.3f}")
        return ", ".join(parts)


def build_local_incident_summary(
    *,
    headline: str,
    predicted_label: str,
    model_used: str,
    confidence: float,
    class_scores: Dict[str, float],
    failure_note: str | None = None,
) -> str:
    top_two = _top_labels(class_scores, limit=2)
    alt_text = ", ".join(f"{name} {score:.1%}" for name, score in top_two)
    note = f" (Gemini fallback: {failure_note})" if failure_note else ""

    return (
        f"- Situation: Headline routed as {predicted_label} by {model_used}.{note}\n"
        f"- Evidence: Confidence={confidence:.2%}; top scores: {alt_text}.\n"
        "- Risk: Misclassification risk increases for short or ambiguous headlines.\n"
        "- Next Action: Cross-check with 2-3 supporting sources before escalation."
    )


def _has_required_sections(text: str) -> bool:
    required = ("Situation", "Evidence", "Risk", "Next Action")
    lowered = text.lower()
    return all(section.lower() in lowered for section in required)


def _normalize_error_code(code: object) -> str:
    if isinstance(code, int):
        return str(code)
    return str(code).strip().upper()


def _is_terminal_api_error(code: str) -> bool:
    terminal_codes = {
        "400",
        "401",
        "403",
        "404",
        "429",
        "INVALID_ARGUMENT",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
    }
    return code in terminal_codes


def _top_labels(class_scores: Dict[str, float], limit: int = 2) -> Iterable[tuple[str, float]]:
    label_names = {
        "0": "World",
        "1": "Sports",
        "2": "Business",
        "3": "Sci/Tech",
    }
    sorted_scores = sorted(
        class_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    for label_id, score in sorted_scores[:limit]:
        yield (label_names.get(str(label_id), str(label_id)), float(score))
