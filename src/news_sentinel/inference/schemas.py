from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=4000)
    model: Literal["auto", "baseline", "textcnn"] = "auto"


class PredictResponse(BaseModel):
    label_id: int
    label_name: str
    model_used: str
    confidence: float
    class_scores: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    available_models: list[str]


class IncidentSummaryRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=4000)
    model: Literal["auto", "baseline", "textcnn"] = "auto"


class IncidentSummaryResponse(BaseModel):
    summary: str
    predicted_label: str
    model_used: str
    confidence: float
