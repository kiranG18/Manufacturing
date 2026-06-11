# app/schemas/response.py
#
# Pydantic response models.
# Strict typing ensures the API contract is stable and breaking changes
# fail fast in tests rather than at the client.

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CostBreakdown(BaseModel):
    machine_time_cost: float = Field(..., description="Machine runtime cost in USD")
    setup_cost: float = Field(..., description="Setup and fixturing cost in USD")
    tooling_cost: float = Field(0.0, description="Amortised tooling cost in USD per part")
    raw_material_cost: float = Field(..., description="Material cost in USD")
    finishing_cost: float = Field(0.0, description="Post-process finishing cost in USD")
    total_cost: float = Field(..., description="Sum of all components in USD")
    confidence: str = Field(..., description="Confidence level: 'high', 'medium', or 'low'")
    price_date: Optional[str] = Field(None, description="Date of material price used (YYYY-MM-DD)")

    class Config:
        json_schema_extra = {
            "example": {
                "machine_time_cost": 38.50,
                "setup_cost": 12.00,
                "tooling_cost": 8.75,
                "raw_material_cost": 14.20,
                "finishing_cost": 3.50,
                "total_cost": 76.95,
                "confidence": "high",
                "price_date": "2024-11-12",
            }
        }


class ProcessRecommendation(BaseModel):
    rank: int = Field(..., ge=1, le=5, description="Recommendation rank (1 = most likely)")
    process_code: str = Field(..., description="Machine-readable process identifier")
    process_name: str = Field(..., description="Human-readable process display name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability score")
    confidence: str = Field(..., description="Confidence tier based on probability thresholds")
    cost_estimate: Optional[CostBreakdown] = Field(
        None,
        description="Itemised cost breakdown. Null if cost models are not available.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "process_code": "cnc_milling",
                "process_name": "CNC Milling",
                "probability": 0.72,
                "confidence": "high",
                "cost_estimate": {
                    "machine_time_cost": 38.50,
                    "setup_cost": 12.00,
                    "tooling_cost": 8.75,
                    "raw_material_cost": 14.20,
                    "finishing_cost": 3.50,
                    "total_cost": 76.95,
                    "confidence": "high",
                    "price_date": "2024-11-12",
                },
            }
        }


class RecommendResponse(BaseModel):
    part_id: Optional[str] = Field(None, description="Client-supplied part identifier")
    recommendations: List[ProcessRecommendation] = Field(..., min_length=1)
    model_version: str = Field(..., description="Model artifact version tag")
    latency_ms: float = Field(..., description="Server-side processing time in milliseconds")
    constraint_filtered_count: int = Field(
        0, description="Number of processes excluded by hard constraints"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "part_id": "PART-00217",
                "model_version": "classifier_v2_cost_v2",
                "latency_ms": 18.4,
                "constraint_filtered_count": 2,
                "recommendations": [
                    {
                        "rank": 1,
                        "process_code": "cnc_milling",
                        "process_name": "CNC Milling",
                        "probability": 0.72,
                        "confidence": "high",
                        "cost_estimate": None,
                    }
                ],
            }
        }


class FeatureContribution(BaseModel):
    feature_name: str
    value: float
    shap_contribution: float
    direction: str = Field(..., description="'positive' or 'negative' effect on predicted class")


class DebugResponse(BaseModel):
    part_id: Optional[str]
    predicted_process: str
    probability: float
    feature_contributions: List[FeatureContribution]
    top_features_summary: str = Field(
        ...,
        description="Plain-English summary of the two most influential features",
    )
    model_version: str
