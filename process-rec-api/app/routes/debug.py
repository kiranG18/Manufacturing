# app/routes/debug.py
#
# GET /v1/debug/{part_id} — explainability endpoint.
# Returns the SHAP feature contributions for a specific prediction so that
# engineers can understand why a process was recommended.
# Only accessible to admin/internal users (prod gating omitted here for clarity).

import logging

from fastapi import APIRouter, HTTPException, Query, status

from ..schemas.request import RecommendRequest
from ..schemas.response import DebugResponse
from ..services import classifier_service
from ..services.explainer_service import explain_prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Debug"])


@router.post(
    "/debug/explain",
    response_model=DebugResponse,
    summary="Explain a process recommendation",
    description=(
        "Returns SHAP feature contributions for the top-1 predicted process. "
        "Useful for diagnosing unexpected recommendations and verifying that the "
        "model is using the right features for each process type."
    ),
)
async def explain(request: RecommendRequest) -> DebugResponse:
    if not classifier_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier not loaded.",
        )

    features = request.to_feature_dict()
    explanation = explain_prediction(features)

    return DebugResponse(
        part_id=request.part_id,
        predicted_process=explanation["predicted_process"],
        probability=explanation["probability"],
        feature_contributions=explanation["shap_values"],
        top_features_summary=explanation["top_features"],
        model_version=classifier_service.model_version(),
    )
