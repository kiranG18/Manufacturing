# app/routes/recommend.py
#
# POST /v1/recommend — the primary inference endpoint.
#
# Request flow:
#   1. Validate incoming geometry features (Pydantic)
#   2. Run hard constraint filter (e.g. exclude molding if undercut=1)
#   3. Rank processes via process classifier
#   4. Fetch per-process cost estimates for top-k results
#   5. Return ranked list with costs and confidence

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas.request import RecommendRequest
from ..schemas.response import RecommendResponse, ProcessRecommendation
from ..services import classifier_service, cost_service
from ..services.constraint_filter import apply_hard_constraints

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Recommendations"])


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Recommend manufacturing processes for a part",
    description=(
        "Takes a geometry feature vector and returns a ranked list of feasible "
        "manufacturing processes with calibrated probability scores and itemised "
        "cost estimates. Hard constraint violations (e.g. undercut + injection molding) "
        "are filtered before the model prediction is applied."
    ),
    responses={
        200: {"description": "Successful recommendation"},
        422: {"description": "Invalid input — feature out of range or required field missing"},
        503: {"description": "Model not loaded — service starting up or artifact missing"},
    },
)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Returns top-k process recommendations with cost breakdowns.
    """
    t_start = time.monotonic()

    if not classifier_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier model is not loaded. Please try again shortly.",
        )

    features = request.to_feature_dict()

    # Apply hard manufacturing constraints before the ML model
    feasible_processes = apply_hard_constraints(features)
    if len(feasible_processes) == 0:
        # Extremely unusual — return 422 with explanation
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No feasible manufacturing processes found for this part geometry. "
                "Check: wall_thickness_min, undercut_flag, and bounding_box dimensions."
            ),
        )

    # Get ranked process predictions
    top_k = min(request.top_k, 5)
    raw_predictions = classifier_service.predict(features, top_k=top_k + 3)

    # Filter to feasible processes only, take top-k
    filtered = [
        p for p in raw_predictions
        if p["process_code"] in feasible_processes
    ][:top_k]

    # Fetch cost estimates
    cost_ready = cost_service.is_ready()
    recommendations = []
    for pred in filtered:
        cost_breakdown = None
        if cost_ready:
            cost_features = {**features, "quantity": request.requirements.quantity or 1}
            cost_breakdown = cost_service.estimate(pred["process_code"], cost_features)

        recommendations.append(
            ProcessRecommendation(
                rank=pred["rank"],
                process_code=pred["process_code"],
                process_name=pred["process_name"],
                probability=pred["probability"],
                confidence=pred["confidence"],
                cost_estimate=cost_breakdown,
            )
        )

    latency_ms = round((time.monotonic() - t_start) * 1000, 1)
    logger.info(
        f"recommend | part_id={request.part_id or 'none'} "
        f"top_k={len(recommendations)} latency={latency_ms}ms"
    )

    return RecommendResponse(
        part_id=request.part_id,
        recommendations=recommendations,
        model_version=classifier_service.model_version(),
        latency_ms=latency_ms,
        constraint_filtered_count=len(raw_predictions) - len(filtered),
    )
