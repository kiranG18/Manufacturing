# app/routes/health.py

from fastapi import APIRouter
from ..services import classifier_service, cost_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Service health check",
    description="Returns service status and model readiness. Used by load balancer health probes.",
)
async def health():
    return {
        "status": "ok",
        "classifier_ready": classifier_service.is_ready(),
        "cost_model_ready": cost_service.is_ready(),
        "version": "1.3.0",
    }
