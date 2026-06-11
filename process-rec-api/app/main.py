# app/main.py
#
# FastAPI application entry point.
# Starts the app, registers all routers, and configures middleware.
# The app is designed to be stateless between requests — models are loaded
# once at startup and shared across all workers via a module-level cache.

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .routes import health, recommend, debug
from .services import classifier_service, cost_service
from .middleware.logging_middleware import LoggingMiddleware as RequestLoggingMiddleware
from .middleware.error_handler import handle_errors as add_error_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown logic.
    Models are loaded here rather than at import time so that the test suite
    can patch the load path before the app starts.
    """
    logger.info("Starting up — loading ML models...")
    t0 = time.monotonic()

    try:
        classifier_service.initialise(settings.classifier_model_path)
        logger.info(f"Classifier loaded in {time.monotonic()-t0:.2f}s")
    except FileNotFoundError:
        logger.warning(
            f"Classifier model not found at {settings.classifier_model_path}. "
            f"The /recommend endpoint will return 503 until a model is available."
        )

    try:
        cost_service.initialise(settings.cost_models_dir)
        logger.info(f"Cost models loaded in {time.monotonic()-t0:.2f}s")
    except FileNotFoundError:
        logger.warning(
            f"Cost model directory not found at {settings.cost_models_dir}. "
            f"Cost estimates will be unavailable."
        )

    logger.info(f"Service ready. Startup time: {time.monotonic()-t0:.3f}s")
    yield

    logger.info("Shutting down — releasing resources.")


app = FastAPI(
    title="Matterize Process Recommendation API",
    description=(
        "Recommends manufacturing processes and cost estimates given a part's "
        "geometry feature vector. Returns ranked process recommendations with "
        "calibrated probabilities and per-component cost breakdowns."
    ),
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow the platform frontend to call this service directly in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.matterize.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request/response logging (logs method, path, status, latency)
app.add_middleware(RequestLoggingMiddleware)

# Structured error responses for unhandled exceptions
add_error_handlers(app)

# Routers
app.include_router(health.router)
app.include_router(recommend.router, prefix="/v1")
app.include_router(debug.router, prefix="/v1")
