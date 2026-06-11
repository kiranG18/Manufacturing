from .recommend import router as recommend_router
from .health import router as health_router
from .debug import router as debug_router

__all__ = ["recommend_router", "health_router", "debug_router"]
