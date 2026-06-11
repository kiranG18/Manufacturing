# app/middleware/logging_middleware.py
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("process_rec_api.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP middleware that logs incoming request details, response status,
    and execution latency in milliseconds.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        t_start = time.monotonic()
        path = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            latency = (time.monotonic() - t_start) * 1000
            status_code = response.status_code
            logger.info(
                f"{method} {path} | Status: {status_code} | Latency: {latency:.2f}ms"
            )
            # Add execution header
            response.headers["X-Process-Latency-MS"] = f"{latency:.2f}"
            return response
        except Exception as e:
            latency = (time.monotonic() - t_start) * 1000
            logger.error(
                f"{method} {path} | FAILED | Error: {str(e)} | Latency: {latency:.2f}ms",
                exc_info=True
            )
            raise e
