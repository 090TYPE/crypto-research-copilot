"""Global exception handlers — turn upstream/logic failures into clean JSON.

Keeps the service from leaking tracebacks or 500-ing on a flaky external API.
"""

import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("copilot")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(httpx.HTTPError)
    async def _upstream_error(request: Request, exc: httpx.HTTPError) -> JSONResponse:
        logger.warning("upstream error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=502,
            content={"detail": "Upstream data source unavailable", "error": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def _bad_input(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
