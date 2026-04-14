import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException, RequestValidationError

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: HTTPException):
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.warning(
        "HTTPException occured",
        extra={
            "correlation_id": correlation_id,
            "status_code": exc.status_code,
            "detail" : exc.detail,
            "endpoint": str(request.url)
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "correlation_id": correlation_id
        }
    )
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    correlation_id = getattr(request.state,"correlation_id","unknown")
    logger.warning(
        "Validation error",
        extra = {"correlation_id": correlation_id,
                 "errors" : str(exc.errors()),
                 "endpoint" : str(request.url)
        }
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail":exc.errors(),
            "correlation_id": correlation_id
        }
    )
async def unexpected_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.error(
        "Unexpected error occurred",
        extra={
            "correlation_id": correlation_id,
            "error": str(exc),
            "endpoint": str(request.url)
        },
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "correlation_id": correlation_id
        }
    )




