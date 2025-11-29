import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure all SQLAlchemy models are registered with Base.metadata
# Required before create_tables() or any ORM operations
import app.models  # noqa: F401
from app.api.v1.endpoints.ai_router import router as ai_router
from app.api.v1.endpoints.auth_router import router as auth_router
from app.api.v1.endpoints.conversation_router import router as conversation_router
from app.api.v1.endpoints.memory_router import router as memory_router
from app.api.v1.endpoints.room_router import router as rooms_router
from app.core.arq_pool import close_arq_pool, create_arq_pool
from app.core.config import settings
from app.core.constants import CORS_ALLOWED_ORIGINS_DEV
from app.core.database import create_tables
from app.core.exceptions import (
    DomainException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)
from app.core.redis_client import close_redis_client, create_redis_client

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    """
    logger.info("fastapi_starting")

    await create_tables()
    logger.info("database_tables_created")

    await create_redis_client()
    logger.info("redis_client_initialized")

    await create_arq_pool()
    logger.info("arq_pool_initialized")

    yield

    await close_arq_pool()
    await close_redis_client()
    logger.info("fastapi_shutdown_complete")


app = FastAPI(
    title=settings.app_name,
    description="Virtual meeting space with 3 type chat system",
    docs_url="/docs",
    lifespan=lifespan,
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS_DEV,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Exception Handlers (Convert Domain Exceptions → HTTP Responses)
# ============================================================================


@app.exception_handler(NotFoundException)
async def not_found_exception_handler(request: Request, exc: NotFoundException) -> JSONResponse:
    """Handle resource not found exceptions."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(UnauthorizedException)
async def unauthorized_exception_handler(request: Request, exc: UnauthorizedException) -> JSONResponse:
    """Handle authentication exceptions."""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(ForbiddenException)
async def forbidden_exception_handler(request: Request, exc: ForbiddenException) -> JSONResponse:
    """Handle authorization exceptions."""
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException) -> JSONResponse:
    """Handle validation exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(DomainException)
async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
    """Fallback handler for all other domain exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ============================================================================
# Router Registration
# ============================================================================

API_V1_PREFIX = "/api/v1"

app.include_router(rooms_router, prefix=API_V1_PREFIX)
app.include_router(auth_router, prefix=API_V1_PREFIX)
app.include_router(conversation_router, prefix=API_V1_PREFIX)
app.include_router(ai_router, prefix=API_V1_PREFIX)
app.include_router(memory_router, prefix=API_V1_PREFIX)


@app.get("/")
def root():
    return {
        "message": "Welcome to The Gathering API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "rooms": "/api/v1/rooms",
            "room_health": "/api/v1/rooms/health/check",
        },
    }


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/ai")
async def ai_health_check():
    """AI features health check with Redis and OpenAI status."""
    from app.core.arq_pool import get_arq_pool

    health_status = {
        "ai_features_enabled": settings.ai_features_enabled,
        "openai_configured": settings.openai_api_key is not None,
        "redis_connected": False,
        "arq_worker_available": False,
    }

    # Check Redis connection
    arq_pool = get_arq_pool()
    if arq_pool:
        try:
            await arq_pool.ping()
            health_status["redis_connected"] = True

            # Check if ARQ worker is processing jobs
            info = await arq_pool.info()
            health_status["arq_worker_available"] = info is not None
            health_status["arq_info"] = info
        except Exception as e:
            health_status["redis_error"] = str(e)

    # Overall status
    health_status["status"] = (
        "healthy" if health_status["redis_connected"] and health_status["openai_configured"] else "degraded"
    )

    return health_status


@app.get("/test")
def endpoint_test():
    return {"status": "FastAPI works!", "project": "The Gathering"}


if __name__ == "__main__":
    print("API Documentation: http://localhost:8000/docs")
    print("Room Endpoint: http://localhost:8000/api/v1/rooms")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
