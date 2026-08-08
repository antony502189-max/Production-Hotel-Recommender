from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .config import get_settings
from .service import RecommendationService
from .storage import load_bundle


settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

REQUESTS = Counter(
    "hotel_recommender_http_requests_total",
    "HTTP requests processed by the recommendation API",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "hotel_recommender_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)


class Recommendation(BaseModel):
    hotel_id: int
    score: float = Field(ge=0.0, le=1.0)
    city: str
    hotel_type: str
    price: float = Field(gt=0)
    stars: int = Field(ge=1, le=5)


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: int
    model_version: str
    recommendations: list[Recommendation]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = load_bundle(settings.model_path)
        app.state.recommendation_service = RecommendationService(
            bundle,
            candidate_pool_size=settings.candidate_pool_size,
        )
        logger.info(
            "loaded model version=%s path=%s",
            bundle.metadata.model_version,
            settings.model_path,
        )
    except FileNotFoundError:
        app.state.recommendation_service = None
        logger.warning(
            "model artifact is missing at %s; API starts in degraded mode",
            settings.model_path,
        )
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Two-stage hotel recommendation API with hybrid retrieval and ML ranking.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(request.method, path).observe(time.perf_counter() - started)
    return response


@app.get("/health", tags=["operations"])
def health(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "recommendation_service", None)
    return {
        "status": "ok" if service is not None else "degraded",
        "model_loaded": service is not None,
        "version": __version__,
    }


@app.get("/v1/model", tags=["operations"])
def model_info(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    metadata = service.bundle.metadata
    return {
        "model_version": metadata.model_version,
        "trained_at_utc": metadata.trained_at_utc,
        "training_rows": metadata.training_rows,
        "positive_rate": metadata.positive_rate,
        "feature_names": metadata.feature_names,
        "training_fingerprint": metadata.training_fingerprint,
    }


@app.get(
    "/v1/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["recommendations"],
)
def recommendations(
    request: Request,
    user_id: int,
    k: int = Query(default=settings.default_k, ge=1, le=settings.max_k),
    exclude_seen: bool = False,
) -> RecommendationResponse:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    try:
        ranked = service.recommend(user_id=user_id, k=k, exclude_seen=exclude_seen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = [
        Recommendation(
            hotel_id=int(row.hotel_id),
            score=float(row.score),
            city=str(row.city),
            hotel_type=str(row.hotel_type),
            price=float(row.price),
            stars=int(row.stars),
        )
        for row in ranked.itertuples(index=False)
    ]
    return RecommendationResponse(
        user_id=user_id,
        model_version=service.bundle.metadata.model_version,
        recommendations=result,
    )
