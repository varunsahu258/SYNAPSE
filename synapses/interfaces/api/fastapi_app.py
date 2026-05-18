"""FastAPI adapter for SYNAPSES use-cases."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
if importlib.util.find_spec("prometheus_client") is not None:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
else:  # pragma: no cover - exercised only in minimal dependency environments
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _Metric:
        def labels(self, *_args: object) -> "_Metric":
            return self

        def inc(self) -> None:
            return None

        def observe(self, _value: float) -> None:
            return None

    def Counter(*_args: object, **_kwargs: object) -> _Metric:
        return _Metric()

    def Histogram(*_args: object, **_kwargs: object) -> _Metric:
        return _Metric()

    def generate_latest() -> bytes:
        return b"# prometheus_client not installed\n"

from synapses.application.services import ExperimentService, SimulationService
from synapses.ai.director import LLMDirector, RLDirectorAdapter, RewardWeights, evaluate_trained_model, train_director_ppo
from synapses.director import DirectorAI
from synapses.environment import Environment
from synapses.experiments import (
    CounterfactualEngine,
    ExperimentRunRecord,
    ExperimentRunner,
    ExperimentSpec,
    aggregate_runs,
    build_agents,
    build_comparison_report,
    export_records_csv,
    parameter_sweep_grid,
)
if importlib.util.find_spec("sqlalchemy") is not None:
    from synapses.persistence.db import DatabaseSettings, build_engine, build_session_factory
    from synapses.persistence.models import Base
    from synapses.persistence.service import PersistenceService
else:  # pragma: no cover - minimal dependency environments
    DatabaseSettings = build_engine = build_session_factory = Base = PersistenceService = None

LOG_LEVEL = os.getenv("SYNAPSES_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("synapses_http_requests_total", "HTTP requests handled by SYNAPSES", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("synapses_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
SIMULATION_RUNS = Counter("synapses_simulation_runs_total", "Simulation runs started", ["director_mode", "status"])
RL_TRAINING_RUNS = Counter("synapses_rl_training_runs_total", "RL director training requests", ["status"])


def _csv_env(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


ALLOWED_ORIGINS = _csv_env("SYNAPSES_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173")
API_KEYS = set(_csv_env("SYNAPSES_API_KEYS"))
AUTH_REQUIRED = bool(API_KEYS)
ARTIFACT_DIR = Path(os.getenv("SYNAPSES_ARTIFACT_DIR", "/tmp/synapses-artifacts"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL")
RL_TRAIN_WINDOW_SECONDS = int(os.getenv("SYNAPSES_RL_TRAIN_WINDOW_SECONDS", "60"))
RL_TRAIN_MAX_REQUESTS = int(os.getenv("SYNAPSES_RL_TRAIN_MAX_REQUESTS", "2"))
_rl_train_requests: dict[str, list[float]] = {}

app = FastAPI(title="SYNAPSES Simulation API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        REQUEST_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
        REQUEST_COUNT.labels(request.method, path, status).inc()


class SimulationRequest(BaseModel):
    num_agents: int = Field(..., ge=1)
    steps: int = Field(..., ge=0)
    tax_rate: float = Field(0.0, ge=0.0, le=1.0)
    gini_threshold: float = Field(0.4, ge=0.0, le=1.0)
    satisfaction_threshold: float = Field(40.0, ge=0.0, le=100.0)
    crime_threshold: float = Field(50.0, ge=0.0, le=100.0)
    director_mode: str = Field("rule_based")
    rl_model_path: str | None = None


class SimulationResponse(BaseModel):
    metrics_over_time: list[dict[str, Any]]
    grid_state: dict[str, Any]
    run_uid: str | None = None


class ExperimentResponse(BaseModel):
    experiments: dict[str, dict[str, Any]]
    comparison: dict[str, dict[str, Any]]


class SweepRequest(BaseModel):
    num_agents: list[int] = Field(default_factory=lambda: [2, 4, 8])
    steps: list[int] = Field(default_factory=lambda: [5, 10])
    tax_rate: list[float] = Field(default_factory=lambda: [0.0, 0.25])
    runs_per_spec: int = Field(2, ge=1, le=50)


class CounterfactualRequest(BaseModel):
    num_agents: int = Field(..., ge=1)
    steps: int = Field(..., ge=0)
    tax_rate: float = Field(0.0, ge=0.0, le=1.0)


class RLTrainRequest(BaseModel):
    output_dir: str | None = None
    total_timesteps: int = Field(5000, ge=1000, le=1_000_000)
    episode_length: int = Field(100, ge=10)
    seed: int = 42
    stability: float = 1.8
    inequality_penalty: float = 1.3
    suffering_penalty: float = 1.4
    crime_penalty: float = 1.2
    sustainability: float = 1.0


_simulation_service = SimulationService()
_experiment_service = ExperimentService()
_session_factory: Any | None = None
if os.getenv("SYNAPSES_DATABASE_URL") and DatabaseSettings is not None and Base is not None:
    try:
        engine = build_engine(DatabaseSettings(url=os.environ["SYNAPSES_DATABASE_URL"]))
        Base.metadata.create_all(engine)
        _session_factory = build_session_factory(engine)
        logger.info("Persistence enabled")
    except Exception:
        logger.exception("Persistence initialization failed; continuing without database writes")


def require_api_key(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> None:
    if not AUTH_REQUIRED:
        return
    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Valid API key required")


def _client_id(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")


def enforce_rl_rate_limit(request: Request) -> None:
    now = time.time()
    key = _client_id(request)
    recent = [ts for ts in _rl_train_requests.get(key, []) if now - ts < RL_TRAIN_WINDOW_SECONDS]
    if len(recent) >= RL_TRAIN_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="RL training rate limit exceeded")
    recent.append(now)
    _rl_train_requests[key] = recent


def _spatial_state(simulation: Any) -> dict[str, Any]:
    world = simulation.grid_world
    return {
        "width": world.width,
        "height": world.height,
        "cell_size": world.cell_size,
        "agents": [{"agent_id": str(index), "position": list(agent.position)} for index, agent in enumerate(simulation.agents)],
        "cells": [{"coord": list(coord), "resource": cell.resource, "crime": cell.crime} for coord, cell in world._cells.items()],
    }


def _build_director(request: SimulationRequest) -> DirectorAI:
    if request.director_mode == "llm":
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured on the API server")
        return LLMDirector(api_key=OPENROUTER_API_KEY, site_url=OPENROUTER_SITE_URL)
    if request.director_mode == "rl":
        if not request.rl_model_path:
            raise HTTPException(status_code=400, detail="rl_model_path is required for RL director mode")
        return RLDirectorAdapter(model_path=str(_validated_artifact_path(request.rl_model_path)))
    if request.director_mode != "rule_based":
        raise HTTPException(status_code=400, detail="director_mode must be one of: rule_based, llm, rl")
    return DirectorAI(request.gini_threshold, request.satisfaction_threshold, request.crime_threshold)


def _build_simulation(request: SimulationRequest) -> Any:
    simulation = _simulation_service.build_simulation(request.num_agents, request.tax_rate)
    simulation.director = _build_director(request)
    return simulation


def _numeric_metrics(metrics: dict[str, Any]) -> Iterable[tuple[str, float]]:
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield key, float(value)


def _persist_run(run_uid: str, request: SimulationRequest, metrics: list[dict[str, Any]], status: str) -> None:
    if _session_factory is None:
        return
    config = request.model_dump()
    try:
        with _session_factory() as session:
            service = PersistenceService(session)
            config_version = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(config, sort_keys=True)).hex[:12]
            service.store_config_version("run_simulation", config_version, config)
            run = service.create_run(run_uid=run_uid, experiment_name="run_simulation", seed=0, config_version=config_version)
            for row in metrics:
                step = int(row.get("step", 0))
                for key, value in _numeric_metrics(row):
                    service.store_metric(run.id, step, key, value)
                for intervention in row.get("interventions", []) or []:
                    service.store_intervention(run.id, step, str(intervention.get("action", "unknown")), dict(intervention))
            service.store_snapshot(run.id, int(metrics[-1].get("step", 0)) if metrics else 0, json.dumps({"metrics_over_time": metrics}).encode())
            service.finalize_run(run.id, status)
            service.build_reproducibility_record(run_uid)
            session.commit()
    except Exception:
        logger.exception("Failed to persist simulation run", extra={"run_uid": run_uid})


def _artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR.resolve()


def _validated_artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _artifact_dir() / candidate
    resolved = candidate.resolve()
    root = _artifact_dir()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="model_path must stay within SYNAPSES_ARTIFACT_DIR")
    return resolved


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/run_simulation", response_model=SimulationResponse, dependencies=[Depends(require_api_key)])
def run_simulation(request: SimulationRequest) -> SimulationResponse:
    run_uid = uuid.uuid4().hex
    logger.info("Starting simulation", extra={"run_uid": run_uid, "director_mode": request.director_mode})
    try:
        simulation = _build_simulation(request)
        metrics = simulation.run(request.steps)
    except HTTPException:
        SIMULATION_RUNS.labels(request.director_mode, "failed").inc()
        logger.exception("Simulation rejected", extra={"run_uid": run_uid})
        raise
    except Exception as exc:
        SIMULATION_RUNS.labels(request.director_mode, "failed").inc()
        logger.exception("Simulation failed", extra={"run_uid": run_uid})
        raise HTTPException(status_code=500, detail="simulation failed") from exc
    _persist_run(run_uid, request, metrics, "completed")
    SIMULATION_RUNS.labels(request.director_mode, "completed").inc()
    return SimulationResponse(metrics_over_time=metrics, grid_state=_spatial_state(simulation), run_uid=run_uid)


@app.post("/grid_state", dependencies=[Depends(require_api_key)])
def get_grid_state(request: SimulationRequest) -> dict[str, Any]:
    simulation = _build_simulation(request)
    simulation.run(request.steps)
    return _spatial_state(simulation)


@app.websocket("/ws/run_simulation")
async def stream_simulation(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        request = SimulationRequest(**payload)
        simulation = _build_simulation(request)
    except (ValidationError, HTTPException) as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1008)
        return
    metrics_over_time: list[dict[str, Any]] = []
    for step_number in range(1, request.steps + 1):
        metrics = await asyncio.to_thread(simulation._run_step, step_number)
        metrics_over_time.append(metrics)
        await websocket.send_json({"type": "step", "metrics": metrics, "spatial": _spatial_state(simulation)})
    await websocket.send_json({"type": "complete", "metrics_over_time": metrics_over_time, "spatial": _spatial_state(simulation)})
    await websocket.close()


@app.post("/run_experiment", response_model=ExperimentResponse, dependencies=[Depends(require_api_key)])
def run_experiment_endpoint(request: SimulationRequest) -> ExperimentResponse:
    return ExperimentResponse(**_experiment_service.run_experiment(request.num_agents, request.steps, request.tax_rate))


@app.post("/experiments/parameter_sweep", dependencies=[Depends(require_api_key)])
def run_parameter_sweep(request: SweepRequest) -> dict[str, Any]:
    grid = parameter_sweep_grid({"num_agents": request.num_agents, "steps": request.steps, "tax_rate": request.tax_rate})
    specs = [ExperimentSpec(experiment_id=f"sweep_{idx}", parameters=params, seed=42 + idx) for idx, params in enumerate(grid)]
    runner = ExperimentRunner(lambda params: _experiment_service.run_experiment(int(params["num_agents"]), int(params["steps"]), float(params["tax_rate"]))["comparison"]["director_based"])
    records: list[ExperimentRunRecord] = runner.run_batch(specs, request.runs_per_spec)
    summaries = aggregate_runs(records)
    csv_path = export_records_csv(records, _artifact_dir() / "parameter_sweep.csv")
    return {"report": build_comparison_report(summaries), "csv_path": str(csv_path), "records": len(records)}


@app.post("/counterfactual/run", dependencies=[Depends(require_api_key)])
def run_counterfactual(request: CounterfactualRequest) -> dict[str, Any]:
    agents = build_agents(request.num_agents, request.tax_rate)
    engine = CounterfactualEngine(base_agents=agents, base_environment=Environment())
    baseline = engine.create_branch("baseline")
    policy = engine.create_branch("policy")
    policy.add_intervention(1, lambda env, _agents, _rng: setattr(env, "food_supply", env.food_supply + 10))
    engine.run_all(request.steps)
    return {"baseline_steps": len(baseline.metrics()), "policy_steps": len(policy.metrics()), "comparison": engine.compare("baseline")}


@app.post("/director/rl/train", dependencies=[Depends(require_api_key)])
def train_director_endpoint(request: RLTrainRequest, raw_request: Request) -> dict[str, Any]:
    enforce_rl_rate_limit(raw_request)
    weights = RewardWeights(request.stability, request.inequality_penalty, request.suffering_penalty, request.crime_penalty, request.sustainability)
    output_dir = _validated_artifact_path(request.output_dir or "rl_runs")
    try:
        model_path = train_director_ppo(output_dir=output_dir, total_timesteps=request.total_timesteps, episode_length=request.episode_length, seed=request.seed, reward_weights=weights)
    except Exception:
        RL_TRAINING_RUNS.labels("failed").inc()
        logger.exception("RL director training failed")
        raise
    RL_TRAINING_RUNS.labels("completed").inc()
    return {"model_path": str(model_path)}


@app.post("/director/rl/evaluate", dependencies=[Depends(require_api_key)])
def evaluate_director_endpoint(model_path: str, episodes: int = 3, episode_length: int = 100) -> dict[str, Any]:
    return evaluate_trained_model(model_path=_validated_artifact_path(model_path), episodes=episodes, episode_length=episode_length)
