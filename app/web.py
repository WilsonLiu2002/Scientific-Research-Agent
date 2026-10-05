from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph import ResearchRunCancelled, build_research_graph
from app.main import initial_research_state, run
from app.experiments import SyntheticExperimentService
from app.materials_ml import CandidateScreenRequest, MaterialsScreeningTool
from app.observability import list_local_traces, load_local_trace
from app.settings import get_chat_provider


STATIC_DIRECTORY = Path(__file__).with_name("static")
materials_tool = MaterialsScreeningTool()
active_cancellations: dict[str, threading.Event] = {}
active_cancellations_lock = threading.Lock()


class ResearchRequest(BaseModel):
    """Validated web request for one unattended literature research run."""

    question: str = Field(min_length=8, max_length=1000)


class ReviewSessionRequest(ResearchRequest):
    """Browser research request with a client-known cancellation identifier."""

    thread_id: str | None = Field(default=None, min_length=8, max_length=128, pattern=r"^[A-Za-z0-9-]+$")


class ReviewDecisionRequest(BaseModel):
    """Validated browser decision used to resume one checkpointed review."""

    action: str = Field(min_length=3, max_length=32)
    instruction: str | None = Field(default=None, max_length=1000)
    queries: list[str] = Field(default_factory=list, max_length=5)
    paper_ids: list[str] = Field(default_factory=list, max_length=10)


def review_session_response(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Separate an interrupt from graph state for a stable browser-session response."""

    interrupts = result.get("__interrupt__", [])
    review = interrupts[0].value if interrupts else None
    graph_state = {key: value for key, value in result.items() if key != "__interrupt__"}
    return {
        "thread_id": thread_id,
        "status": "review_required" if review else "completed",
        "review": review,
        "state": graph_state,
    }


def create_app() -> FastAPI:
    """Create the browser UI and JSON API around the existing research graph."""

    application = FastAPI(title="Scientific Literature Research Agent")
    application.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the research workbench shell."""

        return FileResponse(STATIC_DIRECTORY / "index.html")

    @application.get("/api/health")
    async def health() -> dict[str, str | None]:
        """Expose a minimal readiness probe for local and hosted deployments."""

        provider = get_chat_provider()
        return {
            "status": "ok",
            "llm_provider": provider.provider if provider else None,
            "llm_model": provider.model if provider else None,
        }

    @application.post("/api/research")
    async def research(request: ResearchRequest) -> dict[str, Any]:
        """Run the synchronous graph off the event loop and serialize its complete state."""

        try:
            state = await asyncio.to_thread(run, request.question)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return jsonable_encoder(state)

    @application.post("/api/research/sessions")
    async def start_review_session(request: ReviewSessionRequest) -> dict[str, Any]:
        """Start a durable browser run and return at its first human-review interrupt."""

        thread_id = request.thread_id or uuid.uuid4().hex
        cancellation = threading.Event()
        with active_cancellations_lock:
            active_cancellations[thread_id] = cancellation

        def invoke() -> dict[str, Any]:
            graph = build_research_graph(
                enable_human_review=True, cancellation_event=cancellation
            )
            state = initial_research_state(request.question)
            state["checkpoint_thread_id"] = thread_id
            return graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

        try:
            result = await asyncio.to_thread(invoke)
        except ResearchRunCancelled:
            return {"thread_id": thread_id, "status": "cancelled", "review": None, "state": {"cancelled": True}}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            with active_cancellations_lock:
                active_cancellations.pop(thread_id, None)
        return jsonable_encoder(review_session_response(result, thread_id))

    @application.post("/api/research/sessions/{thread_id}/resume")
    async def resume_review_session(
        thread_id: str, decision: ReviewDecisionRequest
    ) -> dict[str, Any]:
        """Resume one durable browser review and return the next pause or final state."""

        from langgraph.types import Command

        if len(thread_id) > 128 or not thread_id.replace("-", "").isalnum():
            raise HTTPException(status_code=422, detail="Invalid checkpoint thread ID")

        cancellation = threading.Event()
        with active_cancellations_lock:
            active_cancellations[thread_id] = cancellation

        def invoke() -> dict[str, Any]:
            graph = build_research_graph(
                enable_human_review=True, cancellation_event=cancellation
            )
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise KeyError(thread_id)
            return graph.invoke(
                Command(resume=decision.model_dump(exclude_none=True)), config=config
            )

        try:
            result = await asyncio.to_thread(invoke)
        except ResearchRunCancelled:
            return {"thread_id": thread_id, "status": "cancelled", "review": None, "state": {"cancelled": True}}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Review session not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            with active_cancellations_lock:
                active_cancellations.pop(thread_id, None)
        return jsonable_encoder(review_session_response(result, thread_id))

    @application.post("/api/research/sessions/{thread_id}/cancel")
    async def cancel_review_session(thread_id: str) -> dict[str, str]:
        """Signal an active graph and its browser request to stop cooperatively."""

        with active_cancellations_lock:
            cancellation = active_cancellations.get(thread_id)
        if cancellation is not None:
            cancellation.set()
        return {"thread_id": thread_id, "status": "cancellation_requested"}

    @application.get("/api/demo/experiment")
    async def demo_experiment() -> dict[str, Any]:
        """Return a deterministic local experiment for visual demonstration."""

        return SyntheticExperimentService().run_crystal_benchmark()

    @application.get("/api/traces")
    async def traces(limit: int = 25) -> list[dict[str, Any]]:
        """List recent redacted local research traces."""

        return list_local_traces(limit=min(max(limit, 1), 100))

    @application.get("/api/traces/{trace_id}")
    async def trace_detail(trace_id: str) -> dict[str, Any]:
        """Return one redacted local trace for workbench inspection."""

        trace = load_local_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @application.get("/api/materials/status")
    async def materials_status() -> dict[str, Any]:
        """Describe the local scientific-ML dataset and active prediction model."""

        return {
            "dataset": "Matbench v0.1 / matbench_expt_gap",
            "records": len(materials_tool.database.records()),
            "property": "experimental band gap",
            "unit": "eV",
            "model": materials_tool.predictor.model_name,
        }

    @application.post("/api/materials/screen")
    async def screen_materials(request: CandidateScreenRequest) -> dict[str, Any]:
        """Run the local measured-property candidate-screening tool."""

        return jsonable_encoder(await asyncio.to_thread(materials_tool.screen_candidates, request))

    @application.post("/api/materials/predict")
    async def predict_materials(formulas: list[str]) -> list[dict[str, Any]]:
        """Estimate band gaps for formulas with the local composition model."""

        try:
            predictions = await asyncio.to_thread(materials_tool.predict_band_gaps, formulas)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return jsonable_encoder(predictions)

    return application


app = create_app()
