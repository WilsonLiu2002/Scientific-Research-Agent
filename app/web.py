from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph import ResearchRunCancelled, build_research_graph
from app.auth import (
    SIMULATED_ACCESS_GRANTS,
    AccessGrant,
    access_level_allows,
    resolve_access_token,
)
from app.main import initial_research_state, run
from app.mcp_client import AcademicMCPClient
from app.experiments import SyntheticExperimentService
from app.materials_ml import CandidateScreenRequest, MaterialsScreeningTool
from app.memory import WorkspaceMemory
from app.observability import (
    list_local_traces,
    load_local_trace,
    persist_local_trace,
    research_run,
    utc_timestamp,
)
from app.rag import create_vector_store
from app.settings import get_chat_provider


STATIC_DIRECTORY = Path(__file__).with_name("static")
materials_tool = MaterialsScreeningTool()
workspace_memory = WorkspaceMemory()
active_cancellations: dict[str, threading.Event] = {}
active_cancellations_lock = threading.Lock()
live_traces: dict[str, dict[str, Any]] = {}
live_traces_lock = threading.Lock()


def initialize_live_trace(thread_id: str, run_id: str, trace_id: str) -> None:
    """Create the browser-visible trace buffer before background work begins."""

    with live_traces_lock:
        live_traces[thread_id] = {
            "schema_version": 1,
            "created_at": utc_timestamp(),
            "thread_id": thread_id,
            "run_id": run_id,
            "trace_id": trace_id,
            "status": "running",
            "events": [],
            "metrics": {},
        }


def append_live_event(thread_id: str, event: dict[str, Any]) -> None:
    """Append one redacted observability event from the graph worker thread."""

    with live_traces_lock:
        trace = live_traces.get(thread_id)
        if trace is not None:
            trace["events"].append(event)


def update_live_trace(
    thread_id: str, *, status: str, metrics: dict[str, Any] | None = None
) -> None:
    """Update lifecycle and performance data without replacing streamed events."""

    with live_traces_lock:
        trace = live_traces.get(thread_id)
        if trace is not None:
            trace["status"] = status
            if metrics is not None:
                trace["metrics"] = metrics


def live_trace_snapshot(thread_id: str, after: int = 0) -> dict[str, Any] | None:
    """Return an incremental, copy-safe view of one active or completed trace."""

    with live_traces_lock:
        trace = live_traces.get(thread_id)
        if trace is None:
            return None
        events = list(trace["events"])
        return {
            **{key: value for key, value in trace.items() if key != "events"},
            "events": events[max(after, 0):],
            "event_count": len(events),
        }


class ResearchRequest(BaseModel):
    """Validated web request for one unattended literature research run."""

    question: str = Field(min_length=8, max_length=1000)
    access_token: str = Field(default="sim-researcher", min_length=8, max_length=64)


class ReviewSessionRequest(ResearchRequest):
    """Browser research request with a client-known cancellation identifier."""

    thread_id: str | None = Field(default=None, min_length=8, max_length=128, pattern=r"^[A-Za-z0-9-]+$")
    project_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[a-f0-9]+$")
    chat_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[a-f0-9]+$")


class ProjectCreateRequest(BaseModel):
    """Validated request for a durable research project."""

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class ChatCreateRequest(BaseModel):
    """Validated request for one conversation inside a project."""

    title: str = Field(default="New chat", min_length=1, max_length=100)


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


def require_workspace_access(grant: AccessGrant, resource: dict[str, Any]) -> None:
    """Reject project/chat access below the resource's persisted clearance level."""

    required = str(resource.get("minimum_access_level") or "local_reader")
    if not access_level_allows(grant.level, required):
        raise HTTPException(status_code=403, detail="Authorization level cannot access this workspace")


def resolve_header_grant(token: str) -> AccessGrant:
    """Resolve one simulated request header into a server-trusted access grant."""

    try:
        return resolve_access_token(token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


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

    @application.get("/api/auth/levels")
    async def authorization_levels() -> list[dict[str, object]]:
        """List the simulated access tiers available to the browser demonstration."""

        return [grant.public_payload() for grant in SIMULATED_ACCESS_GRANTS.values()]

    @application.get("/api/projects")
    async def projects(
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> list[dict[str, Any]]:
        """List durable projects, creating the initial workspace on first use."""

        grant = resolve_header_grant(x_simulated_authorization)
        all_projects = workspace_memory.list_projects()
        visible = [
            project for project in all_projects
            if access_level_allows(grant.level, project.get("minimum_access_level", "local_reader"))
        ]
        if not visible:
            visible = [workspace_memory.create_project(
                f"{grant.label} Research", minimum_access_level=grant.level
            )]
        return visible

    @application.post("/api/projects")
    async def create_project(
        request: ProjectCreateRequest,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Create a project that may contain multiple independent chats."""

        grant = resolve_header_grant(x_simulated_authorization)
        return workspace_memory.create_project(
            request.name, request.description, minimum_access_level=grant.level
        )

    @application.get("/api/projects/{project_id}/chats")
    async def project_chats(
        project_id: str,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> list[dict[str, Any]]:
        """List all chats belonging to one project."""

        try:
            grant = resolve_header_grant(x_simulated_authorization)
            project = workspace_memory.get_project(project_id)
            require_workspace_access(grant, project)
            return [
                chat for chat in workspace_memory.list_chats(project_id)
                if access_level_allows(grant.level, chat.get("minimum_access_level", "local_reader"))
            ]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc

    @application.post("/api/projects/{project_id}/chats")
    async def create_chat(
        project_id: str,
        request: ChatCreateRequest,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Create a new conversation under an existing project."""

        try:
            grant = resolve_header_grant(x_simulated_authorization)
            project = workspace_memory.get_project(project_id)
            require_workspace_access(grant, project)
            return workspace_memory.create_chat(project_id, request.title)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc

    @application.get("/api/chats/{chat_id}")
    async def chat_detail(
        chat_id: str,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Restore a chat's ordered messages and saved structured research results."""

        try:
            grant = resolve_header_grant(x_simulated_authorization)
            chat = workspace_memory.get_chat(chat_id)
            require_workspace_access(grant, chat)
            require_workspace_access(grant, workspace_memory.get_project(chat["project_id"]))
            return chat
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc

    @application.post("/api/research")
    async def research(request: ResearchRequest) -> dict[str, Any]:
        """Run the synchronous graph off the event loop and serialize its complete state."""

        try:
            resolve_access_token(request.access_token)
            state = await asyncio.to_thread(
                run, request.question, False, None, request.access_token
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return jsonable_encoder(state)

    @application.post("/api/research/sessions")
    async def start_review_session(request: ReviewSessionRequest) -> dict[str, Any]:
        """Start a durable browser run and return at its first human-review interrupt."""

        thread_id = request.thread_id or uuid.uuid4().hex
        cancellation = threading.Event()
        try:
            grant = resolve_access_token(request.access_token)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        state = initial_research_state(
            request.question, access_profile=grant.public_payload()
        )
        try:
            project = (
                workspace_memory.get_project(request.project_id)
                if request.project_id
                else next(
                    (
                        item for item in workspace_memory.list_projects()
                        if access_level_allows(
                            grant.level, item.get("minimum_access_level", "local_reader")
                        )
                    ),
                    None,
                )
                or workspace_memory.create_project(
                    f"{grant.label} Research", minimum_access_level=grant.level
                )
            )
            chat = (
                workspace_memory.get_chat(request.chat_id, include_messages=False)
                if request.chat_id
                else workspace_memory.create_chat(project["id"])
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project or chat not found") from exc
        if chat["project_id"] != project["id"]:
            raise HTTPException(status_code=409, detail="Chat does not belong to the selected project")
        require_workspace_access(grant, project)
        require_workspace_access(grant, chat)
        memory_context, memory_stats = workspace_memory.build_context(
            project["id"], chat["id"]
        )
        state["memory_context"] = memory_context
        state["memory_stats"] = memory_stats
        state["checkpoint_thread_id"] = thread_id
        workspace_memory.begin_run(chat["id"], thread_id, state["run_id"], request.question)
        initialize_live_trace(thread_id, state["run_id"], state["trace_id"])
        with active_cancellations_lock:
            active_cancellations[thread_id] = cancellation

        def invoke() -> dict[str, Any]:
            graph = build_research_graph(
                enable_human_review=True,
                cancellation_event=cancellation,
                search_client=AcademicMCPClient(authorization_token=grant.token_id),
                vector_store=create_vector_store(access_scope=grant.level),
            )
            with research_run(
                state["run_id"],
                state["trace_id"],
                event_sink=lambda event: append_live_event(thread_id, event),
            ) as telemetry:
                result = graph.invoke(
                    state, config={"configurable": {"thread_id": thread_id}}
                )
                result["run_metrics"] = telemetry.metrics.summary(result)
            return result

        try:
            result = await asyncio.to_thread(invoke)
        except ResearchRunCancelled:
            update_live_trace(thread_id, status="cancelled")
            workspace_memory.set_chat_status(chat["id"], "cancelled")
            return {"thread_id": thread_id, "project_id": project["id"], "chat_id": chat["id"], "status": "cancelled", "review": None, "state": {"cancelled": True}}
        except Exception as exc:
            workspace_memory.set_chat_status(chat["id"], "failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            with active_cancellations_lock:
                active_cancellations.pop(thread_id, None)
        response = review_session_response(result, thread_id)
        response["project_id"] = project["id"]
        response["chat_id"] = chat["id"]
        status = "review_required" if response["review"] else "completed"
        workspace_memory.set_chat_status(chat["id"], status)
        update_live_trace(thread_id, status=status, metrics=result.get("run_metrics"))
        response["state"]["run_trace"] = live_trace_snapshot(thread_id)
        if status == "completed":
            persist_local_trace(response["state"]["run_trace"])
            stored_state = jsonable_encoder(response["state"])
            workspace_memory.complete_run(
                chat["id"],
                state["run_id"],
                str(stored_state.get("answer") or "Research completed."),
                stored_state,
            )
        return jsonable_encoder(response)

    @application.post("/api/research/sessions/{thread_id}/resume")
    async def resume_review_session(
        thread_id: str,
        decision: ReviewDecisionRequest,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Resume one durable browser review and return the next pause or final state."""

        from langgraph.types import Command

        if len(thread_id) > 128 or not thread_id.replace("-", "").isalnum():
            raise HTTPException(status_code=422, detail="Invalid checkpoint thread ID")

        cancellation = threading.Event()
        chat = workspace_memory.get_chat_by_thread(thread_id)
        request_grant = resolve_header_grant(x_simulated_authorization)
        if chat:
            require_workspace_access(request_grant, chat)
            require_workspace_access(request_grant, workspace_memory.get_project(chat["project_id"]))
        with active_cancellations_lock:
            active_cancellations[thread_id] = cancellation

        def invoke() -> dict[str, Any]:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint_graph = build_research_graph(enable_human_review=True)
            snapshot = checkpoint_graph.get_state(config)
            if not snapshot.values:
                raise KeyError(thread_id)
            run_id = snapshot.values.get("run_id") or uuid.uuid4().hex
            trace_id = snapshot.values.get("trace_id") or uuid.uuid4().hex
            profile = snapshot.values.get("access_profile") or {}
            grant = (
                resolve_access_token(profile.get("token_id"))
                if profile.get("token_id")
                else resolve_access_token("sim-researcher")
            )
            graph = build_research_graph(
                enable_human_review=True,
                cancellation_event=cancellation,
                search_client=AcademicMCPClient(authorization_token=grant.token_id),
                vector_store=create_vector_store(access_scope=grant.level),
            )
            if live_trace_snapshot(thread_id) is None:
                initialize_live_trace(thread_id, run_id, trace_id)
            update_live_trace(thread_id, status="running")
            with research_run(
                run_id,
                trace_id,
                event_sink=lambda event: append_live_event(thread_id, event),
            ) as telemetry:
                result = graph.invoke(
                    Command(resume=decision.model_dump(exclude_none=True)), config=config
                )
                result["run_metrics"] = telemetry.metrics.summary(result)
            return result

        try:
            result = await asyncio.to_thread(invoke)
        except ResearchRunCancelled:
            update_live_trace(thread_id, status="cancelled")
            if chat:
                workspace_memory.set_chat_status(chat["id"], "cancelled")
            return {"thread_id": thread_id, "status": "cancelled", "review": None, "state": {"cancelled": True}}
        except KeyError as exc:
            update_live_trace(thread_id, status="failed")
            raise HTTPException(status_code=404, detail="Review session not found") from exc
        except Exception as exc:
            update_live_trace(thread_id, status="failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            with active_cancellations_lock:
                active_cancellations.pop(thread_id, None)
        response = review_session_response(result, thread_id)
        if chat:
            response["project_id"] = chat["project_id"]
            response["chat_id"] = chat["id"]
        status = "review_required" if response["review"] else "completed"
        if chat:
            workspace_memory.set_chat_status(chat["id"], status)
        update_live_trace(thread_id, status=status, metrics=result.get("run_metrics"))
        response["state"]["run_trace"] = live_trace_snapshot(thread_id)
        if status == "completed":
            persist_local_trace(response["state"]["run_trace"])
            if chat:
                stored_state = jsonable_encoder(response["state"])
                workspace_memory.complete_run(
                    chat["id"],
                    str(stored_state.get("run_id") or uuid.uuid4().hex),
                    str(stored_state.get("answer") or "Research completed."),
                    stored_state,
                )
        return jsonable_encoder(response)

    @application.post("/api/research/sessions/{thread_id}/cancel")
    async def cancel_review_session(
        thread_id: str,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, str]:
        """Signal an active graph and its browser request to stop cooperatively."""

        chat = workspace_memory.get_chat_by_thread(thread_id)
        if chat:
            grant = resolve_header_grant(x_simulated_authorization)
            require_workspace_access(grant, chat)
        with active_cancellations_lock:
            cancellation = active_cancellations.get(thread_id)
        if cancellation is not None:
            cancellation.set()
        update_live_trace(thread_id, status="cancelling")
        if chat:
            workspace_memory.set_chat_status(chat["id"], "cancelling")
        return {"thread_id": thread_id, "status": "cancellation_requested"}

    @application.get("/api/research/sessions/{thread_id}")
    async def review_session_detail(
        thread_id: str,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Restore a paused checkpoint so reopening a chat can continue its review."""

        if len(thread_id) > 128 or not thread_id.replace("-", "").isalnum():
            raise HTTPException(status_code=422, detail="Invalid checkpoint thread ID")
        chat = workspace_memory.get_chat_by_thread(thread_id)
        if chat:
            grant = resolve_header_grant(x_simulated_authorization)
            require_workspace_access(grant, chat)
        graph = build_research_graph(enable_human_review=True)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Review session not found")
        review = None
        for task in getattr(snapshot, "tasks", ()):
            interrupts = getattr(task, "interrupts", ())
            if interrupts:
                review = interrupts[0].value
                break
        return jsonable_encoder(
            {
                "thread_id": thread_id,
                "project_id": chat["project_id"] if chat else None,
                "chat_id": chat["id"] if chat else None,
                "status": "review_required" if review else "completed",
                "review": review,
                "state": dict(snapshot.values),
            }
        )

    @application.get("/api/research/sessions/{thread_id}/trace")
    async def live_review_trace(
        thread_id: str,
        after: int = 0,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Return newly emitted trace events while a browser research run is active."""

        chat = workspace_memory.get_chat_by_thread(thread_id)
        if chat:
            grant = resolve_header_grant(x_simulated_authorization)
            require_workspace_access(grant, chat)
        trace = live_trace_snapshot(thread_id, after=min(max(after, 0), 100000))
        if trace is None:
            raise HTTPException(status_code=404, detail="Live trace not found")
        return jsonable_encoder(trace)

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
    async def materials_status(
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Describe the local scientific-ML dataset and active prediction model."""

        try:
            grant = resolve_access_token(x_simulated_authorization)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not grant.allows("materials:read"):
            raise HTTPException(status_code=403, detail="Token does not permit materials database access")
        return {
            "dataset": "Matbench v0.1 / matbench_expt_gap",
            "records": len(materials_tool.database.records()),
            "property": "experimental band gap",
            "unit": "eV",
            "model": materials_tool.predictor.model_name,
        }

    @application.post("/api/materials/screen")
    async def screen_materials(
        request: CandidateScreenRequest,
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> dict[str, Any]:
        """Run the local measured-property candidate-screening tool."""

        try:
            grant = resolve_access_token(x_simulated_authorization)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not grant.allows("materials:read"):
            raise HTTPException(status_code=403, detail="Token does not permit materials database access")
        return jsonable_encoder(await asyncio.to_thread(materials_tool.screen_candidates, request))

    @application.post("/api/materials/predict")
    async def predict_materials(
        formulas: list[str],
        x_simulated_authorization: str = Header(default="sim-researcher"),
    ) -> list[dict[str, Any]]:
        """Estimate band gaps for formulas with the local composition model."""

        try:
            grant = resolve_access_token(x_simulated_authorization)
            if not grant.allows("materials:read"):
                raise PermissionError("Token does not permit materials database access")
            predictions = await asyncio.to_thread(materials_tool.predict_band_gaps, formulas)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return jsonable_encoder(predictions)

    return application


app = create_app()
