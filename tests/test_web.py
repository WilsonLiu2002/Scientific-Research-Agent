from fastapi.testclient import TestClient
from types import SimpleNamespace
import threading

from app.models import ResearchGoal
from app.web import app
from app import web


def test_web_ui_and_health_endpoint() -> None:
    """Serve the workbench shell and readiness response."""

    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert "Research Atlas" in client.get("/").text
    assert "review-dialog" in client.get("/").text
    assert client.get("/api/health").json() == {
        "status": "ok", "llm_provider": None, "llm_model": None
    }


def test_research_endpoint_serializes_graph_state(monkeypatch) -> None:
    """Return Pydantic graph objects as browser-safe JSON."""

    def fake_run(question: str) -> dict:
        return {
            "question": question,
            "answer": "Overview\nEvidence-based answer.",
            "goal": ResearchGoal(
                objective=question,
                success_criteria=["Produce an answer"],
                status="completed",
                progress=1.0,
            ),
        }

    monkeypatch.setattr("app.web.run", fake_run)
    response = TestClient(app).post("/api/research", json={"question": "How are crystals modeled?"})

    assert response.status_code == 200
    assert response.json()["goal"]["status"] == "completed"


def test_browser_review_session_pauses_and_resumes(monkeypatch) -> None:
    """Expose checkpointed graph interrupts as browser-safe review sessions."""

    interrupt = SimpleNamespace(
        value={
            "review_type": "search_plan",
            "question": "How are crystals modeled?",
            "queries": ["crystal graph models"],
            "allowed_actions": ["approve", "edit", "add_instruction", "stop"],
        }
    )

    class FakeReviewGraph:
        def invoke(self, value, config):
            if isinstance(value, dict):
                return {**value, "__interrupt__": [interrupt]}
            return {"question": "How are crystals modeled?", "answer": "Completed"}

        def get_state(self, config):
            return SimpleNamespace(values={"question": "How are crystals modeled?"})

    graph = FakeReviewGraph()
    monkeypatch.setattr(web, "build_research_graph", lambda **kwargs: graph)
    client = TestClient(app)

    started = client.post(
        "/api/research/sessions", json={"question": "How are crystals modeled?"}
    )
    thread_id = started.json()["thread_id"]
    resumed = client.post(
        f"/api/research/sessions/{thread_id}/resume", json={"action": "approve"}
    )

    assert started.status_code == 200
    assert started.json()["status"] == "review_required"
    assert started.json()["review"]["review_type"] == "search_plan"
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "completed"
    assert resumed.json()["state"]["answer"] == "Completed"


def test_browser_cancel_endpoint_signals_active_run() -> None:
    """Let the visible stop control signal an in-flight backend operation."""

    cancellation = threading.Event()
    with web.active_cancellations_lock:
        web.active_cancellations["cancel-test"] = cancellation

    response = TestClient(app).post("/api/research/sessions/cancel-test/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancellation_requested"
    assert cancellation.is_set()
    with web.active_cancellations_lock:
        web.active_cancellations.pop("cancel-test", None)


def test_demo_experiment_endpoint_is_explicitly_synthetic() -> None:
    """Expose reproducible demo data without presenting it as published evidence."""

    response = TestClient(app).get("/api/demo/experiment")

    assert response.status_code == 200
    assert response.json()["status"] == "synthetic"


def test_materials_status_and_screening_endpoints() -> None:
    """Expose the local measured-property database through the browser API."""

    client = TestClient(app)
    status = client.get("/api/materials/status")
    result = client.post(
        "/api/materials/screen",
        json={"min_band_gap_ev": 1.2, "max_band_gap_ev": 1.8, "top_k": 3},
    )

    assert status.status_code == 200
    assert status.json()["records"] == 4604
    assert result.status_code == 200
    assert len(result.json()["candidates"]) == 3
    assert all(item["value_type"] == "measured" for item in result.json()["candidates"])


def test_material_prediction_endpoint_reports_model_provenance() -> None:
    """Return model-derived values with an explicit model identifier."""

    response = TestClient(app).post("/api/materials/predict", json=["TiO2"])

    assert response.status_code == 200
    assert response.json()[0]["model_name"] == "composition-knn-matbench-expt-gap-v1"


def test_local_trace_list_and_detail_endpoints(monkeypatch) -> None:
    """Expose redacted local traces to the existing workbench."""

    payload = {"trace_id": "a" * 32, "run_id": "run-1", "events": [], "metrics": {}}
    monkeypatch.setattr(web, "list_local_traces", lambda limit=25: [payload])
    monkeypatch.setattr(web, "load_local_trace", lambda trace_id: payload if trace_id == "a" * 32 else None)
    client = TestClient(app)

    assert client.get("/api/traces").json()[0]["run_id"] == "run-1"
    assert client.get(f"/api/traces/{'a' * 32}").json()["trace_id"] == "a" * 32
    assert client.get(f"/api/traces/{'b' * 32}").status_code == 404
