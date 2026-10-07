import asyncio
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import observability
from app.main import arun, run


class RecordHandler(logging.Handler):
    """Capture structured records without depending on console formatting."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def capture_agent_records(monkeypatch) -> RecordHandler:
    observability.configure_observability()
    handler = RecordHandler()
    logger = logging.getLogger("scientific_agent")
    monkeypatch.setattr(logger, "handlers", [handler])
    return handler


def test_nested_spans_preserve_run_trace_and_parent_ids(monkeypatch) -> None:
    handler = capture_agent_records(monkeypatch)

    with observability.research_run(run_id="run-1", trace_id="trace-1"):
        with observability.span("parent") as parent_id:
            with observability.span("child"):
                pass

    child_start = next(
        record for record in handler.records
        if getattr(record, "fields", {}).get("name") == "child"
        and getattr(record, "event", "") == "operation.started"
    )
    assert child_start.run_id == "run-1"
    assert child_start.trace_id == "trace-1"
    assert child_start.fields["parent_span_id"] == parent_id


def test_research_run_streams_redacted_events_live() -> None:
    streamed: list[dict] = []

    with observability.research_run(
        run_id="live-run", trace_id="live-trace", event_sink=streamed.append
    ):
        with observability.span("live-operation"):
            observability.log_event("provider.result", api_key="must-not-leak")

    assert [event["event"] for event in streamed] == [
        "operation.started",
        "operation.started",
        "provider.result",
        "operation.completed",
        "operation.completed",
    ]
    assert streamed[2]["api_key"] == "[REDACTED]"
    assert all(event["run_id"] == "live-run" for event in streamed)


def test_error_and_retry_metrics_are_recorded(monkeypatch) -> None:
    handler = capture_agent_records(monkeypatch)

    with observability.research_run() as telemetry:
        observability.metrics_increment("retries")
        with pytest.raises(ValueError):
            with observability.span("failing-operation"):
                raise ValueError("provider unavailable")
        summary = telemetry.metrics.summary({"errors": []})

    assert summary["retries"] == 1
    assert summary["errors"] == 1
    assert any(getattr(record, "event", "") == "operation.failed" for record in handler.records)


def test_concurrent_subtasks_keep_trace_but_receive_distinct_spans(monkeypatch) -> None:
    capture_agent_records(monkeypatch)

    with observability.research_run(run_id="concurrent", trace_id="shared"):
        def work(subtask_id: str) -> tuple[str | None, str | None, str]:
            with observability.span("subtask", fields={"subtask_id": subtask_id}) as span_id:
                fields = observability.correlation_fields()
                return fields["run_id"], fields["trace_id"], span_id

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(contextvars.copy_context().run, work, subtask_id)
                for subtask_id in ("task-a", "task-b")
            ]
            results = [future.result() for future in futures]

    assert {item[0] for item in results} == {"concurrent"}
    assert {item[1] for item in results} == {"shared"}
    assert len({item[2] for item in results}) == 2


def test_secret_fields_are_redacted() -> None:
    payload = observability.redact(
        {"api_key": "top-secret", "nested": {"authorization": "Bearer secret"}, "model": "qwen"}
    )

    assert payload == {
        "api_key": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]"},
        "model": "qwen",
    }


def test_local_trace_persistence_and_loading(tmp_path) -> None:
    payload = {
        "run_id": "run-local",
        "trace_id": "a" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
        "metrics": {"errors": 0},
        "events": [],
    }

    path = observability.persist_local_trace(payload, tmp_path)

    assert path is not None and path.exists()
    assert observability.load_local_trace("a" * 32, tmp_path) == payload
    assert observability.list_local_traces(tmp_path)[0]["run_id"] == "run-local"
    assert observability.load_local_trace("../unsafe", tmp_path) is None


def test_local_trace_storage_failure_never_breaks_workflow(monkeypatch, tmp_path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")

    result = observability.persist_local_trace(
        {"trace_id": "b" * 32, "events": []}, blocked
    )

    assert result is None


class FakeGraph:
    def invoke(self, state):
        return {**state, "search_iteration": 1}

    async def ainvoke(self, state):
        return {**state, "search_iteration": 1}


def test_sync_and_async_entrypoints_return_metrics(monkeypatch) -> None:
    monkeypatch.setattr("app.main.build_research_graph", lambda **kwargs: FakeGraph())

    sync_result = run("A sufficiently long research question")
    async_result = asyncio.run(arun("A sufficiently long research question"))

    assert sync_result["run_id"] and sync_result["trace_id"]
    assert async_result["run_id"] and async_result["trace_id"]
    assert "total_duration_ms" in sync_result["run_metrics"]
    assert "total_duration_ms" in async_result["run_metrics"]
    assert sync_result["run_trace"]["trace_id"] == sync_result["trace_id"]
    assert async_result["run_trace"]["trace_id"] == async_result["trace_id"]
