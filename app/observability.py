from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar


_run_id = contextvars.ContextVar("research_run_id", default=None)
_trace_id = contextvars.ContextVar("research_trace_id", default=None)
_span_id = contextvars.ContextVar("research_span_id", default=None)
_collector = contextvars.ContextVar("research_metrics", default=None)
_configured = False
_T = TypeVar("_T")
SECRET_FRAGMENTS = ("api_key", "authorization", "credential", "password", "secret", "token")


class JsonFormatter(logging.Formatter):
    """Render one structured JSON object while excluding secret-bearing fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "severity": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
            "trace_id": getattr(record, "trace_id", None),
            "span_id": getattr(record, "span_id", None),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(redact(payload), default=str, separators=(",", ":"))


def configure_observability() -> None:
    """Configure the agent logger once from environment variables."""

    global _configured
    if _configured:
        return
    logger = logging.getLogger("scientific_agent")
    logger.setLevel(os.getenv("OBSERVABILITY_LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler()
    if env_flag("OBSERVABILITY_JSON_LOGS", True):
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [handler]
    logger.propagate = False
    _configured = True


@dataclass
class RunMetrics:
    """Thread-safe counters and totals for one complete research execution."""

    started_at: float = field(default_factory=time.perf_counter)
    counters: Counter[str] = field(default_factory=Counter)
    values: dict[str, Any] = field(default_factory=dict)
    token_usage: Counter[str] = field(default_factory=Counter)
    estimated_cost_usd: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def set(self, name: str, value: Any) -> None:
        with self._lock:
            self.values[name] = value

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self.token_usage["input_tokens"] += input_tokens
            self.token_usage["output_tokens"] += output_tokens

    def record_event(self, event: dict[str, Any]) -> None:
        """Append one already-redacted event from any worker thread."""

        with self._lock:
            self.events.append(event)

    def trace_payload(
        self, run_id: str, trace_id: str, summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the complete local trace document returned by the API and UI."""

        with self._lock:
            events = list(self.events)
        return {
            "schema_version": 1,
            "run_id": run_id,
            "trace_id": trace_id,
            "created_at": events[0]["timestamp"] if events else utc_timestamp(),
            "metrics": summary,
            "events": events,
        }

    def summary(self, state: dict[str, Any]) -> dict[str, Any]:
        goal = state.get("goal")
        return {
            "total_duration_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
            "llm_calls": self.counters["llm_calls"],
            "tool_calls": self.counters["tool_calls"],
            "search_iterations": state.get("search_iteration", 0),
            "papers_discovered": len(state.get("papers", [])),
            "papers_selected": len(state.get("retrieved_papers", [])),
            "evidence_passages": len(state.get("retrieved_passages", [])),
            "deep_read_operations": self.counters["deep_read_operations"],
            "input_tokens": self.token_usage["input_tokens"],
            "output_tokens": self.token_usage["output_tokens"],
            "estimated_cost_usd": self.estimated_cost_usd,
            "retries": self.counters["retries"],
            "errors": self.counters["errors"] + len(state.get("errors", [])),
            "final_goal_status": getattr(goal, "status", None),
            **self.values,
        }


@dataclass(frozen=True)
class RunContext:
    """Stable correlation identifiers and metrics for one research run."""

    run_id: str
    trace_id: str
    metrics: RunMetrics


@contextlib.contextmanager
def research_run(run_id: str | None = None, trace_id: str | None = None) -> Iterator[RunContext]:
    """Establish root correlation context shared by sync, async, and nested operations."""

    configure_observability()
    context = RunContext(run_id or uuid.uuid4().hex, trace_id or uuid.uuid4().hex, RunMetrics())
    tokens = (_run_id.set(context.run_id), _trace_id.set(context.trace_id), _collector.set(context.metrics))
    try:
        with span("research.run", kind="chain", export=True):
            yield context
    finally:
        _collector.reset(tokens[2])
        _trace_id.reset(tokens[1])
        _run_id.reset(tokens[0])


@contextlib.contextmanager
def span(
    name: str,
    *,
    kind: str = "operation",
    export: bool = True,
    fields: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Record a nested local span; `export` remains for API compatibility."""

    configure_observability()
    started = time.perf_counter()
    span_identifier = uuid.uuid4().hex[:16]
    parent_span_id = _span_id.get()
    token = _span_id.set(span_identifier)
    log_event("operation.started", name=name, kind=kind, parent_span_id=parent_span_id, **(fields or {}))
    try:
        yield span_identifier
    except Exception as exc:
        interrupted = "interrupt" in type(exc).__name__.lower()
        if not interrupted:
            metrics_increment("errors")
        log_event(
            "operation.interrupted" if interrupted else "operation.failed",
            severity=logging.INFO if interrupted else logging.ERROR,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            status="interrupted" if interrupted else "error",
            error_type=type(exc).__name__,
            error=None if interrupted else str(exc),
        )
        raise
    else:
        log_event(
            "operation.completed",
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            status="ok",
        )
    finally:
        _span_id.reset(token)


def observe_node(name: str, operation: Callable[[dict[str, Any]], _T]) -> Callable[[dict[str, Any]], _T]:
    """Wrap a LangGraph node with one correlated local operation span."""

    def wrapped(state: dict[str, Any]) -> _T:
        with span(f"langgraph.node.{name}", kind="chain", export=False, fields={"node": name}):
            return operation(state)

    return wrapped


def observe_route(name: str, operation: Callable[[dict[str, Any]], str]) -> Callable[[dict[str, Any]], str]:
    """Log a conditional routing decision with its selected branch."""

    def wrapped(state: dict[str, Any]) -> str:
        with span(f"langgraph.route.{name}", kind="chain", export=False, fields={"node": name}):
            decision = operation(state)
            log_event("routing.decision", node=name, decision=decision)
            return decision

    return wrapped


def run_in_context(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Execute work in a copied context so concurrent subtasks retain correlation IDs."""

    return contextvars.copy_context().run(function, *args, **kwargs)


def correlation_fields() -> dict[str, str | None]:
    return {"run_id": _run_id.get(), "trace_id": _trace_id.get(), "parent_span_id": _span_id.get()}


def metrics_increment(name: str, amount: int = 1) -> None:
    collector = _collector.get()
    if collector:
        collector.increment(name, amount)


def metrics_set(name: str, value: Any) -> None:
    collector = _collector.get()
    if collector:
        collector.set(name, value)


def record_token_usage(response: Any) -> None:
    """Record provider usage metadata without inspecting prompt or response content."""

    collector = _collector.get()
    usage = getattr(response, "usage", None)
    if collector and usage:
        collector.add_tokens(
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )


def observed_chat_completion(client: Any, operation: str, **kwargs: Any) -> Any:
    """Call an OpenAI-compatible model while recording metadata, never message content."""

    model = str(kwargs.get("model", "unknown"))
    with span(f"llm.{operation}", kind="llm", fields={"operation": operation, "model": model}):
        metrics_increment("llm_calls")
        response = client.chat.completions.create(**kwargs)
        record_token_usage(response)
        log_event("llm.result", operation=operation, model=model, retry_count=0, status="ok")
        return response


def current_metrics() -> RunMetrics | None:
    return _collector.get()


def log_event(event: str, severity: int = logging.INFO, **fields: Any) -> None:
    """Emit a redacted structured event with active correlation identifiers."""

    configure_observability()
    safe_fields = redact(fields)
    timestamp = utc_timestamp()
    collector = _collector.get()
    if collector:
        collector.record_event(
            {
                "timestamp": timestamp,
                "severity": logging.getLevelName(severity),
                "event": event,
                "run_id": _run_id.get(),
                "trace_id": _trace_id.get(),
                "span_id": _span_id.get(),
                **safe_fields,
            }
        )
    logging.getLogger("scientific_agent").log(
        severity,
        event,
        extra={
            "event": event,
            "run_id": _run_id.get(),
            "trace_id": _trace_id.get(),
            "span_id": _span_id.get(),
            "fields": safe_fields,
        },
    )


def redact(value: Any) -> Any:
    """Recursively remove secrets and raw model content from telemetry payloads."""

    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(part in key.lower() for part in SECRET_FRAGMENTS) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_secret_patterns(value)
    return value


def _redact_secret_patterns(value: str) -> str:
    """Remove common bearer and API-key assignments embedded inside error strings."""

    import re

    sanitized = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
    return re.sub(
        r"(?i)(api[_-]?key|password|secret)\s*[=:]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        sanitized,
    )


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def persist_local_trace(payload: dict[str, Any], directory: str | Path | None = None) -> Path | None:
    """Atomically persist a redacted trace; storage failures never break research."""

    if not env_flag("LOCAL_TRACE_ENABLED", True):
        return None
    trace_directory = Path(directory or os.getenv("LOCAL_TRACE_DIR", ".cache/traces"))
    try:
        trace_directory.mkdir(parents=True, exist_ok=True)
        destination = trace_directory / f"{payload['trace_id']}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(redact(payload), default=str), encoding="utf-8")
        temporary.replace(destination)
        return destination
    except Exception as exc:
        logging.getLogger("scientific_agent").warning(
            "local_trace.persist_failed",
            extra={"event": "local_trace.persist_failed", "fields": {"error_type": type(exc).__name__}},
        )
        return None


def list_local_traces(directory: str | Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """List newest local trace summaries without loading full event arrays."""

    trace_directory = Path(directory or os.getenv("LOCAL_TRACE_DIR", ".cache/traces"))
    if not trace_directory.exists():
        return []
    summaries = []
    for path in sorted(trace_directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({key: payload.get(key) for key in ("run_id", "trace_id", "created_at", "metrics")})
        except (OSError, json.JSONDecodeError):
            continue
    return summaries


def load_local_trace(trace_id: str, directory: str | Path | None = None) -> dict[str, Any] | None:
    """Load one trace by strict hexadecimal identifier to prevent path traversal."""

    import re

    if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
        return None
    path = Path(directory or os.getenv("LOCAL_TRACE_DIR", ".cache/traces")) / f"{trace_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
