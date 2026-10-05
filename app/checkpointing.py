from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_PATH = Path(".cache/checkpoints/research.sqlite")
CHECKPOINT_MODEL_ALLOWLIST = [
    ("app.models", name)
    for name in (
        "Paper",
        "SearchPlan",
        "ResearchIntent",
        "PlannedSearchQuery",
        "StrategicSearchPlan",
        "ResearchGoal",
        "ResearchSubtask",
        "ResearchCritique",
        "EvidencePassage",
        "DocumentSection",
        "FullTextDocument",
        "EvidenceClaim",
        "PaperReading",
    )
] + [("app.grounding", "ClaimVerification")]


def checkpoint_path() -> Path:
    """Resolve the local durable-checkpoint database configured for human review."""

    configured = os.getenv("CHECKPOINT_DB_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT_PATH


def create_durable_checkpointer(path: str | Path | None = None) -> Any:
    """Create a thread-safe LangGraph SQLite saver that survives process restarts."""

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Durable human review requires `langgraph-checkpoint-sqlite`. "
            "Install the project dependencies before enabling interactive mode."
        ) from exc

    database = Path(path).expanduser() if path else checkpoint_path()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_MODEL_ALLOWLIST)
    return SqliteSaver(connection, serde=serializer)
