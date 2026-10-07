from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_PATH = Path(".cache/memory/workspaces.sqlite")


def memory_path() -> Path:
    """Resolve the durable project/chat memory database path."""

    configured = os.getenv("MEMORY_DB_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_MEMORY_PATH


def utc_now() -> str:
    """Return a sortable UTC timestamp for persisted workspace records."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkspaceMemory:
    """SQLite repository for projects, chats, messages, and saved research results."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path else memory_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection configured for concurrent browser requests."""

        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        """Create the version-one workspace schema idempotently."""

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    minimum_access_level TEXT NOT NULL DEFAULT 'local_reader'
                );
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    active_thread_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    minimum_access_level TEXT NOT NULL DEFAULT 'local_reader'
                );
                CREATE INDEX IF NOT EXISTS chats_project_updated
                    ON chats(project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    run_id TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(chat_id, run_id, role)
                );
                CREATE INDEX IF NOT EXISTS messages_chat_created
                    ON messages(chat_id, created_at ASC);
                CREATE TABLE IF NOT EXISTS chat_runs (
                    thread_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_runs_chat
                    ON chat_runs(chat_id, created_at DESC);
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO chat_runs(thread_id, chat_id, run_id, created_at)
                SELECT active_thread_id, id, '', created_at FROM chats
                WHERE active_thread_id IS NOT NULL
                """
            )
            self._add_column_if_missing(
                connection, "projects", "minimum_access_level", "TEXT NOT NULL DEFAULT 'local_reader'"
            )
            self._add_column_if_missing(
                connection, "chats", "minimum_access_level", "TEXT NOT NULL DEFAULT 'local_reader'"
            )

    @staticmethod
    def _add_column_if_missing(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        """Apply a small forward-only SQLite migration for existing local databases."""

        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def ensure_default_project(self, minimum_access_level: str = "local_reader") -> dict[str, Any]:
        """Return the most recent project, creating a friendly default when empty."""

        projects = self.list_projects()
        return projects[0] if projects else self.create_project(
            "My Research", minimum_access_level=minimum_access_level
        )

    def create_project(
        self, name: str, description: str = "", minimum_access_level: str = "local_reader"
    ) -> dict[str, Any]:
        """Create one project that can contain any number of chats."""

        project_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO projects(id, name, description, created_at, updated_at, minimum_access_level) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, name.strip(), description.strip(), timestamp, timestamp, minimum_access_level),
            )
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        """List projects with chat and message counts, newest activity first."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*, COUNT(DISTINCT c.id) AS chat_count,
                       COUNT(m.id) AS message_count
                FROM projects p
                LEFT JOIN chats c ON c.project_id = p.id
                LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY p.id
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Load one project or raise KeyError for a missing identifier."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        return dict(row)

    def create_chat(
        self, project_id: str, title: str = "New chat", minimum_access_level: str | None = None
    ) -> dict[str, Any]:
        """Create an independent conversation inside an existing project."""

        project = self.get_project(project_id)
        chat_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chats(id, project_id, title, created_at, updated_at, minimum_access_level) VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, project_id, title.strip() or "New chat", timestamp, timestamp, minimum_access_level or project["minimum_access_level"]),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id)
            )
        return self.get_chat(chat_id, include_messages=False)

    def list_chats(self, project_id: str) -> list[dict[str, Any]]:
        """List a project's chats with compact history metadata."""

        self.get_project(project_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count,
                       MAX(m.created_at) AS last_message_at
                FROM chats c
                LEFT JOIN messages m ON m.chat_id = c.id
                WHERE c.project_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: str, include_messages: bool = True) -> dict[str, Any]:
        """Load a chat and optionally its ordered question/response history."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row is None:
                raise KeyError(chat_id)
            chat = dict(row)
            if include_messages:
                messages = connection.execute(
                    "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC, rowid ASC",
                    (chat_id,),
                ).fetchall()
        if include_messages:
            chat["messages"] = [self._message_payload(item) for item in messages]
        return chat

    def get_chat_by_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Resolve the protected chat owning any current or historical run thread."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.* FROM chats c
                JOIN chat_runs r ON r.chat_id = c.id
                WHERE r.thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def begin_run(self, chat_id: str, thread_id: str, run_id: str, question: str) -> None:
        """Persist a user question and associate its checkpoint thread with the chat."""

        chat = self.get_chat(chat_id, include_messages=False)
        timestamp = utc_now()
        title = chat["title"]
        if title == "New chat":
            title = " ".join(question.split())[:72]
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO chat_runs(thread_id, chat_id, run_id, created_at) VALUES (?, ?, ?, ?)",
                (thread_id, chat_id, run_id, timestamp),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO messages(id, chat_id, role, content, run_id, created_at)
                VALUES (?, ?, 'user', ?, ?, ?)
                """,
                (uuid.uuid4().hex, chat_id, question.strip(), run_id, timestamp),
            )
            connection.execute(
                "UPDATE chats SET title = ?, status = 'running', active_thread_id = ?, updated_at = ? WHERE id = ?",
                (title, thread_id, timestamp, chat_id),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (timestamp, chat["project_id"]),
            )

    def set_chat_status(self, chat_id: str, status: str) -> None:
        """Update a chat lifecycle state without altering its message history."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE chats SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), chat_id),
            )

    def complete_run(
        self, chat_id: str, run_id: str, answer: str, result: dict[str, Any]
    ) -> None:
        """Persist the assistant response and full structured result for exact restoration."""

        timestamp = utc_now()
        encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO messages(id, chat_id, role, content, run_id, result_json, created_at)
                VALUES (?, ?, 'assistant', ?, ?, ?, ?)
                """,
                (uuid.uuid4().hex, chat_id, answer, run_id, encoded, timestamp),
            )
            connection.execute(
                "UPDATE chats SET status = 'completed', active_thread_id = NULL, updated_at = ? WHERE id = ?",
                (timestamp, chat_id),
            )

    def build_context(
        self, project_id: str, chat_id: str, max_messages: int = 12, max_chars: int = 6000
    ) -> tuple[str, dict[str, int]]:
        """Build bounded chat- and project-level continuity context for query planning."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.role, m.content, m.chat_id, c.title
                FROM messages m
                JOIN chats c ON c.id = m.chat_id
                WHERE c.project_id = ?
                ORDER BY CASE WHEN c.id = ? THEN 0 ELSE 1 END,
                         m.created_at DESC, m.rowid DESC
                LIMIT ?
                """,
                (project_id, chat_id, max_messages),
            ).fetchall()
        lines: list[str] = []
        chat_messages = 0
        project_messages = 0
        for row in reversed(rows):
            scope = "current chat" if row["chat_id"] == chat_id else f"project chat: {row['title']}"
            if row["chat_id"] == chat_id:
                chat_messages += 1
            else:
                project_messages += 1
            lines.append(f"[{scope}] {row['role']}: {row['content']}")
        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[-max_chars:]
        return context, {
            "chat_messages": chat_messages,
            "project_messages": project_messages,
            "characters": len(context),
        }

    @staticmethod
    def _message_payload(row: sqlite3.Row) -> dict[str, Any]:
        """Decode one message row while keeping absent structured results lightweight."""

        payload = dict(row)
        encoded = payload.pop("result_json", None)
        payload["result"] = json.loads(encoded) if encoded else None
        return payload
