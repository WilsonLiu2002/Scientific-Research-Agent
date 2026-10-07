from __future__ import annotations

from app.memory import WorkspaceMemory


def test_project_contains_multiple_reopenable_chats(tmp_path) -> None:
    """Persist independent chat histories beneath one durable project."""

    memory = WorkspaceMemory(tmp_path / "memory.sqlite")
    project = memory.create_project("Battery discovery")
    first = memory.create_chat(project["id"])
    second = memory.create_chat(project["id"], "Electrolytes")

    memory.begin_run(first["id"], "thread-1", "run-1", "Which cathodes are promising?")
    memory.complete_run(
        first["id"], "run-1", "LFP remains important.", {"answer": "LFP remains important."}
    )

    assert len(memory.list_chats(project["id"])) == 2
    reopened = memory.get_chat(first["id"])
    assert [message["role"] for message in reopened["messages"]] == ["user", "assistant"]
    assert reopened["messages"][1]["result"]["answer"] == "LFP remains important."
    assert second["title"] == "Electrolytes"

    memory.begin_run(second["id"], "thread-2", "run-2", "What about solid electrolytes?")
    context, stats = memory.build_context(project["id"], second["id"])
    assert "Which cathodes are promising?" in context
    assert "What about solid electrolytes?" in context
    assert stats["chat_messages"] == 1
    assert stats["project_messages"] == 2


def test_repeated_completion_does_not_duplicate_chat_messages(tmp_path) -> None:
    """Make resume retries idempotent for the same research run and role."""

    memory = WorkspaceMemory(tmp_path / "memory.sqlite")
    project = memory.ensure_default_project()
    chat = memory.create_chat(project["id"])
    memory.begin_run(chat["id"], "thread-1", "run-1", "How are crystals modeled?")
    memory.complete_run(chat["id"], "run-1", "With graphs.", {"answer": "With graphs."})
    memory.complete_run(chat["id"], "run-1", "With graphs.", {"answer": "With graphs."})

    assert len(memory.get_chat(chat["id"])["messages"]) == 2
