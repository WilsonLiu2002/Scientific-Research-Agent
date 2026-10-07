from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any

from app.graph import build_research_graph
from app.auth import default_access_grant, resolve_access_token
from app.observability import log_event, persist_local_trace, research_run
from app.rag import create_vector_store
from app.state import ResearchState


def initial_research_state(
    question: str,
    run_id: str | None = None,
    trace_id: str | None = None,
    access_profile: dict[str, Any] | None = None,
) -> ResearchState:
    """Create the common initial state used by interactive and unattended runs."""

    return {
        "question": question,
        "run_id": run_id or uuid.uuid4().hex,
        "trace_id": trace_id or uuid.uuid4().hex,
        "access_profile": access_profile or default_access_grant().public_payload(),
        "papers": [],
        "retrieved_papers": [],
        "retrieved_passages": [],
        "answer": None,
        "errors": [],
    }


def run(
    question: str,
    interactive: bool = False,
    thread_id: str | None = None,
    access_token: str | None = None,
) -> ResearchState:
    """Run the complete research-agent workflow for one user question."""

    with research_run() as telemetry:
        grant = resolve_access_token(access_token) if access_token else default_access_grant()
        graph = build_research_graph(
            enable_human_review=interactive,
            vector_store=create_vector_store(access_scope=grant.level),
        )
        initial_state = initial_research_state(
            question, telemetry.run_id, telemetry.trace_id, grant.public_payload()
        )
        result = (
            graph.invoke(initial_state)
            if not interactive
            else run_with_human_review(graph, initial_state, thread_id=thread_id)
        )
        result["run_metrics"] = telemetry.metrics.summary(result)
        log_event("research.metrics", **result["run_metrics"])
    result["run_trace"] = telemetry.metrics.trace_payload(
        telemetry.run_id, telemetry.trace_id, result["run_metrics"]
    )
    persist_local_trace(result["run_trace"])
    return result


async def arun(question: str) -> ResearchState:
    """Run the unattended workflow through LangGraph's asynchronous invocation API."""

    with research_run() as telemetry:
        graph = build_research_graph()
        result = await graph.ainvoke(
            initial_research_state(question, telemetry.run_id, telemetry.trace_id)
        )
        result["run_metrics"] = telemetry.metrics.summary(result)
        log_event("research.metrics", **result["run_metrics"])
    result["run_trace"] = telemetry.metrics.trace_payload(
        telemetry.run_id, telemetry.trace_id, result["run_metrics"]
    )
    persist_local_trace(result["run_trace"])
    return result


def resume(thread_id: str) -> ResearchState:
    """Resume a durable human-review thread with a fresh local telemetry segment."""

    with research_run() as telemetry:
        graph = build_research_graph(enable_human_review=True)
        result = resume_with_human_review(graph, thread_id)
        result["run_metrics"] = telemetry.metrics.summary(result)
        log_event("research.metrics", **result["run_metrics"])
    result["run_trace"] = telemetry.metrics.trace_payload(
        telemetry.run_id, telemetry.trace_id, result["run_metrics"]
    )
    persist_local_trace(result["run_trace"])
    return result


def run_with_human_review(
    graph: Any,
    initial_state: ResearchState,
    thread_id: str | None = None,
) -> ResearchState:
    """Drive checkpointed LangGraph interrupts from a terminal reviewer session."""

    from langgraph.types import Command

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state["checkpoint_thread_id"] = thread_id
    print(f"Checkpoint thread: {thread_id}")
    result = graph.invoke(initial_state, config=config)
    while interrupts := result.get("__interrupt__"):
        payload = interrupts[0].value
        decision = prompt_for_review(payload)
        result = graph.invoke(Command(resume=decision), config=config)
    return result


def resume_with_human_review(graph: Any, thread_id: str) -> ResearchState:
    """Resume a paused durable checkpoint without repeating completed graph nodes."""

    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise ValueError(f"No checkpoint exists for thread `{thread_id}`.")
    result: ResearchState = dict(snapshot.values)
    payload = pending_interrupt(snapshot)
    while payload is not None:
        decision = prompt_for_review(payload)
        result = graph.invoke(Command(resume=decision), config=config)
        interrupts = result.get("__interrupt__", [])
        payload = interrupts[0].value if interrupts else None
    return result


def pending_interrupt(snapshot: Any) -> dict[str, Any] | None:
    """Extract the current LangGraph interrupt payload from a persisted snapshot."""

    for task in getattr(snapshot, "tasks", ()):
        interrupts = getattr(task, "interrupts", ())
        if interrupts:
            value = interrupts[0].value
            return value if isinstance(value, dict) else None
    return None


def prompt_for_review(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one review request and collect a structured resume decision."""

    review_type = payload.get("review_type")
    if review_type == "search_plan":
        print("\n[Human Review: Search Plan]\n")
        for index, query in enumerate(payload.get("queries", []), start=1):
            print(f"{index}. {query}")
        action = input("Approve, edit, add instruction, or stop? [approve]: ").strip().lower() or "approve"
        if action == "edit":
            raw = input("Enter replacement queries separated by semicolons: ")
            return {"action": "edit", "queries": [item.strip() for item in raw.split(";") if item.strip()]}
        if action in {"add instruction", "instruction", "instruct"}:
            return {"action": "add_instruction", "instruction": input("New instruction: ").strip()}
        return {"action": action}

    print("\n[Human Review: Full-Text Reading]\n")
    for paper in payload.get("papers", []):
        print(f"- {paper['id']}: {paper['title']}")
    action = input("Approve, choose papers, add instruction, skip, or stop? [approve]: ").strip().lower() or "approve"
    if action in {"choose", "edit"}:
        raw = input("Enter paper IDs separated by commas: ")
        return {"action": "edit", "paper_ids": [item.strip() for item in raw.split(",") if item.strip()]}
    if action in {"add instruction", "instruction", "instruct"}:
        return {"action": "add_instruction", "instruction": input("New instruction: ").strip()}
    return {"action": action}


def main() -> None:
    """CLI entry point for the scientific literature research agent."""

    parser = argparse.ArgumentParser(description="Run a scientific literature research agent.")
    parser.add_argument("question", nargs="?", help="Scientific research question to investigate.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause for approval before search and optional full-text acquisition.",
    )
    parser.add_argument(
        "--async-work",
        action="store_true",
        help="Use the asynchronous application entry point for unattended orchestration.",
    )
    parser.add_argument(
        "--thread-id",
        help="Stable checkpoint thread ID for a new interactive run.",
    )
    parser.add_argument(
        "--resume-thread-id",
        help="Resume a paused interactive run from the durable checkpoint database.",
    )
    args = parser.parse_args()
    if args.interactive and args.async_work:
        parser.error("--interactive and --async-work cannot be used together")
    if args.resume_thread_id and (args.question or args.async_work):
        parser.error("--resume-thread-id is used alone")
    if not args.resume_thread_id and not args.question:
        parser.error("a question is required unless --resume-thread-id is used")

    print("[Planner]\n")
    if args.resume_thread_id:
        state = resume(args.resume_thread_id)
    else:
        state = asyncio.run(arun(args.question)) if args.async_work else run(
            args.question, interactive=args.interactive, thread_id=args.thread_id
        )
    if state.get("cancelled"):
        print("\n[Cancelled]\n")
        print(state.get("answer"))
        return
    print("Generated search queries:")
    for index, query in enumerate(state.get("search_queries", []), start=1):
        print(f"{index}. {query}")
    goal = state.get("goal")
    if goal:
        print(f"Goal: {goal.objective}")
        print(f"Goal status: {goal.status} ({goal.progress:.0%})")
    subtasks = state.get("subtasks", [])
    if subtasks:
        status_counts: dict[str, int] = {}
        for task in subtasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        summary = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        print(f"Subtasks: {len(subtasks)} ({summary})")

    print("\n[Search]\n")
    print(f"Retrieved {len(state.get('papers', []))} unique papers.")
    for error in state.get("errors", []):
        print(f"Warning: {error}")

    print("\n[RAG]\n")
    print(f"Indexed {state.get('indexed_paper_count', 0)} papers with usable abstracts.")
    print(f"Retrieved {len(state.get('retrieved_passages', []))} evidence passages.")
    context_stats = state.get("context_stats", {})
    if context_stats:
        print(
            f"Synthesis context: {context_stats.get('selected_passages', 0)} passages from "
            f"{context_stats.get('distinct_papers', 0)} papers, approximately "
            f"{context_stats.get('estimated_tokens', 0)} tokens."
        )
        dropped = context_stats.get("duplicate_passages_removed", 0) + context_stats.get(
            "budget_passages_dropped", 0
        )
        if dropped:
            print(f"Context filtering removed or deferred {dropped} passage(s).")
    print(
        f"Evidence coverage: {state.get('evidence_coverage', 0.0):.0%} "
        f"after {state.get('search_iteration', 1)} search iteration(s)."
    )
    for note in state.get("evidence_notes", []):
        print(f"Coverage note: {note}")
    for warning in state.get("citation_warnings", []):
        print(f"Citation warning: {warning}")
    if state.get("human_review_history"):
        print(f"Human review decisions: {len(state['human_review_history'])}")
    if metrics := state.get("run_metrics"):
        print(
            f"Run metrics: {metrics['total_duration_ms']:.0f} ms, "
            f"{metrics['llm_calls']} LLM call(s), {metrics['tool_calls']} tool call(s), "
            f"{metrics['errors']} error(s)."
        )

    print("\n[Research]\n")
    print("=================================\n")
    print("Research Report\n")
    print(state.get("answer") or "No answer generated.")
    print("\n=================================")


if __name__ == "__main__":
    main()
