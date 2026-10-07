from __future__ import annotations

import re
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.context import ResearchContextBuilder
from app.local_corpus import LocalLiteratureCorpus
from app.checkpointing import create_durable_checkpointer
from app.critic import ResearchCritic, create_default_critic
from app.evidence import PaperEvidenceExtractor
from app.grounding import GroundingVerifier, repair_unsupported_claims
from app.llm import generate_research_plan, synthesize_answer, validate_citations
from app.mcp_client import AcademicMCPClient
from app.models import EvidencePassage, Paper, ResearchGoal, ResearchSubtask, deduplicate_papers
from app.observability import (
    log_event,
    metrics_increment,
    metrics_set,
    observe_node,
    observe_route,
    span,
)
from app.rag import create_vector_store, retrieve_for_intent
from app.skill_loader import load_skill
from app.state import ResearchState


class ResearchRunCancelled(RuntimeError):
    """Raised at graph operation boundaries after a user requests cancellation."""


class SimpleCompiledGraph:
    """Fallback runner that preserves the iterative workflow without LangGraph."""

    def __init__(self, builder: "ResearchGraphBuilder") -> None:
        self.builder = builder

    def invoke(self, state: ResearchState) -> ResearchState:
        """Execute planning, bounded research loops, synthesis, and validation."""

        current: ResearchState = dict(state)
        current.update(self.builder.plan_research(current))
        while True:
            current.update(self.builder.search_papers(current))
            current.update(self.builder.index_papers(current))
            current.update(self.builder.retrieve_relevant_papers(current))
            current.update(self.builder.evaluate_evidence(current))
            current.update(self.builder.critique_research(current))
            route = self.builder.route_after_critique(current)
            if route == "write_answer":
                break
            if route == "deep_read_papers":
                current.update(self.builder.deep_read_papers(current))
                break
            current.update(self.builder.generate_followup_queries(current))
        current.update(self.builder.prepare_context(current))
        current.update(self.builder.write_answer(current))
        current.update(self.builder.verify_grounding(current))
        current.update(self.builder.validate_answer(current))
        return current


class ResearchGraphBuilder:
    """Factory that wires dependencies into the LangGraph research workflow nodes."""

    def __init__(
        self,
        search_client: Any | None = None,
        vector_store: Any | None = None,
        max_search_iterations: int = 3,
        min_relevant_papers: int = 4,
        min_evidence_passages: int = 4,
        min_concept_coverage: float = 0.6,
        enable_full_text: bool = True,
        max_fulltext_papers: int = 3,
        enable_human_review: bool = False,
        checkpointer: Any | None = None,
        context_builder: ResearchContextBuilder | None = None,
        evidence_extractor: PaperEvidenceExtractor | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        research_critic: ResearchCritic | None = None,
        max_critic_retries: int = 2,
        max_concurrent_tasks: int = 4,
        cancellation_event: Any | None = None,
        local_corpus: LocalLiteratureCorpus | None = None,
    ) -> None:
        self.search_client = search_client or AcademicMCPClient()
        self.vector_store = vector_store or create_vector_store()
        self.max_search_iterations = max(1, max_search_iterations)
        self.min_relevant_papers = max(1, min_relevant_papers)
        self.min_evidence_passages = max(1, min_evidence_passages)
        self.min_concept_coverage = min(max(min_concept_coverage, 0.0), 1.0)
        self.enable_full_text = enable_full_text
        self.max_fulltext_papers = max(1, max_fulltext_papers)
        self.enable_human_review = enable_human_review
        self.checkpointer = checkpointer
        self.context_builder = context_builder or ResearchContextBuilder()
        self.evidence_extractor = evidence_extractor or PaperEvidenceExtractor()
        self.grounding_verifier = grounding_verifier or GroundingVerifier()
        self.research_critic = research_critic or create_default_critic()
        self.max_critic_retries = max(0, max_critic_retries)
        self.max_concurrent_tasks = max(1, max_concurrent_tasks)
        self.cancellation_event = cancellation_event
        self.local_corpus = local_corpus or LocalLiteratureCorpus()

    def check_cancelled(self) -> None:
        """Stop before another costly operation when the browser cancels a run."""

        if self.cancellation_event is not None and self.cancellation_event.is_set():
            raise ResearchRunCancelled("Research stopped by the reviewer.")

    def plan_research(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: load the skill and generate literature-search queries."""

        self.check_cancelled()
        skill = state.get("skill") or load_skill("literature-research")
        plan = generate_research_plan(
            question=state["question"],
            skill=skill,
            memory_context=state.get("memory_context"),
        )
        self.check_cancelled()
        queries = [item.query for item in plan.queries]
        criteria = [
            f"Retrieve at least {self.min_relevant_papers} relevant papers",
            f"Retrieve at least {self.min_evidence_passages} evidence passages",
            f"Cover at least {self.min_concept_coverage:.0%} of question concepts",
            "Produce a citation-validated answer",
        ]
        return {
            "skill": skill,
            "research_intent": plan.intent,
            "search_queries": queries,
            "query_rationales": {item.query: item.rationale for item in plan.queries},
            "search_iteration": 1,
            "searched_queries": [],
            "deep_read_requested": access_allows(state, "fulltext:read")
            and self.enable_full_text
            and question_requires_full_text(state["question"]),
            "fulltext_attempted": False,
            "human_review_history": [],
            "reviewer_instructions": [],
            "critique_history": [],
            "cancelled": False,
            "goal": ResearchGoal(
                objective=f"Answer the research question: {state['question']}",
                success_criteria=criteria,
                status="running",
            ),
            "subtasks": make_search_subtasks(queries, iteration=1),
        }

    def review_search_plan(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph HITL node: pause before external searches for approval or query edits."""

        from langgraph.types import interrupt

        response = interrupt(
            {
                "review_type": "search_plan",
                "question": state["question"],
                "research_intent": (
                    state.get("research_intent").model_dump()
                    if state.get("research_intent") else None
                ),
                "queries": state.get("search_queries", []),
                "query_rationales": state.get("query_rationales", {}),
                "goal": state.get("goal").model_dump() if state.get("goal") else None,
                "subtasks": [task.model_dump() for task in state.get("subtasks", []) if task.status == "pending"],
                "reviewer_instructions": state.get("reviewer_instructions", []),
                "allowed_actions": ["approve", "edit", "add_instruction", "stop"],
            }
        )
        action = str(response.get("action", "approve")).lower() if isinstance(response, dict) else "approve"
        action = {"cancel": "stop", "instruct": "add_instruction"}.get(action, action)
        if action not in {"approve", "edit", "add_instruction", "stop"}:
            action = "approve"
        instruction = normalize_instruction(response.get("instruction")) if isinstance(response, dict) else None
        instructions = list(state.get("reviewer_instructions", []))
        if instruction and instruction not in instructions:
            instructions.append(instruction)
        queries = state.get("search_queries", [])
        if action == "edit" and isinstance(response, dict):
            edited = response.get("queries", [])
            queries = normalize_string_list(edited, maximum=5) or queries
        elif action == "add_instruction" and instruction:
            queries = apply_instruction_to_queries(queries, instruction)
        existing_tasks = [
            task
            for task in state.get("subtasks", [])
            if not (task.kind == "literature_search" and task.status == "pending")
        ]
        subtasks = [
            *existing_tasks,
            *make_search_subtasks(queries, iteration=state.get("search_iteration", 1)),
        ]
        history = [
            *state.get("human_review_history", []),
            {
                "review_type": "search_plan",
                "action": action,
                "queries": queries,
                "instruction": instruction,
            },
        ]
        return {
            "human_review_action": "approve" if action == "add_instruction" else action,
            "human_review_history": history,
            "reviewer_instructions": instructions,
            "search_queries": queries,
            "subtasks": subtasks,
            "cancelled": action == "stop",
        }

    def route_after_search_review(self, state: ResearchState) -> str:
        """Route an approved search to execution or a cancellation to a terminal response."""

        return "cancel_workflow" if state.get("cancelled") else "search_papers"

    def search_papers(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: execute approved literature-search subtasks concurrently."""

        self.check_cancelled()
        papers: list[Paper] = list(state.get("papers", []))
        errors = list(state.get("errors", []))
        searched_queries = list(state.get("searched_queries", []))
        pending_queries = [
            query for query in state.get("search_queries", []) if query not in searched_queries
        ]
        task_by_query = {
            task.query: task
            for task in state.get("subtasks", [])
            if task.kind == "literature_search" and task.status == "pending" and task.query
        }
        pending_tasks = [
            task_by_query.get(query)
            or ResearchSubtask(
                id=f"search-{state.get('search_iteration', 1)}-{index}",
                kind="literature_search",
                objective=f"Find literature for: {query}",
                iteration=state.get("search_iteration", 1),
                query=query,
            )
            for index, query in enumerate(pending_queries, start=1)
        ]

        if not access_allows(state, "literature:search"):
            skipped = {
                task.id: task.model_copy(
                    update={"status": "skipped", "error": "Authorization tier permits local RAG only"}
                )
                for task in pending_tasks
            }
            log_event(
                "authorization.denied",
                capability="literature:search",
                access_level=state.get("access_profile", {}).get("level"),
                node="search_papers",
            )
            return {
                "papers": papers,
                "searched_queries": [*searched_queries, *pending_queries],
                "subtasks": [skipped.get(task.id, task) for task in state.get("subtasks", [])],
                "errors": errors,
            }

        def execute(task: ResearchSubtask) -> tuple[ResearchSubtask, list[Paper]]:
            """Run one isolated search subtask and return its auditable outcome."""

            try:
                with span(
                    "subtask.literature_search",
                    kind="tool",
                    export=False,
                    fields={"subtask_id": task.id, "node": "search_papers"},
                ):
                    results = self.search_client.search_papers(query=task.query or "", limit=10)
                completed = task.model_copy(
                    update={
                        "status": "completed",
                        "result_count": len(results),
                        "result_ids": [paper.id for paper in results],
                    }
                )
                return completed, results
            except Exception as exc:
                failed = task.model_copy(update={"status": "failed", "error": str(exc)})
                return failed, []

        workers = min(self.max_concurrent_tasks, len(pending_tasks)) or 1
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="literature-search") as executor:
            futures = [executor.submit(contextvars.copy_context().run, execute, task) for task in pending_tasks]
            outcomes = [future.result() for future in futures]

        completed_by_id = {task.id: task for task, _ in outcomes}
        for task, results in outcomes:
            self.check_cancelled()
            papers.extend(results)
            if task.error:
                errors.append(f"Search failed for `{task.query}`: {task.error}")
            if task.query:
                searched_queries.append(task.query)
        subtasks = [
            completed_by_id.get(task.id, task) for task in state.get("subtasks", [])
        ]
        known_ids = {task.id for task in subtasks}
        subtasks.extend(task for task, _ in outcomes if task.id not in known_ids)
        return {
            "papers": deduplicate_papers(papers),
            "errors": errors,
            "searched_queries": searched_queries,
            "subtasks": subtasks,
        }

    def index_papers(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: index durable local records together with newly discovered papers."""

        self.check_cancelled()
        local_papers = self.local_corpus.papers() if access_allows(state, "rag:read") else []
        papers = deduplicate_papers([*local_papers, *state.get("papers", [])])
        indexed_count = self.vector_store.index_papers(papers)
        log_event(
            "rag.corpus_indexed",
            local_document_count=len(local_papers),
            discovered_document_count=len(state.get("papers", [])),
            indexed_document_count=indexed_count,
        )
        return {
            "papers": papers,
            "indexed_paper_count": indexed_count,
            "local_corpus_paper_count": len(local_papers),
        }

    def retrieve_relevant_papers(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: retrieve the most relevant evidence passages."""

        self.check_cancelled()
        if state.get("indexed_paper_count", 0) == 0:
            return {"retrieved_passages": [], "retrieved_papers": []}
        passages, probes = retrieve_for_intent(
            self.vector_store,
            state["question"],
            state.get("research_intent"),
            top_k=8,
        )
        papers = deduplicate_papers([passage.paper for passage in passages])
        metrics_set("retrieval_passage_ids", [passage.id for passage in passages])
        metrics_set("retrieval_scores", [passage.score for passage in passages])
        return {
            "retrieved_passages": passages,
            "retrieved_papers": papers,
            "retrieval_probes": probes,
        }

    def evaluate_evidence(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: measure evidence volume and question-concept coverage."""

        passages = state.get("retrieved_passages", [])
        papers = state.get("retrieved_papers", [])
        concepts = extract_question_concepts(state["question"])
        evidence_text = " ".join(
            f"{passage.paper.title} {passage.text}" for passage in passages
        ).lower()
        missing = [concept for concept in concepts if not concept_is_covered(concept, evidence_text)]
        coverage = 1.0 if not concepts else (len(concepts) - len(missing)) / len(concepts)

        notes: list[str] = []
        if len(papers) < self.min_relevant_papers:
            notes.append(
                f"Only {len(papers)} relevant papers were retrieved; target is {self.min_relevant_papers}."
            )
        if len(passages) < self.min_evidence_passages:
            notes.append(
                f"Only {len(passages)} passages were retrieved; target is {self.min_evidence_passages}."
            )
        if coverage < self.min_concept_coverage:
            notes.append(
                f"Question-concept coverage is {coverage:.0%}; target is {self.min_concept_coverage:.0%}."
            )
        goal = state.get("goal")
        paper_progress = min(len(papers) / self.min_relevant_papers, 1.0)
        passage_progress = min(len(passages) / self.min_evidence_passages, 1.0)
        coverage_progress = (
            1.0 if self.min_concept_coverage == 0 else min(coverage / self.min_concept_coverage, 1.0)
        )
        progress = (paper_progress + passage_progress + coverage_progress) / 3
        if goal:
            if not notes:
                goal_status = "evidence_ready"
            elif state.get("search_iteration", 1) >= self.max_search_iterations:
                goal_status = "budget_exhausted"
            else:
                goal_status = "running"
            goal = goal.model_copy(update={"status": goal_status, "progress": progress})
        return {
            "evidence_sufficient": not notes,
            "evidence_coverage": coverage,
            "missing_concepts": missing,
            "evidence_notes": notes,
            "goal": goal,
        }

    def generate_followup_queries(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: execute the critic's targeted, bounded retry plan."""

        iteration = state.get("search_iteration", 1) + 1
        critique = state.get("critique")
        queries = critique.recommended_queries if critique else []
        searched = set(state.get("searched_queries", []))
        unique_queries = [query for query in queries if query not in searched]
        return {
            "search_queries": unique_queries,
            "search_iteration": iteration,
            "subtasks": [
                *state.get("subtasks", []),
                *make_search_subtasks(unique_queries, iteration=iteration),
            ],
        }

    def critique_research(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: reflect on evidence quality and choose one bounded next action."""

        self.check_cancelled()
        critique = self.research_critic.critique(
            dict(state),
            max_search_iterations=self.max_search_iterations,
            max_critic_retries=self.max_critic_retries,
        )
        self.check_cancelled()
        if critique.verdict == "retry":
            metrics_increment("retries")
        return {
            "critique": critique,
            "critique_history": [*state.get("critique_history", []), critique],
        }

    def route_after_critique(self, state: ResearchState) -> str:
        """Route the critic decision while keeping external work inside fixed budgets."""

        critique = state.get("critique")
        verdict = critique.verdict if critique else "stop"
        if verdict == "retry" and critique and critique.recommended_queries:
            return "generate_followup_queries"
        if verdict == "deep_read" and not state.get("fulltext_attempted"):
            return "review_deep_read" if self.enable_human_review else "deep_read_papers"
        return "write_answer"

    def route_after_evaluation(self, state: ResearchState) -> str:
        """Backward-compatible routing helper for callers with precomputed critiques."""

        return self.route_after_critique(state)

    def review_deep_read(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph HITL node: approve, narrow, skip, or cancel full-text acquisition."""

        from langgraph.types import interrupt

        candidates = state.get("retrieved_papers", [])[: self.max_fulltext_papers]
        response = interrupt(
            {
                "review_type": "deep_read",
                "question": state["question"],
                "papers": [
                    {"id": paper.id, "title": paper.title, "doi": paper.doi, "url": paper.url}
                    for paper in candidates
                ],
                "reviewer_instructions": state.get("reviewer_instructions", []),
                "allowed_actions": ["approve", "edit", "add_instruction", "skip", "stop"],
            }
        )
        action = str(response.get("action", "approve")).lower() if isinstance(response, dict) else "approve"
        action = {"cancel": "stop", "instruct": "add_instruction"}.get(action, action)
        if action not in {"approve", "edit", "add_instruction", "skip", "stop"}:
            action = "approve"
        instruction = normalize_instruction(response.get("instruction")) if isinstance(response, dict) else None
        instructions = list(state.get("reviewer_instructions", []))
        if instruction and instruction not in instructions:
            instructions.append(instruction)
        available_ids = {paper.id for paper in candidates}
        selected_ids = [paper.id for paper in candidates]
        if action == "edit" and isinstance(response, dict):
            requested = normalize_string_list(response.get("paper_ids", []), self.max_fulltext_papers)
            selected_ids = [paper_id for paper_id in requested if paper_id in available_ids]
            action = "approve" if selected_ids else "skip"
        history = [
            *state.get("human_review_history", []),
            {
                "review_type": "deep_read",
                "action": action,
                "paper_ids": selected_ids,
                "instruction": instruction,
            },
        ]
        return {
            "human_review_action": "approve" if action == "add_instruction" else action,
            "human_review_history": history,
            "reviewer_instructions": instructions,
            "selected_fulltext_paper_ids": selected_ids,
            "fulltext_attempted": action == "skip",
            "cancelled": action == "stop",
        }

    def route_after_deep_review(self, state: ResearchState) -> str:
        """Route a deep-reading decision to acquisition, synthesis, or cancellation."""

        action = state.get("human_review_action", "approve")
        if state.get("cancelled") or action == "stop":
            return "cancel_workflow"
        if action == "skip":
            return "write_answer"
        return "deep_read_papers"

    def deep_read_papers(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: acquire selected papers and extract focused evidence."""

        self.check_cancelled()
        if not access_allows(state, "fulltext:read"):
            log_event(
                "authorization.denied",
                capability="fulltext:read",
                access_level=state.get("access_profile", {}).get("level"),
                node="deep_read_papers",
            )
            return {
                "fulltext_attempted": False,
                "evidence_notes": [
                    *state.get("evidence_notes", []),
                    "Full-text acquisition is not permitted by the selected authorization tier.",
                ],
            }
        documents = []
        readings = []
        fulltext_passages: list[EvidencePassage] = []
        errors = list(state.get("errors", []))
        get_full_text = getattr(self.search_client, "get_full_text", None)
        if not callable(get_full_text):
            errors.append("The MCP full-text tool is unavailable on the configured client.")
            return {"fulltext_attempted": True, "errors": errors}

        selected_ids = set(state.get("selected_fulltext_paper_ids", []))
        candidates = state.get("retrieved_papers", [])[: self.max_fulltext_papers]
        selected = [paper for paper in candidates if not selected_ids or paper.id in selected_ids]
        iteration = state.get("search_iteration", 1)
        deep_tasks = [
            ResearchSubtask(
                id=f"deep-{iteration}-{index}-{paper.id}",
                kind="deep_read",
                objective=f"Acquire and read full text for: {paper.title}",
                iteration=iteration,
                paper_id=paper.id,
            )
            for index, paper in enumerate(selected, start=1)
        ]

        def execute(item: tuple[Paper, ResearchSubtask]) -> tuple[ResearchSubtask, Any, Any]:
            """Run one full-text acquisition and focused reading subtask."""

            paper, task = item
            try:
                with span(
                    "subtask.deep_read",
                    kind="tool",
                    export=False,
                    fields={"subtask_id": task.id, "node": "deep_read_papers"},
                ):
                    metrics_increment("deep_read_operations")
                    document = get_full_text(paper)
                if not document:
                    return task.model_copy(
                        update={"status": "skipped", "error": "No open-access full text available"}
                    ), None, None
                reading = self.evidence_extractor.extract(document, state["question"])
                claim_count = sum(
                    len(items)
                    for items in (
                        reading.methods,
                        reading.datasets,
                        reading.results,
                        reading.limitations,
                        reading.conclusions,
                    )
                )
                return task.model_copy(
                    update={
                        "status": "completed",
                        "result_count": claim_count,
                        "result_ids": [document.document_id],
                    }
                ), document, reading
            except Exception as exc:
                return task.model_copy(update={"status": "failed", "error": str(exc)}), None, None

        workers = min(self.max_concurrent_tasks, len(deep_tasks)) or 1
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deep-read") as executor:
            futures = [
                executor.submit(contextvars.copy_context().run, execute, item)
                for item in zip(selected, deep_tasks)
            ]
            outcomes = [future.result() for future in futures]

        for paper, (task, document, reading) in zip(selected, outcomes):
            self.check_cancelled()
            if task.status == "failed":
                errors.append(f"Deep reading failed for `{paper.title}`: {task.error}")
            if not document or not reading:
                continue
            documents.append(document)
            readings.append(reading)
            claims = [
                *reading.methods,
                *reading.datasets,
                *reading.results,
                *reading.limitations,
                *reading.conclusions,
            ]
            for index, claim in enumerate(claims, start=1):
                fulltext_passages.append(
                    EvidencePassage(
                        id=f"full:{document.document_id}:{index}",
                        paper=paper,
                        text=claim.text,
                        section=claim.section,
                        page=claim.page,
                        source_url=claim.source_url,
                        score=1.0,
                    )
                )

        combined = [*fulltext_passages, *state.get("retrieved_passages", [])]
        notes = list(state.get("evidence_notes", []))
        if selected and not documents:
            notes.append("No open-access full text could be acquired for the selected papers.")
        return {
            "fulltext_attempted": True,
            "fulltext_documents": documents,
            "paper_readings": readings,
            "retrieved_passages": combined[:12],
            "evidence_notes": notes,
            "errors": errors,
            "subtasks": [*state.get("subtasks", []), *(task for task, _, _ in outcomes)],
        }

    def cancel_workflow(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: terminate cleanly without executing further external actions."""

        goal = state.get("goal")
        if goal:
            goal = goal.model_copy(update={"status": "cancelled"})
        return {
            "cancelled": True,
            "answer": "Research cancelled during human review. No additional actions were executed.",
            "goal": goal,
        }

    def prepare_context(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: build a safe, diverse, budgeted evidence packet for synthesis."""

        packet = self.context_builder.build(
            question=state["question"],
            passages=state.get("retrieved_passages", []),
            intent=state.get("research_intent"),
        )
        for key, value in packet.stats.items():
            metrics_set(f"context_{key}", value)
        synthesis_context = packet.text
        instructions = state.get("reviewer_instructions", [])
        if instructions:
            guidance = "\n".join(f"- {item}" for item in instructions)
            synthesis_context += (
                "\n\nREVIEWER GUIDANCE\n"
                "These are explicit human constraints for the final analysis:\n"
                f"{guidance}"
            )
        return {
            "retrieved_passages": packet.passages,
            "synthesis_context": synthesis_context,
            "context_stats": packet.stats,
        }

    def write_answer(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: synthesize the final answer from retrieved passages only."""

        self.check_cancelled()
        answer = synthesize_answer(
            question=state["question"],
            retrieved_passages=state.get("retrieved_passages", []),
            skill=state.get("skill", ""),
            synthesis_context=state.get("synthesis_context"),
        )
        self.check_cancelled()
        return {"answer": answer}

    def validate_answer(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: remove claims lacking valid passage citations and expose warnings."""

        answer, warnings = validate_citations(
            state.get("answer") or "", state.get("retrieved_passages", [])
        )
        goal = state.get("goal")
        if goal and goal.status == "evidence_ready" and answer and not warnings:
            goal = goal.model_copy(update={"status": "completed", "progress": 1.0})
        return {"answer": answer, "citation_warnings": warnings, "goal": goal}

    def verify_grounding(self, state: ResearchState) -> dict[str, Any]:
        """LangGraph node: verify cited claims and perform one bounded repair pass."""

        self.check_cancelled()
        answer = state.get("answer") or ""
        verdicts = self.grounding_verifier.verify(answer, state.get("retrieved_passages", []))
        self.check_cancelled()
        repaired, warnings = repair_unsupported_claims(answer, verdicts)
        return {
            "answer": repaired,
            "grounding_verdicts": verdicts,
            "grounding_warnings": warnings,
        }

    def compile(self) -> Any:
        """Build the LangGraph workflow, falling back to a compatible linear runner if unavailable."""

        try:
            from langgraph.graph import END, START, StateGraph
        except (ImportError, ModuleNotFoundError):
            if self.enable_human_review:
                raise RuntimeError("Human review requires LangGraph with checkpoint support.")
            return SimpleCompiledGraph(self)

        graph = StateGraph(ResearchState)
        graph.add_node("plan_research", observe_node("plan_research", self.plan_research))
        graph.add_node("review_search_plan", observe_node("review_search_plan", self.review_search_plan))
        graph.add_node("search_papers", observe_node("search_papers", self.search_papers))
        graph.add_node("index_papers", observe_node("index_papers", self.index_papers))
        graph.add_node("retrieve_relevant_papers", observe_node("retrieve_relevant_papers", self.retrieve_relevant_papers))
        graph.add_node("evaluate_evidence", observe_node("evaluate_evidence", self.evaluate_evidence))
        graph.add_node("critique_research", observe_node("critique_research", self.critique_research))
        graph.add_node("generate_followup_queries", observe_node("generate_followup_queries", self.generate_followup_queries))
        graph.add_node("deep_read_papers", observe_node("deep_read_papers", self.deep_read_papers))
        graph.add_node("review_deep_read", observe_node("review_deep_read", self.review_deep_read))
        graph.add_node("cancel_workflow", observe_node("cancel_workflow", self.cancel_workflow))
        graph.add_node("prepare_context", observe_node("prepare_context", self.prepare_context))
        graph.add_node("write_answer", observe_node("write_answer", self.write_answer))
        graph.add_node("verify_grounding", observe_node("verify_grounding", self.verify_grounding))
        graph.add_node("validate_answer", observe_node("validate_answer", self.validate_answer))

        graph.add_edge(START, "plan_research")
        if self.enable_human_review:
            graph.add_edge("plan_research", "review_search_plan")
            graph.add_conditional_edges(
                "review_search_plan",
                observe_route("after_search_review", self.route_after_search_review),
                {"search_papers": "search_papers", "cancel_workflow": "cancel_workflow"},
            )
        else:
            graph.add_edge("plan_research", "search_papers")
        graph.add_edge("search_papers", "index_papers")
        graph.add_edge("index_papers", "retrieve_relevant_papers")
        graph.add_edge("retrieve_relevant_papers", "evaluate_evidence")
        graph.add_edge("evaluate_evidence", "critique_research")
        graph.add_conditional_edges(
            "critique_research",
            observe_route("after_critique", self.route_after_critique),
            {
                "generate_followup_queries": "generate_followup_queries",
                "review_deep_read": "review_deep_read",
                "deep_read_papers": "deep_read_papers",
                "write_answer": "prepare_context",
            },
        )
        graph.add_edge(
            "generate_followup_queries",
            "review_search_plan" if self.enable_human_review else "search_papers",
        )
        graph.add_conditional_edges(
            "review_deep_read",
            observe_route("after_deep_review", self.route_after_deep_review),
            {
                "deep_read_papers": "deep_read_papers",
                "write_answer": "prepare_context",
                "cancel_workflow": "cancel_workflow",
            },
        )
        graph.add_edge("deep_read_papers", "prepare_context")
        graph.add_edge("prepare_context", "write_answer")
        graph.add_edge("write_answer", "verify_grounding")
        graph.add_edge("verify_grounding", "validate_answer")
        graph.add_edge("validate_answer", END)
        graph.add_edge("cancel_workflow", END)
        checkpointer = self.checkpointer
        if self.enable_human_review and checkpointer is None:
            checkpointer = create_durable_checkpointer()
        return graph.compile(checkpointer=checkpointer)


def extract_question_concepts(question: str) -> list[str]:
    """Extract meaningful question terms for lightweight evidence-coverage checks."""

    stopwords = {
        "a", "an", "and", "are", "do", "does", "for", "how", "in", "is", "of", "on",
        "the", "to", "used", "using", "what", "which", "with",
    }
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return list(dict.fromkeys(token for token in tokens if token not in stopwords and len(token) > 2))


def access_allows(state: ResearchState, capability: str) -> bool:
    """Check a server-resolved capability stored in checkpoint-safe graph state."""

    profile = state.get("access_profile")
    if profile is None:
        return True
    capabilities = profile.get("capabilities", []) if isinstance(profile, dict) else []
    return capability in capabilities


def concept_is_covered(concept: str, evidence_text: str) -> bool:
    """Match a concept against evidence with small scientific acronym aliases."""

    aliases = {
        "gnn": ("gnn", "graph neural network"),
        "gnns": ("gnn", "graph neural network"),
        "ml": ("machine learning",),
        "ai": ("artificial intelligence", "machine learning"),
    }
    candidates = aliases.get(concept, (concept,))
    return any(candidate in evidence_text for candidate in candidates)


def question_requires_full_text(question: str) -> bool:
    """Detect questions whose requested detail is rarely supported by abstracts alone."""

    detail_terms = {
        "architecture", "compare", "comparison", "dataset", "datasets", "detailed",
        "experimental", "exact", "implementation", "limitation", "limitations", "mechanism",
        "method", "methods", "numerical", "performance", "quantitative", "reproduce",
        "reproducibility", "setup",
    }
    tokens = set(re.findall(r"[a-z]+", question.lower()))
    return bool(tokens & detail_terms)


def normalize_string_list(value: Any, maximum: int) -> list[str]:
    """Normalize reviewer-provided string lists and enforce workflow size limits."""

    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str) and (cleaned := " ".join(item.split())) and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized[:maximum]


def normalize_instruction(value: Any) -> str | None:
    """Normalize one reviewer directive while bounding checkpoint and prompt growth."""

    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned[:1000] or None


def apply_instruction_to_queries(queries: list[str], instruction: str) -> list[str]:
    """Add one focused search probe so new reviewer guidance affects retrieval immediately."""

    if not queries:
        return [instruction]
    guided_query = f"{queries[0]} {instruction}"
    combined = [*queries, guided_query]
    if len(combined) > 5:
        combined = [*queries[:4], guided_query]
    return list(dict.fromkeys(combined))


def make_search_subtasks(queries: list[str], iteration: int) -> list[ResearchSubtask]:
    """Translate a search-query batch into stable, independently executable subtasks."""

    return [
        ResearchSubtask(
            id=f"search-{iteration}-{index}",
            kind="literature_search",
            objective=f"Find literature for: {query}",
            iteration=iteration,
            query=query,
        )
        for index, query in enumerate(queries, start=1)
    ]


def build_research_graph(
    search_client: Any | None = None,
    vector_store: Any | None = None,
    **workflow_options: Any,
) -> Any:
    """Convenience constructor for the compiled research workflow."""

    return ResearchGraphBuilder(
        search_client=search_client,
        vector_store=vector_store,
        **workflow_options,
    ).compile()
