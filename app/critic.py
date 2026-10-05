from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from app.models import ResearchCritique
from app.settings import create_chat_client, get_chat_provider, provider_feature_enabled
from app.observability import log_event, metrics_increment, observed_chat_completion


class ResearchCritic(Protocol):
    """Dependency boundary for local rules today and an LLM critic later."""

    def critique(
        self,
        state: dict[str, Any],
        *,
        max_search_iterations: int,
        max_critic_retries: int,
    ) -> ResearchCritique:
        """Judge current evidence and recommend exactly one bounded next action."""


class LocalResearchCritic:
    """Deterministic critic that diagnoses evidence gaps without an API key."""

    def critique(
        self,
        state: dict[str, Any],
        *,
        max_search_iterations: int,
        max_critic_retries: int,
    ) -> ResearchCritique:
        """Reflect on coverage, source diversity, retrieval strength, and retry history."""

        iteration = state.get("search_iteration", 1)
        history = state.get("critique_history", [])
        retries_used = sum(item.verdict == "retry" for item in history)
        remaining = max(0, max_critic_retries - retries_used)
        passages = state.get("retrieved_passages", [])
        papers = state.get("retrieved_papers", [])
        missing = list(state.get("missing_concepts", []))
        notes = list(state.get("evidence_notes", []))
        source_count = len({passage.paper.id for passage in passages})
        weak_claims: list[str] = []
        if passages and max((passage.score for passage in passages), default=0.0) < 0.35:
            weak_claims.append("All retrieved passages have low hybrid relevance scores.")
        if passages and source_count < 2:
            weak_claims.append("Evidence is concentrated in a single paper.")

        signature_source = "|".join(sorted(missing) + sorted(weak_claims)) or "evidence-ready"
        signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()[:12]
        repeated = sum(item.signature == signature for item in history)
        detailed = bool(state.get("deep_read_requested")) and not state.get("fulltext_attempted")
        sufficient = bool(state.get("evidence_sufficient"))

        if sufficient and detailed:
            verdict = "deep_read"
            reason = "Abstract evidence is broad enough, but the question requires methodological detail."
        elif sufficient:
            verdict = "sufficient"
            reason = "Evidence volume, concept coverage, diversity, and relevance meet local thresholds."
        elif remaining and iteration < max_search_iterations and repeated < max_critic_retries:
            verdict = "retry"
            reason = "Evidence remains incomplete and a targeted search retry is still available."
        elif detailed:
            verdict = "deep_read"
            reason = "Search retries are exhausted; selected full text may resolve remaining detail gaps."
        else:
            verdict = "stop"
            reason = "Evidence is incomplete, but retry limits or repeated critique prevent another search."

        question = " ".join(str(state.get("question", "")).replace("?", "").split())
        focus = " ".join(missing[:4]) or "evidence comparison limitations"
        queries = []
        if verdict == "retry":
            angle = "benchmark comparison" if iteration == 1 else "limitations validation study"
            queries = [f"{question} {focus} {angle}", f"{focus} materials science {angle}"]
        return ResearchCritique(
            iteration=iteration,
            verdict=verdict,
            reason=reason,
            confidence=0.9 if verdict in {"sufficient", "deep_read"} else 0.75,
            missing_topics=missing,
            weak_claims=[*notes, *weak_claims],
            recommended_queries=queries,
            recommended_paper_ids=[paper.id for paper in papers[:3]] if verdict == "deep_read" else [],
            retry_budget_remaining=max(0, remaining - (1 if verdict == "retry" else 0)),
            signature=signature,
        )


class ProviderResearchCritic:
    """LLM-backed critic constrained by the deterministic critic's safety budget."""

    def __init__(self, fallback: LocalResearchCritic | None = None) -> None:
        self.fallback = fallback or LocalResearchCritic()

    def critique(
        self,
        state: dict[str, Any],
        *,
        max_search_iterations: int,
        max_critic_retries: int,
    ) -> ResearchCritique:
        """Request semantic reflection, then enforce local routing and retry constraints."""

        baseline = self.fallback.critique(
            state,
            max_search_iterations=max_search_iterations,
            max_critic_retries=max_critic_retries,
        )
        config = get_chat_provider()
        if config is None:
            return baseline
        passages = state.get("retrieved_passages", [])[:8]
        payload = {
            "question": state.get("question"),
            "iteration": baseline.iteration,
            "local_measurement": {
                "evidence_sufficient": state.get("evidence_sufficient", False),
                "coverage": state.get("evidence_coverage", 0.0),
                "missing_topics": state.get("missing_concepts", []),
                "notes": state.get("evidence_notes", []),
                "deep_read_requested": state.get("deep_read_requested", False),
                "fulltext_attempted": state.get("fulltext_attempted", False),
                "retry_budget_remaining": baseline.retry_budget_remaining,
            },
            "evidence": [
                {
                    "paper_id": passage.paper.id,
                    "title": passage.paper.title,
                    "text": passage.text,
                    "score": passage.score,
                }
                for passage in passages
            ],
        }
        try:
            response = observed_chat_completion(
                create_chat_client(),
                "research_critic",
                model=config.model,
                max_tokens=600,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict scientific research critic. Judge only the supplied evidence. "
                            "Return JSON with verdict (sufficient, retry, deep_read, or stop), reason, "
                            "confidence, missing_topics, weak_claims, contradictions, recommended_queries, "
                            "and recommended_paper_ids. Do not recommend retry when retry_budget_remaining "
                            "is zero. Prefer deep_read only when methodological detail is requested."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            data = json.loads(response.choices[0].message.content or "{}")
            verdict = str(data.get("verdict", baseline.verdict))
            allowed = {"sufficient", "retry", "deep_read", "stop"}
            if verdict not in allowed:
                verdict = baseline.verdict
            if verdict == "retry" and baseline.verdict != "retry":
                verdict = baseline.verdict
            if verdict == "deep_read" and not state.get("deep_read_requested"):
                verdict = baseline.verdict
            queries = [str(item).strip() for item in data.get("recommended_queries", []) if str(item).strip()]
            if verdict == "retry" and not queries:
                queries = baseline.recommended_queries
            return baseline.model_copy(
                update={
                    "verdict": verdict,
                    "reason": str(data.get("reason") or baseline.reason),
                    "confidence": min(max(float(data.get("confidence", baseline.confidence)), 0.0), 1.0),
                    "missing_topics": data.get("missing_topics", baseline.missing_topics),
                    "weak_claims": data.get("weak_claims", baseline.weak_claims),
                    "contradictions": data.get("contradictions", baseline.contradictions),
                    "recommended_queries": queries if verdict == "retry" else [],
                    "recommended_paper_ids": data.get(
                        "recommended_paper_ids", baseline.recommended_paper_ids
                    ),
                }
            )
        except Exception as exc:
            metrics_increment("errors")
            log_event("llm.critic_fallback", operation="research_critic", error_type=type(exc).__name__)
            return baseline


def create_default_critic() -> ResearchCritic:
    """Use semantic provider reflection when configured, otherwise remain fully local."""

    return ProviderResearchCritic() if provider_feature_enabled("critic") else LocalResearchCritic()
