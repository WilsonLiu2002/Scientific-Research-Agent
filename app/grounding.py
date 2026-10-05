from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel

from app.models import EvidencePassage
from app.rag import lexical_overlap
from app.settings import create_chat_client, get_chat_provider, provider_feature_enabled
from app.observability import observed_chat_completion


class ClaimVerification(BaseModel):
    """Auditable verdict linking one answer claim to its cited evidence."""

    line_number: int
    claim: str
    citations: list[str]
    verdict: Literal["entailed", "partial", "unsupported"]
    explanation: str


class GroundingVerifier:
    """Verify scientific claims semantically when configured, with a local baseline."""

    def verify(
        self, answer: str, passages: list[EvidencePassage]
    ) -> list[ClaimVerification]:
        """Return one entailment verdict for every citable scientific claim line."""

        claims = extract_scientific_claims(answer)
        if not claims or not passages:
            return []
        if provider_feature_enabled("grounding"):
            try:
                return self._verify_openai(claims, passages)
            except Exception:
                pass
        return self._verify_locally(claims, passages)

    def _verify_locally(
        self,
        claims: list[tuple[int, str, list[str]]],
        passages: list[EvidencePassage],
    ) -> list[ClaimVerification]:
        """Use transparent token overlap as the deterministic offline baseline."""

        evidence_by_label = {f"P{index}": item.text for index, item in enumerate(passages, start=1)}
        results: list[ClaimVerification] = []
        for line_number, claim, citations in claims:
            evidence = " ".join(evidence_by_label.get(label, "") for label in citations)
            score = lexical_overlap(strip_citations(claim), evidence)
            verdict: Literal["entailed", "partial", "unsupported"]
            if score >= 0.70:
                verdict = "entailed"
            elif score >= 0.50:
                verdict = "partial"
            else:
                verdict = "unsupported"
            results.append(
                ClaimVerification(
                    line_number=line_number,
                    claim=claim,
                    citations=citations,
                    verdict=verdict,
                    explanation=f"Offline lexical support score: {score:.2f}",
                )
            )
        return results

    def _verify_openai(
        self,
        claims: list[tuple[int, str, list[str]]],
        passages: list[EvidencePassage],
    ) -> list[ClaimVerification]:
        """Use an LLM as a strict claim-to-cited-evidence entailment judge."""

        evidence = {f"P{index}": item.text for index, item in enumerate(passages, start=1)}
        payload = [
            {
                "line_number": number,
                "claim": strip_citations(claim),
                "citations": citations,
                "evidence": {label: evidence.get(label, "") for label in citations},
            }
            for number, claim, citations in claims
        ]
        config = get_chat_provider()
        if config is None:
            raise RuntimeError("No chat provider is configured.")
        response = observed_chat_completion(
            create_chat_client(),
            "grounding_verification",
            model=config.model,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Judge whether each scientific claim is entailed by only its cited evidence. "
                        "Return JSON with a `verdicts` array. Each item must preserve line_number, claim, "
                        "and citations and provide verdict as entailed, partial, or unsupported plus a "
                        "brief explanation. Be strict about numbers, causality, and comparisons."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return [ClaimVerification.model_validate(item) for item in data.get("verdicts", [])]


def repair_unsupported_claims(
    answer: str, verdicts: list[ClaimVerification]
) -> tuple[str, list[str]]:
    """Perform one bounded repair pass by removing unsupported scientific lines."""

    unsupported = {item.line_number: item for item in verdicts if item.verdict == "unsupported"}
    warnings = [
        f"Line {number} was removed after grounding verification: {item.explanation}."
        for number, item in unsupported.items()
    ]
    repaired = "\n".join(
        line for number, line in enumerate(answer.splitlines(), start=1) if number not in unsupported
    ).strip()
    return repaired, warnings


def extract_scientific_claims(answer: str) -> list[tuple[int, str, list[str]]]:
    """Extract cited, non-reference lines that require evidence entailment checks."""

    headings = {
        "Overview", "Main Approaches", "Applications / Predicted Properties",
        "Key Observations", "Limitations", "References",
    }
    meta_markers = (
        "the evidence below", "see the cited evidence", "only abstracts were searched",
        "selected open-access full-text", "no papers with usable abstracts",
    )
    claims: list[tuple[int, str, list[str]]] = []
    in_references = False
    for line_number, line in enumerate(answer.splitlines(), start=1):
        stripped = line.strip()
        if stripped == "References":
            in_references = True
            continue
        if not stripped or stripped in headings or in_references:
            continue
        if any(marker in stripped.lower() for marker in meta_markers):
            continue
        citations = re.findall(r"\[(P\d+)\]", stripped)
        if citations:
            claims.append((line_number, stripped, citations))
    return claims


def strip_citations(text: str) -> str:
    """Remove evidence labels before comparing claim language with evidence text."""

    return re.sub(r"\s*\[P\d+\]", "", text).lstrip("- ")
