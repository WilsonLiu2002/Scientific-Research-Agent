from __future__ import annotations

import re

from app.models import DocumentSection, EvidenceClaim, FullTextDocument, PaperReading
from app.rag import lexical_overlap


class PaperEvidenceExtractor:
    """Interpret acquired article text into question-focused, traceable evidence."""

    def __init__(self, max_claims: int = 12, max_passage_chars: int = 900) -> None:
        self.max_claims = max_claims
        self.max_passage_chars = max_passage_chars

    def extract(
        self,
        document: FullTextDocument,
        research_question: str,
        sections: list[str] | None = None,
    ) -> PaperReading:
        """Rank article passages by relevance and group them by evidence role."""

        requested = {name.lower() for name in sections or []}
        candidates: list[tuple[float, DocumentSection, str]] = []
        for section in document.sections:
            if requested and not any(name in section.heading.lower() for name in requested):
                continue
            for passage in split_section(section.text, self.max_passage_chars):
                score = lexical_overlap(research_question, f"{section.heading} {passage}")
                candidates.append((score, section, passage))
        candidates.sort(key=lambda item: item[0], reverse=True)

        reading = PaperReading(
            document_id=document.document_id,
            research_question=research_question,
        )
        for _, section, passage in candidates[: self.max_claims]:
            claim = EvidenceClaim(
                text=passage,
                section=section.heading,
                page=section.page,
                source_url=document.source_url,
            )
            getattr(reading, classify_section(section.heading, passage)).append(claim)
        return reading


def split_section(text: str, max_chars: int = 900) -> list[str]:
    """Split one article section into bounded, sentence-aware evidence passages."""

    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    passages: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            passages.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        passages.append(current)
    return passages


def classify_section(heading: str, text: str) -> str:
    """Map an extractive passage to a stable evidence category."""

    value = f"{heading} {text}".lower()
    rules = {
        "limitations": ("limitation", "future work", "caveat"),
        "methods": ("method", "model", "architecture", "experimental", "simulation"),
        "datasets": ("dataset", "data set", "training data", "database"),
        "results": ("result", "performance", "accuracy", "error", "improve"),
        "conclusions": ("conclusion", "discussion", "summary"),
    }
    for category, terms in rules.items():
        if any(term in value for term in terms):
            return category
    return "conclusions"
