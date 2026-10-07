from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.models import EvidencePassage, Paper, ResearchIntent
from app.observability import log_event, metrics_set, span


class EmbeddingFunction(Protocol):
    """Protocol shared by local and API-backed embedding functions."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of strings into vectors."""


class LocalHashEmbeddingFunction:
    """Deterministic local embedding function for tests and no-key demos."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed texts by hashing tokens into a fixed-size normalized vector."""

        return [self.embed_text(text) for text in input]

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one query string for Chroma's current embedding protocol."""

        if isinstance(input, list):
            return self(input)
        return self.embed_text(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed document strings for Chroma's current embedding protocol."""

        return self(input)

    def name(self) -> str:
        """Return a stable Chroma embedding-function name."""

        return "local_hash_embedding"

    def embed_text(self, text: str) -> list[float]:
        """Create one normalized vector using a lightweight hashing trick."""

        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingFunction:
    """OpenAI-backed embedding function used when `EMBEDDING_PROVIDER=openai`."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed texts by calling the configured OpenAI embedding model."""

        response = self.client.embeddings.create(model=self.model, input=input)
        return [item.embedding for item in response.data]

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one query string for Chroma's current embedding protocol."""

        if isinstance(input, list):
            return self(input)
        return self([input])[0]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed document strings for Chroma's current embedding protocol."""

        return self(input)

    def name(self) -> str:
        """Return a stable Chroma embedding-function name."""

        return f"openai_{self.model}"


@dataclass
class InMemoryVectorStore:
    """Small vector store fallback with the same index/query behavior needed by tests."""

    embedding_function: EmbeddingFunction = field(default_factory=LocalHashEmbeddingFunction)
    passages: list[EvidencePassage] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    def index_papers(self, papers: list[Paper]) -> int:
        """Index papers that contain abstracts and return the number inserted."""

        self.passages = passages_from_papers(papers)
        self.embeddings = self.embedding_function([passage.text for passage in self.passages])
        return len({passage.paper.id for passage in self.passages})

    def retrieve(self, question: str, top_k: int = 8) -> list[EvidencePassage]:
        """Return passages using a hybrid semantic and keyword relevance score."""

        if not self.passages:
            return []
        query_embedding = self.embedding_function([question])[0]
        ranked = sorted(
            zip(self.passages, self.embeddings),
            key=lambda item: hybrid_score(question, item[0].text, query_embedding, item[1]),
            reverse=True,
        )
        return [
            passage.model_copy(
                update={"score": hybrid_score(question, passage.text, query_embedding, embedding)}
            )
            for passage, embedding in ranked[:top_k]
        ]


class ChromaVectorStore:
    """Chroma-backed store for passage-level hybrid RAG retrieval."""

    def __init__(
        self,
        collection_name: str = "scientific_papers",
        embedding_function: EmbeddingFunction | None = None,
        persist_directory: str | None = None,
    ) -> None:
        import chromadb

        self.embedding_function = embedding_function or select_embedding_function()
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def index_papers(self, papers: list[Paper]) -> int:
        """Insert abstract-bearing papers into Chroma with citation metadata."""

        passages = passages_from_papers(papers)
        if not passages:
            return 0

        ids = [passage.id for passage in passages]
        documents = [passage.text for passage in passages]
        metadatas: list[dict[str, Any]] = [
            {
                "paper_id": passage.paper.id,
                "title": passage.paper.title,
                "authors": ", ".join(passage.paper.authors),
                "year": passage.paper.year or "",
                "doi": passage.paper.doi or "",
                "venue": passage.paper.venue or "",
                "url": passage.paper.url or "",
                "citation_count": passage.paper.citation_count or 0,
                "source": passage.paper.source,
            }
            for passage in passages
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len({passage.paper.id for passage in passages})

    def retrieve(self, question: str, top_k: int = 8) -> list[EvidencePassage]:
        """Query Chroma, then rerank vector candidates with lexical relevance."""

        if self.collection.count() == 0:
            return []
        candidate_count = min(self.collection.count(), max(top_k * 3, top_k))
        results = self.collection.query(query_texts=[question], n_results=candidate_count)
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        distances = results.get("distances", [[]])[0]
        passages: list[EvidencePassage] = []
        for passage_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            paper = Paper(
                id=str(metadata.get("paper_id") or passage_id),
                title=str(metadata.get("title") or "Untitled"),
                abstract=document,
                authors=parse_authors(metadata.get("authors")),
                year=parse_optional_int(metadata.get("year")),
                doi=str(metadata.get("doi") or "") or None,
                venue=str(metadata.get("venue") or "") or None,
                url=str(metadata.get("url") or "") or None,
                citation_count=parse_optional_int(metadata.get("citation_count")),
                source=str(metadata.get("source") or "Chroma"),
            )
            semantic_score = 1.0 / (1.0 + float(distance or 0.0))
            score = 0.7 * semantic_score + 0.3 * lexical_overlap(question, document)
            passages.append(
                EvidencePassage(id=passage_id, paper=paper, text=document, score=score)
            )
        return sorted(passages, key=lambda passage: passage.score, reverse=True)[:top_k]


def passages_from_papers(papers: list[Paper], max_chars: int = 700) -> list[EvidencePassage]:
    """Split abstracts into sentence-aware, stable passages suitable for citations."""

    passages: list[EvidencePassage] = []
    for paper in papers:
        if not paper.abstract:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paper.abstract.strip())
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        for index, chunk in enumerate(chunks, start=1):
            passages.append(EvidencePassage(id=f"{paper.id}::p{index}", paper=paper, text=chunk))
    return passages


def lexical_overlap(query: str, text: str) -> float:
    """Calculate normalized token overlap for the keyword half of hybrid retrieval."""

    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_tokens & text_tokens) / max(len(query_tokens), 1)


def hybrid_score(
    query: str, text: str, query_embedding: list[float], text_embedding: list[float]
) -> float:
    """Blend vector similarity with exact token overlap to improve technical-term recall."""

    return 0.7 * cosine_similarity(query_embedding, text_embedding) + 0.3 * lexical_overlap(query, text)


def build_retrieval_probes(question: str, intent: ResearchIntent | None) -> list[str]:
    """Translate research intent into complementary passage-retrieval probes."""

    probes = [" ".join(question.replace("?", "").split())]
    if intent is None:
        return probes
    concepts = [*intent.entities, *intent.properties, *intent.methods]
    if concepts:
        probes.append(" ".join(dict.fromkeys(concepts)))
    purpose_terms = {
        "overview": "major approaches applications review",
        "methods": "method architecture model experimental setup protocol",
        "mechanism": "mechanism causal pathway explanation evidence",
        "comparison": "benchmark comparison baseline performance tradeoff",
        "quantitative": "quantitative results accuracy error metric evaluation",
        "datasets": "dataset samples measurements train test split",
        "limitations": "limitations failure generalization bias uncertainty challenge",
        "reproducibility": "reproducibility implementation hyperparameters protocol code data",
    }
    probes.append(f"{' '.join(concepts[:8])} {purpose_terms[intent.purpose]}".strip())
    for requirement in intent.evidence_requirements[:2]:
        probes.append(f"{' '.join(concepts[:6])} {requirement}".strip())
    return list(dict.fromkeys(probe for probe in probes if probe))[:5]


def retrieve_for_intent(
    vector_store: Any,
    question: str,
    intent: ResearchIntent | None,
    top_k: int = 8,
) -> tuple[list[EvidencePassage], list[str]]:
    """Retrieve with multiple intent probes and fuse rankings with purpose alignment."""

    probes = build_retrieval_probes(question, intent)
    embedding = getattr(vector_store, "embedding_function", None)
    embedding_name = embedding.name() if callable(getattr(embedding, "name", None)) else type(embedding).__name__
    with span(
        "rag.retrieve_for_intent",
        kind="retriever",
        fields={"top_k": top_k, "probe_count": len(probes), "embedding_model": embedding_name},
    ):
        fused: dict[str, dict[str, Any]] = {}
        candidate_count = max(top_k * 2, 12)
        for probe_index, probe in enumerate(probes):
            query_id = hashlib.sha256(probe.encode("utf-8")).hexdigest()[:12]
            for rank, passage in enumerate(vector_store.retrieve(probe, top_k=candidate_count), start=1):
                item = fused.setdefault(
                    passage.id,
                    {"passage": passage, "rrf": 0.0, "best": passage.score, "anchor": 0.0},
                )
                item["rrf"] += 1.0 / (60 + rank)
                item["best"] = max(item["best"], passage.score)
                if probe_index == 0:
                    item["anchor"] = passage.score
            log_event("rag.probe_completed", query_id=query_id, top_k=candidate_count)

    probe_count = max(len(probes), 1)
    ranked: list[EvidencePassage] = []
    for item in fused.values():
        passage = item["passage"]
        role_score = purpose_alignment(intent, f"{passage.paper.title} {passage.text}")
        fused_score = (
            0.40 * item["anchor"]
            + 0.30 * item["best"]
            + 0.20 * min(item["rrf"] * 60 / probe_count, 1.0)
            + 0.10 * role_score
        )
        ranked.append(passage.model_copy(update={"score": fused_score}))
    selected = sorted(ranked, key=lambda passage: passage.score, reverse=True)[:top_k]
    metrics_set("embedding_model", embedding_name)
    log_event(
        "rag.retrieval_result",
        top_k=top_k,
        passage_ids=[passage.id for passage in selected],
        scores=[round(passage.score, 6) for passage in selected],
    )
    return selected, probes


def purpose_alignment(intent: ResearchIntent | None, text: str) -> float:
    """Score whether a passage contains cues for the requested evidence role."""

    if intent is None:
        return 0.0
    cues = {
        "overview": ("review", "overview", "application", "approach"),
        "methods": ("method", "architecture", "model", "experimental", "protocol", "layer"),
        "mechanism": ("mechanism", "causal", "pathway", "because", "explain"),
        "comparison": ("compare", "benchmark", "baseline", "outperform", "tradeoff"),
        "quantitative": ("accuracy", "error", "mae", "rmse", "percent", "performance"),
        "datasets": ("dataset", "samples", "database", "train", "test", "measurement"),
        "limitations": ("limitation", "failure", "bias", "uncertainty", "generalization", "challenge"),
        "reproducibility": ("reproduc", "implementation", "hyperparameter", "code", "protocol"),
    }
    lowered = text.lower()
    matched = sum(term in lowered for term in cues[intent.purpose])
    return min(matched / 2, 1.0)


def select_embedding_function() -> EmbeddingFunction:
    """Choose OpenAI embeddings when configured, otherwise use the local deterministic fallback."""

    if os.getenv("EMBEDDING_PROVIDER", "local").lower() == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddingFunction()
    return LocalHashEmbeddingFunction()


def create_vector_store(
    force_memory: bool = False, access_scope: str = "principal_investigator"
) -> ChromaVectorStore | InMemoryVectorStore:
    """Create a persistent embedded index, falling back to memory when Chroma is absent."""

    if force_memory:
        return InMemoryVectorStore()
    safe_scope = re.sub(r"[^a-z0-9_-]", "-", access_scope.lower())[:40]
    try:
        return ChromaVectorStore(
            collection_name=f"scientific_papers_{safe_scope}",
            persist_directory=os.getenv("RAG_PERSIST_DIRECTORY", ".cache/chroma")
        )
    except (ImportError, ModuleNotFoundError):
        return InMemoryVectorStore()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two vectors."""

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


def extract_abstract(document: str) -> str | None:
    """Extract the abstract portion from the simple title-plus-abstract document format."""

    marker = "Abstract:\n"
    if marker not in document:
        return document or None
    abstract = document.split(marker, 1)[1].strip()
    return abstract or None


def parse_authors(value: Any) -> list[str]:
    """Parse comma-separated authors stored in Chroma metadata."""

    if not value:
        return []
    return [author.strip() for author in str(value).split(",") if author.strip()]


def parse_optional_int(value: Any) -> int | None:
    """Convert Chroma metadata values back to optional integers."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
