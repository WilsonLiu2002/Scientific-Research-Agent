from __future__ import annotations

import json
from pathlib import Path

from app.models import Paper


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "literature" / "seed_corpus.json"


class LocalLiteratureCorpus:
    """Load the durable, versioned literature records used by local RAG."""

    def __init__(self, path: Path = DEFAULT_CORPUS_PATH) -> None:
        self.path = path

    def papers(self) -> list[Paper]:
        """Return validated papers from disk so retrieval never depends only on live search."""

        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [Paper.model_validate(item) for item in payload]
