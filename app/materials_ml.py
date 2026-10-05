from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, model_validator


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "materials" / "matbench_expt_gap.json.gz"
)


class MaterialRecord(BaseModel):
    """One provenance-bearing material property record from a local database."""

    formula: str
    band_gap_ev: float
    provenance: str = "Matbench matbench_expt_gap"
    value_type: str = "measured"


class CandidateScreenRequest(BaseModel):
    """Scientific constraints defining a material candidate screen."""

    min_band_gap_ev: float = Field(default=1.0, ge=0.0, le=20.0)
    max_band_gap_ev: float = Field(default=2.5, ge=0.0, le=20.0)
    include_elements: list[str] = Field(default_factory=list)
    exclude_elements: list[str] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_range(self) -> "CandidateScreenRequest":
        """Reject inverted property windows before any ranking work begins."""

        if self.min_band_gap_ev > self.max_band_gap_ev:
            raise ValueError("min_band_gap_ev must not exceed max_band_gap_ev")
        return self


class RankedCandidate(BaseModel):
    """One screened candidate with score and transparent selection reasons."""

    rank: int
    formula: str
    band_gap_ev: float
    score: float
    value_type: str
    provenance: str
    reasons: list[str]


class CandidateScreenResult(BaseModel):
    """Complete candidate-screening tool response including RAG-ready context."""

    property_name: str = "experimental band gap"
    property_unit: str = "eV"
    dataset_name: str
    dataset_size: int
    matched_count: int
    candidates: list[RankedCandidate]
    rag_context: str
    warnings: list[str] = Field(default_factory=list)


class PropertyPrediction(BaseModel):
    """Model estimate kept distinct from database measurements."""

    formula: str
    predicted_band_gap_ev: float
    uncertainty_ev: float
    model_name: str
    neighbor_formulas: list[str]


class MaterialPropertyPredictor(Protocol):
    """Adapter contract for local baselines and future pretrained structure models."""

    model_name: str

    def predict(self, formula: str) -> PropertyPrediction:
        """Predict a property with uncertainty and model provenance."""


class MatbenchBandGapDatabase:
    """Load and query the official composition-level Matbench experimental-gap data."""

    def __init__(self, path: str | Path = DEFAULT_DATASET_PATH) -> None:
        self.path = Path(path)
        self._records: list[MaterialRecord] | None = None

    def records(self) -> list[MaterialRecord]:
        """Load the compressed dataset once using only Python standard-library parsers."""

        if self._records is not None:
            return self._records
        if not self.path.exists():
            raise FileNotFoundError(f"Materials dataset not found: {self.path}")
        payload = json.loads(gzip.decompress(self.path.read_bytes()))
        columns = payload.get("columns", [])
        formula_index = columns.index("composition")
        gap_index = columns.index("gap expt")
        self._records = [
            MaterialRecord(formula=str(row[formula_index]), band_gap_ev=float(row[gap_index]))
            for row in payload.get("data", [])
        ]
        return self._records

    def screen(self, request: CandidateScreenRequest) -> CandidateScreenResult:
        """Apply chemistry and property constraints, then rank candidates near window center."""

        include = {normalize_element(item) for item in request.include_elements if item.strip()}
        exclude = {normalize_element(item) for item in request.exclude_elements if item.strip()}
        center = (request.min_band_gap_ev + request.max_band_gap_ev) / 2
        half_width = max((request.max_band_gap_ev - request.min_band_gap_ev) / 2, 0.05)
        matched: list[tuple[float, MaterialRecord, set[str]]] = []
        skipped = 0
        for record in self.records():
            composition = parse_formula(record.formula)
            if not composition:
                skipped += 1
                continue
            elements = set(composition)
            if not include.issubset(elements) or elements & exclude:
                continue
            if not request.min_band_gap_ev <= record.band_gap_ev <= request.max_band_gap_ev:
                continue
            proximity = max(0.0, 1.0 - abs(record.band_gap_ev - center) / half_width)
            simplicity = 1.0 / max(len(elements), 1)
            score = 0.85 * proximity + 0.15 * simplicity
            matched.append((score, record, elements))
        matched.sort(key=lambda item: (-item[0], item[1].formula))
        candidates = [
            RankedCandidate(
                rank=index,
                formula=record.formula,
                band_gap_ev=record.band_gap_ev,
                score=round(score, 4),
                value_type=record.value_type,
                provenance=record.provenance,
                reasons=[
                    f"Measured gap lies within {request.min_band_gap_ev:g}-{request.max_band_gap_ev:g} eV",
                    f"Contains required elements: {', '.join(sorted(include))}" if include else "No required-element constraint",
                    f"Contains {len(elements)} distinct element(s)",
                ],
            )
            for index, (score, record, elements) in enumerate(matched[: request.top_k], start=1)
        ]
        warnings = []
        if skipped:
            warnings.append(f"Skipped {skipped} formulas that the local parser could not validate.")
        return CandidateScreenResult(
            dataset_name="Matbench v0.1 / matbench_expt_gap",
            dataset_size=len(self.records()),
            matched_count=len(matched),
            candidates=candidates,
            rag_context=format_material_rag_context(request, candidates),
            warnings=warnings,
        )


class CompositionKNNBandGapModel:
    """Dependency-free composition nearest-neighbor baseline backed by Matbench records."""

    model_name = "composition-knn-matbench-expt-gap-v1"

    def __init__(self, database: MatbenchBandGapDatabase | None = None, neighbors: int = 7) -> None:
        self.database = database or MatbenchBandGapDatabase()
        self.neighbors = max(2, neighbors)

    def predict(self, formula: str) -> PropertyPrediction:
        """Estimate band gap from composition similarity and neighbor disagreement."""

        query = element_fractions(parse_formula(formula))
        if not query:
            raise ValueError(f"Could not parse chemical formula: {formula}")
        distances: list[tuple[float, MaterialRecord]] = []
        for record in self.database.records():
            vector = element_fractions(parse_formula(record.formula))
            if not vector:
                continue
            distances.append((composition_distance(query, vector), record))
        nearest = sorted(distances, key=lambda item: item[0])[: self.neighbors]
        weights = [1.0 / max(distance, 0.02) for distance, _ in nearest]
        prediction = sum(weight * item.band_gap_ev for weight, (_, item) in zip(weights, nearest)) / sum(weights)
        variance = sum(
            weight * (item.band_gap_ev - prediction) ** 2
            for weight, (_, item) in zip(weights, nearest)
        ) / sum(weights)
        return PropertyPrediction(
            formula=formula,
            predicted_band_gap_ev=round(prediction, 4),
            uncertainty_ev=round(math.sqrt(variance), 4),
            model_name=self.model_name,
            neighbor_formulas=[item.formula for _, item in nearest],
        )


class MaterialsScreeningTool:
    """Agent-callable facade for database screening and composition-model prediction."""

    def __init__(self, database: MatbenchBandGapDatabase | None = None) -> None:
        self.database = database or MatbenchBandGapDatabase()
        self.predictor: MaterialPropertyPredictor = CompositionKNNBandGapModel(self.database)

    def screen_candidates(self, request: CandidateScreenRequest) -> CandidateScreenResult:
        """Return ranked measured candidates and RAG-ready structured evidence."""

        return self.database.screen(request)

    def predict_band_gaps(self, formulas: list[str]) -> list[PropertyPrediction]:
        """Predict arbitrary formulas while clearly labeling model-derived values."""

        return [self.predictor.predict(formula) for formula in formulas[:50]]


def parse_formula(formula: str) -> dict[str, float]:
    """Parse flat chemical formulas into element counts without guessing grouped notation."""

    compact = formula.replace(" ", "")
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", compact)
    if not tokens or "".join(f"{element}{count}" for element, count in tokens) != compact:
        return {}
    composition: dict[str, float] = {}
    for element, raw_count in tokens:
        composition[element] = composition.get(element, 0.0) + (float(raw_count) if raw_count else 1.0)
    return composition


def element_fractions(composition: dict[str, float]) -> dict[str, float]:
    """Normalize element counts for scale-invariant composition comparison."""

    total = sum(composition.values())
    return {element: count / total for element, count in composition.items()} if total else {}


def composition_distance(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute L1 distance over the union of composition-fraction features."""

    return sum(abs(left.get(element, 0.0) - right.get(element, 0.0)) for element in left | right)


def normalize_element(value: str) -> str:
    """Normalize and validate an element symbol used in screening constraints."""

    symbol = value.strip().capitalize()
    if not re.fullmatch(r"[A-Z][a-z]?", symbol):
        raise ValueError(f"Invalid element symbol: {value}")
    return symbol


def format_material_rag_context(
    request: CandidateScreenRequest, candidates: list[RankedCandidate]
) -> str:
    """Render measured candidate records as instruction-safe structured RAG context."""

    lines = [
        "MATERIAL DATA CONTEXT",
        "These are database measurements, not literature claims or model predictions.",
        f"Screen: experimental band gap {request.min_band_gap_ev:g}-{request.max_band_gap_ev:g} eV.",
    ]
    lines.extend(
        f"[M{item.rank}] formula={item.formula}; measured_band_gap={item.band_gap_ev:g} eV; "
        f"source={item.provenance}"
        for item in candidates
    )
    return "\n".join(lines)
