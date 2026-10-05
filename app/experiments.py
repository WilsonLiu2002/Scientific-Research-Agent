from __future__ import annotations

import random
from statistics import mean, stdev
from typing import Any


class SyntheticExperimentService:
    """Create reproducible local benchmark data for UI demonstrations only."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run_crystal_benchmark(self) -> dict[str, Any]:
        """Simulate repeated formation-energy prediction runs without external data."""

        rng = random.Random(self.seed)
        specifications = [
            ("Ridge regression", 0.118, 0.9, 0x617A8A),
            ("Random forest", 0.083, 6.8, 0xBE7C3F),
            ("Crystal graph network", 0.052, 31.4, 0x28715D),
            ("Equivariant GNN", 0.041, 54.7, 0xC6523E),
        ]
        models: list[dict[str, Any]] = []
        for name, baseline_mae, runtime, color in specifications:
            runs = [max(0.01, rng.gauss(baseline_mae, baseline_mae * 0.075)) for _ in range(5)]
            models.append(
                {
                    "name": name,
                    "mae_runs": [round(value, 4) for value in runs],
                    "mean_mae": round(mean(runs), 4),
                    "std_mae": round(stdev(runs), 4),
                    "runtime_seconds": runtime,
                    "color": f"#{color:06x}",
                }
            )

        best = min(models, key=lambda item: item["mean_mae"])
        baseline = models[0]
        improvement = 100 * (baseline["mean_mae"] - best["mean_mae"]) / baseline["mean_mae"]
        return {
            "id": f"synthetic-crystal-{self.seed}",
            "title": "Formation-energy prediction benchmark",
            "status": "synthetic",
            "seed": self.seed,
            "dataset": {
                "name": "Synthetic crystal structures",
                "samples": 2400,
                "train_samples": 1920,
                "test_samples": 480,
                "target": "Formation energy (eV/atom)",
                "splits": 5,
            },
            "models": models,
            "summary": (
                f"{best['name']} achieved the lowest simulated MAE at {best['mean_mae']:.3f} "
                f"eV/atom, a {improvement:.1f}% reduction relative to ridge regression."
            ),
            "provenance": (
                "Generated locally from a seeded statistical simulation. These values are illustrative, "
                "not measured results and not extracted from scientific literature."
            ),
        }
