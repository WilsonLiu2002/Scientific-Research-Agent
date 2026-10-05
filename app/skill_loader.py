from __future__ import annotations

from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"


def load_skill(name: str) -> str:
    """Load a skill's Markdown instructions by name from the local skills directory."""

    skill_path = SKILLS_ROOT / name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill not found: {name}")
    return skill_path.read_text(encoding="utf-8")
