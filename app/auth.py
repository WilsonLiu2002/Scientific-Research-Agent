from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccessGrant:
    """Resolved capabilities for one opaque simulated user-level token."""

    token_id: str
    level: str
    label: str
    capabilities: frozenset[str]

    def allows(self, capability: str) -> bool:
        """Return whether this grant permits one RAG, tool, or database action."""

        return capability in self.capabilities

    def public_payload(self) -> dict[str, object]:
        """Expose non-secret authorization details suitable for state and UI display."""

        return {
            "token_id": self.token_id,
            "level": self.level,
            "label": self.label,
            "capabilities": sorted(self.capabilities),
        }


SIMULATED_ACCESS_GRANTS = {
    "sim-local-reader": AccessGrant(
        token_id="sim-local-reader",
        level="local_reader",
        label="Local Reader",
        capabilities=frozenset({"rag:read"}),
    ),
    "sim-researcher": AccessGrant(
        token_id="sim-researcher",
        level="researcher",
        label="Researcher",
        capabilities=frozenset({"rag:read", "literature:search", "materials:read"}),
    ),
    "sim-principal-investigator": AccessGrant(
        token_id="sim-principal-investigator",
        level="principal_investigator",
        label="Principal Investigator",
        capabilities=frozenset(
            {"rag:read", "literature:search", "materials:read", "fulltext:read"}
        ),
    ),
}

ACCESS_LEVEL_RANKS = {
    "local_reader": 1,
    "researcher": 2,
    "principal_investigator": 3,
}


def access_level_allows(actual_level: str, required_level: str) -> bool:
    """Apply the simulated hierarchical clearance policy to one protected resource."""

    return ACCESS_LEVEL_RANKS.get(actual_level, 0) >= ACCESS_LEVEL_RANKS.get(required_level, 99)


def resolve_access_token(token: str | None) -> AccessGrant:
    """Resolve a demo token without trusting a browser-provided role or capability list."""

    grant = SIMULATED_ACCESS_GRANTS.get(token or "")
    if grant is None:
        raise PermissionError("Unknown simulated authorization token")
    return grant


def default_access_grant() -> AccessGrant:
    """Keep CLI and existing integrations fully capable unless they select a demo tier."""

    return SIMULATED_ACCESS_GRANTS["sim-principal-investigator"]
