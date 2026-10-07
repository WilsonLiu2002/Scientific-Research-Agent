from __future__ import annotations

import pytest

from app.auth import access_level_allows, resolve_access_token


def test_simulated_tokens_resolve_server_side_capabilities() -> None:
    """Resolve capability sets from tokens rather than trusting a client-provided role."""

    local = resolve_access_token("sim-local-reader")
    principal = resolve_access_token("sim-principal-investigator")

    assert local.allows("rag:read")
    assert not local.allows("literature:search")
    assert principal.allows("fulltext:read")


def test_unknown_simulated_token_is_rejected() -> None:
    """Reject arbitrary role strings that were not issued by the simulation."""

    with pytest.raises(PermissionError):
        resolve_access_token("admin")


def test_access_levels_are_hierarchical() -> None:
    """Allow higher simulated clearances to open lower-level workspaces only."""

    assert access_level_allows("principal_investigator", "researcher")
    assert access_level_allows("researcher", "local_reader")
    assert not access_level_allows("local_reader", "researcher")
