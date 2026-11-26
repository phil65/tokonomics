"""Tests for the Copilot token manager."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import anyenv
import pytest

from tokonomics.model_discovery.copilot_provider.token_manager import CopilotTokenManager


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> tuple[CopilotTokenManager, list[int]]:
    calls: list[int] = []

    async def fake_get_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        await asyncio.sleep(0.1)
        return {"token": f"tok-{len(calls)}", "expires_at": time.time() + 3600}

    monkeypatch.setattr(anyenv, "get_json", fake_get_json)
    m = CopilotTokenManager()
    m._github_oauth_token = "gho_test"
    return m, calls


async def test_concurrent_refresh_does_not_block_the_loop(manager):
    """Two concurrent refreshes used to deadlock (thread lock held across await)."""
    m, calls = manager
    result: list[Any] = []

    def run() -> None:
        async def main() -> list[str]:
            return await asyncio.wait_for(asyncio.gather(m.get_token(), m.get_token()), 3)

        result.append(asyncio.run(main()))

    # Run in a separate thread: a deadlocked loop must not hang the test process.
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "event loop deadlocked in get_token"
    assert result == [["tok-1", "tok-1"]]
    assert len(calls) == 1  # single refresh for both callers


async def test_token_reused_while_valid(manager):
    m, calls = manager
    assert await m.get_token() == "tok-1"
    assert await m.get_token() == "tok-1"
    assert len(calls) == 1


async def test_provider_uses_endpoint_from_token_response(monkeypatch: pytest.MonkeyPatch):
    """The models URL comes from the token's ``endpoints.api``, fetched before the request."""
    from tokonomics.model_discovery.copilot_provider import provider as provider_mod

    m = CopilotTokenManager()
    m._github_oauth_token = "gho_test"

    async def fake_get_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "token": "tok",
            "expires_at": time.time() + 3600,
            "endpoints": {"api": "https://api.business.githubcopilot.com"},
        }

    urls: list[str] = []

    class FakeResponse:
        async def json(self) -> dict[str, Any]:
            return {"data": []}

    async def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(anyenv, "get_json", fake_get_json)
    monkeypatch.setattr(anyenv, "get", fake_get)
    monkeypatch.setattr(provider_mod, "token_manager", m)
    await provider_mod.CopilotProvider().get_models()
    assert urls == ["https://api.business.githubcopilot.com/models"]
