"""Discovery parses a provider's model list once per MODELS_CACHE_TTL."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from tokonomics.model_discovery import clear_models_cache, get_all_models
from tokonomics.model_discovery.model_info import ModelInfo
from tokonomics.model_discovery.modelsdev_provider import ModelsDevProvider


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_models_cache()
    yield
    clear_models_cache()


def _models() -> list[ModelInfo]:
    old = datetime.now() - timedelta(days=400)
    return [
        ModelInfo(id="new-model", name="New", provider="acme"),
        ModelInfo(id="old-model", name="Old", provider="acme", created_at=old),
    ]


async def test_repeated_discovery_parses_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def get_models(self: ModelsDevProvider) -> list[ModelInfo]:
        calls.append("fetch")
        return _models()

    monkeypatch.setattr(ModelsDevProvider, "get_models", get_models)
    first = await get_all_models(providers=["models.dev"])
    recent = await get_all_models(providers=["models.dev"], max_age=timedelta(days=200))
    assert calls == ["fetch"]  # The second call filtered the cached list
    assert [m.id for m in first] == ["new-model", "old-model"]
    assert [m.id for m in recent] == ["new-model"]


async def test_other_headers_are_fetched_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []
    token = "first"
    original_init = ModelsDevProvider.__init__

    def init(self: ModelsDevProvider) -> None:
        original_init(self)
        self.headers = {"Authorization": f"Bearer {token}"}

    async def get_models(self: ModelsDevProvider) -> list[ModelInfo]:
        calls.append(dict(self.headers))
        return _models()

    monkeypatch.setattr(ModelsDevProvider, "__init__", init)
    monkeypatch.setattr(ModelsDevProvider, "get_models", get_models)
    await get_all_models(providers=["models.dev"])
    token = "second"  # Another API key may see other models
    await get_all_models(providers=["models.dev"])
    assert [c["Authorization"] for c in calls] == ["Bearer first", "Bearer second"]


async def test_failed_fetch_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [[], _models()]

    async def get_models(self: ModelsDevProvider) -> list[ModelInfo]:
        return results.pop(0)

    monkeypatch.setattr(ModelsDevProvider, "get_models", get_models)
    assert await get_all_models(providers=["models.dev"]) == []
    assert [m.id for m in await get_all_models(providers=["models.dev"])] == [
        "new-model",
        "old-model",
    ]
