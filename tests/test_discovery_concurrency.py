"""Discovery fetches run concurrently, don't block the loop and are fetched once."""

import asyncio
import sys
import time
import types
from typing import TYPE_CHECKING, Any

import anyenv
import pytest

from tokonomics import token_count
from tokonomics.model_discovery import (
    cache,
    clear_models_cache,
    get_all_models,
    github_provider as _gh,  # noqa: F401
)
from tokonomics.model_discovery.github_provider import provider as gh_mod
from tokonomics.model_discovery.modelsdev_provider import ModelsDevProvider, provider as md_mod
from tokonomics.model_discovery.ollama_provider.provider import OllamaProvider


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_models_cache()
    md_mod.clear_api_cache()
    yield
    clear_models_cache()
    md_mod.clear_api_cache()


class _Resp:
    """Like anyenv 3's HttpResponse: the body is read, json() is sync."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def json(self) -> Any:
        return self._data


async def test_github_fetches_free_and_paid_concurrently(monkeypatch: pytest.MonkeyPatch):
    active = 0
    peak = 0

    async def fake_post(url: str, **kwargs: Any) -> _Resp:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return _Resp({"summaries": []})

    monkeypatch.setattr(anyenv, "post", fake_post)
    await gh_mod.GitHubProvider(token="t").get_models()
    assert peak == 2  # noqa: PLR2004


async def test_ollama_show_requests_run_concurrently(monkeypatch: pytest.MonkeyPatch):
    active = 0
    peak = 0
    tag_kwargs: dict[str, Any] = {}

    async def fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
        tag_kwargs.update(kwargs)
        return {"models": [{"name": f"m{i}", "details": {}} for i in range(4)]}

    async def fake_post_json(url: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return {}

    monkeypatch.setattr(anyenv, "get_json", fake_get_json)
    monkeypatch.setattr(anyenv, "post_json", fake_post_json)
    models = await OllamaProvider().get_models()
    assert [m.id for m in models] == ["m0", "m1", "m2", "m3"]
    assert peak == 4  # noqa: PLR2004
    assert not tag_kwargs.get("cache")  # a freshly pulled model must show up


def test_ollama_model_list_cached_briefly():
    assert OllamaProvider().models_cache_ttl < cache.MODELS_CACHE_TTL


async def test_models_dev_partials_share_one_download(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    payload = {
        "zai": {"models": {"glm": {"name": "GLM"}}},
        "moonshotai": {"models": {"kimi": {"name": "Kimi"}}},
    }

    async def fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        await asyncio.sleep(0.05)
        return payload

    monkeypatch.setattr(anyenv, "get_json", fake_get_json)
    zai, kimi = await asyncio.gather(
        ModelsDevProvider("zai").get_models(), ModelsDevProvider("moonshotai").get_models()
    )
    assert [m.id for m in zai] == ["glm"]
    assert [m.id for m in kimi] == ["kimi"]
    assert len(calls) == 1


def test_models_dev_zai_and_moonshot_need_keys(monkeypatch: pytest.MonkeyPatch):
    for var in ("ZAI_API_KEY", "ZHIPU_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not ModelsDevProvider("zai").is_available()
    assert not ModelsDevProvider("moonshotai").is_available()
    monkeypatch.setenv("ZAI_API_KEY", "k")
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    assert ModelsDevProvider("zai").is_available()
    assert ModelsDevProvider("moonshotai").is_available()


def test_cache_key_includes_params_and_base_url():
    a = OllamaProvider()
    b = OllamaProvider()
    b.params = {"key": "other"}
    c = OllamaProvider(base_url="http://other:11434")
    keys = {cache._key("x", p) for p in (a, b, c)}
    assert len(keys) == 3  # noqa: PLR2004


async def test_availability_probes_do_not_block_the_loop(monkeypatch: pytest.MonkeyPatch):
    from tokonomics import model_discovery

    def slow_available(self: Any) -> bool:
        time.sleep(0.3)
        return False

    monkeypatch.setattr(OllamaProvider, "is_available", slow_available)
    monkeypatch.setattr(model_discovery, "_PROVIDER_MAP", {"ollama": OllamaProvider})
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        assert await get_all_models() == []
    finally:
        task.cancel()
    assert ticks >= 5  # noqa: PLR2004


def test_gh_cli_token_resolved_once(monkeypatch: pytest.MonkeyPatch):
    import subprocess

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="gho_x\n", stderr="")

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    gh_mod.get_token_from_gh_cli.cache_clear()
    try:
        assert gh_mod.GitHubProvider().token == "gho_x"
        assert gh_mod.GitHubProvider().token == "gho_x"
    finally:
        gh_mod.get_token_from_gh_cli.cache_clear()
    assert len(calls) == 1


def test_transformers_tokenizer_loaded_once(monkeypatch: pytest.MonkeyPatch):
    loads: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str) -> FakeTokenizer:
            loads.append(name)
            return FakeTokenizer()

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(token_count, "has_library", lambda name: name == "transformers")
    token_count._transformers_tokenizer.cache_clear()
    try:
        assert token_count.count_tokens("a b c") == 3  # noqa: PLR2004
        assert token_count.count_tokens(["a", "b c"]) == 3  # noqa: PLR2004
    finally:
        token_count._transformers_tokenizer.cache_clear()
    assert len(loads) == 1


def test_models_dev_filter_without_key_mapping_is_unavailable():
    """A filter we can't check a key for isn't assumed usable (the catalog is)."""
    assert not ModelsDevProvider("alibaba").is_available()
    assert ModelsDevProvider().is_available()
