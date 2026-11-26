"""Model discovery against a local server: anyenv 3 responses, one connection per provider."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import pytest

from tokonomics.model_discovery.gemini_provider.provider import GeminiProvider


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.connections = 0
        self.paths: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def setup(self) -> None:
        super().setup()
        self.server.connections += 1

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.server.paths.append(self.path)
        page = int(parse_qs(urlsplit(self.path).query).get("pageToken", ["0"])[0])
        data: dict[str, object] = {
            "models": [{"name": f"models/m{page}", "supportedGenerationMethods": []}]
        }
        if page < 2:  # noqa: PLR2004
            data["nextPageToken"] = str(page + 1)
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    import appdirs

    # anyenv's HTTP cache goes to a temporary directory.
    monkeypatch.setattr(appdirs, "user_cache_dir", lambda *args, **kwargs: str(tmp_path))
    srv = _Server()
    threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


async def test_gemini_pages_share_one_connection(server: _Server):
    provider = GeminiProvider(api_key="k")
    provider.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    models = await provider.get_models()
    assert [m.id for m in models] == ["m0", "m1", "m2"]
    assert len(server.paths) == 3  # noqa: PLR2004
    assert server.connections == 1
