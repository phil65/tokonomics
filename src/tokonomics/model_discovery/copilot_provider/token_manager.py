"""GitHub Copilot provider for model discovery."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
import os
import threading
from typing import Any
import weakref


logger = logging.getLogger(__name__)

# Constants for token management
BASE_URL = "https://api.github.com/copilot_internal/v2/token"
EDITOR_VERSION = "vscode/1.107.0"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"
USER_AGENT = "GitHubCopilotChat/0.35.0"
COPILOT_INTEGRATION_ID = "vscode-chat"
TOKEN_EXPIRY_BUFFER_SECONDS = 120  # Refresh token 2 minutes before expiry
DELTA = timedelta(seconds=TOKEN_EXPIRY_BUFFER_SECONDS)


def get_token_headers(github_oauth_token: str) -> dict[str, str]:
    """Get headers for token request."""
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {github_oauth_token}",
        "User-Agent": USER_AGENT,
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
    }


class CopilotTokenManager:
    """Manager for GitHub Copilot API tokens."""

    def __init__(self) -> None:
        self._github_oauth_token = os.environ.get("GITHUB_COPILOT_API_KEY")
        self._copilot_token = None
        self._token_expires_at = datetime.now()
        # Guards the token state and the per-loop locks; never held across an await.
        self._token_lock = threading.Lock()
        self._async_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )
        self._api_endpoint = "https://api.githubcopilot.com"

    def _needs_refresh(self) -> bool:
        return self._copilot_token is None or datetime.now() > self._token_expires_at - DELTA

    def _refresh_lock(self) -> asyncio.Lock:
        """Return the asyncio lock for the running loop (one per loop, created lazily)."""
        loop = asyncio.get_running_loop()
        with self._token_lock:
            lock = self._async_locks.get(loop)
            if lock is None:
                lock = self._async_locks[loop] = asyncio.Lock()
            return lock

    async def get_token(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        import anyenv

        if not self._needs_refresh():
            assert self._copilot_token
            return self._copilot_token
        # An asyncio lock, not the thread lock: holding a threading.Lock across the
        # await would block the whole event loop when a second coroutine waits on it.
        async with self._refresh_lock():
            if self._needs_refresh():
                if not self._github_oauth_token:
                    msg = "GitHub OAuth token not found in GITHUB_COPILOT_API_KEY env var"
                    raise RuntimeError(msg)

                try:
                    logger.debug("Fetching fresh GitHub Copilot token")
                    data = await anyenv.get_json(
                        BASE_URL,
                        headers=get_token_headers(self._github_oauth_token),
                        return_type=dict,
                    )
                except Exception as e:
                    logger.exception("Failed to refresh GitHub Copilot token")
                    if not self._copilot_token:
                        msg = "Failed to obtain GitHub Copilot token"
                        raise RuntimeError(msg) from e
                else:
                    with self._token_lock:
                        self.handle_token_response(data)
            assert self._copilot_token, "Copilot token is missing"
            return self._copilot_token

    def get_token_sync(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        import anyenv

        with self._token_lock:
            # If token is missing or expires in less than buffer time, refresh it
            if self._needs_refresh():
                if not self._github_oauth_token:
                    msg = "GitHub OAuth token not found in GITHUB_COPILOT_API_KEY env var"
                    raise RuntimeError(msg)

                try:
                    logger.debug("Fetching fresh GitHub Copilot token")
                    data = anyenv.get_json_sync(
                        BASE_URL,
                        headers=get_token_headers(self._github_oauth_token),
                        return_type=dict,
                    )
                except Exception as e:
                    logger.exception("Failed to refresh GitHub Copilot token")
                    if not self._copilot_token:
                        msg = "Failed to obtain GitHub Copilot token"
                        raise RuntimeError(msg) from e
                else:
                    self.handle_token_response(data)
            assert self._copilot_token, "Copilot token is missing"
            return self._copilot_token

    def handle_token_response(self, data: dict[str, Any]) -> None:
        self._copilot_token = data.get("token")
        if not self._copilot_token:
            msg = "No token found in response from Copilot API"
            raise RuntimeError(msg)
        if expires_at := data.get("expires_at"):
            self._token_expires_at = datetime.fromtimestamp(expires_at)
        else:
            # Default expiry: 25 minutes if not specified
            self._token_expires_at = datetime.now() + timedelta(minutes=25)
        if "api" in (endpoints := data.get("endpoints", {})):
            self._api_endpoint = endpoints["api"]
        expires_at = self._token_expires_at.isoformat()
        logger.debug("Copilot token refreshed, valid until: %s", expires_at)

    def is_available(self) -> bool:
        """Check whether the token manager is available for use."""
        return bool(self._github_oauth_token)

    async def generate_headers(self) -> dict[str, str]:
        """Generate headers for GitHub Copilot API requests."""
        return {
            "Authorization": f"Bearer {await self.get_token()}",
            "User-Agent": USER_AGENT,
            "Editor-Version": EDITOR_VERSION,
            "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
            "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
            "Openai-Intent": "conversation-edits",
            "X-Initiator": "user",
        }


token_manager = CopilotTokenManager()

if __name__ == "__main__":
    token = token_manager.get_token_sync()
    print(token)
