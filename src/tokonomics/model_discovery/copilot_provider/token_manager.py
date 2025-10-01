"""GitHub Copilot provider for model discovery."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import threading


logger = logging.getLogger(__name__)

# Constants for token management
BASE_URL = "https://api.github.com/copilot_internal/v2/token"
EDITOR_VERSION = "Neovim/0.6.1"
EDITOR_PLUGIN_VERSION = "copilot.vim/1.16.0"
USER_AGENT = "GithubCopilot/1.155.0"
TOKEN_EXPIRY_BUFFER_SECONDS = 120  # Refresh token 2 minutes before expiry
DELTA = timedelta(seconds=TOKEN_EXPIRY_BUFFER_SECONDS)


def get_token_headers(github_oauth_token: str) -> dict[str, str]:
    """Get headers for token request."""
    return {
        "Authorization": f"Bearer {github_oauth_token}",
        "User-Agent": USER_AGENT,
        "X-Editor-Version": EDITOR_VERSION,
        "X-Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
    }


class CopilotTokenManager:
    """Manager for GitHub Copilot API tokens."""

    def __init__(self):
        self._github_oauth_token = os.environ.get("GITHUB_COPILOT_API_KEY")
        self._copilot_token = None
        self._token_expires_at = datetime.now()
        self._token_lock = threading.Lock()
        self._api_endpoint = "https://api.githubcopilot.com"

    async def get_token(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        import anyenv

        with self._token_lock:
            # If token is missing or expires in less than buffer time, refresh it
            now = datetime.now()
            if self._copilot_token is None or now > self._token_expires_at - DELTA:
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
                    self.handle_token_response(data)
            assert self._copilot_token, "Copilot token is missing"
            return self._copilot_token

    def get_token_sync(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        import anyenv

        with self._token_lock:
            # If token is missing or expires in less than buffer time, refresh it
            now = datetime.now()
            if self._copilot_token is None or now > self._token_expires_at - DELTA:
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

    def handle_token_response(self, data):
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
            "editor-version": "Neovim/0.9.0",
            "Copilot-Integration-Id": "vscode-chat",
        }


token_manager = CopilotTokenManager()

if __name__ == "__main__":
    token = token_manager.get_token_sync()
    print(token)
