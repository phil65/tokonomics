"""GitHub Copilot provider for model discovery."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import threading


logger = logging.getLogger(__name__)

# Constants for token management
EDITOR_VERSION = "Neovim/0.6.1"
EDITOR_PLUGIN_VERSION = "copilot.vim/1.16.0"
USER_AGENT = "GithubCopilot/1.155.0"
TOKEN_EXPIRY_BUFFER_SECONDS = 120  # Refresh token 2 minutes before expiry
DELTA = timedelta(seconds=TOKEN_EXPIRY_BUFFER_SECONDS)


class CopilotTokenManager:
    """Manager for GitHub Copilot API tokens."""

    def __init__(self):
        # Get the GitHub OAuth token from environment
        self._github_oauth_token = os.environ.get("GITHUB_COPILOT_API_KEY")
        # This will store the short-lived Copilot token
        self._copilot_token = None
        self._token_expires_at = datetime.now()
        self._token_lock = threading.Lock()
        self._api_endpoint = "https://api.githubcopilot.com"

    def get_token(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        with self._token_lock:
            # If token is missing or expires in less than buffer time, refresh it
            now = datetime.now()
            if self._copilot_token is None or now > self._token_expires_at - DELTA:
                self._refresh_token()
            assert self._copilot_token, "Copilot token is missing"
            return self._copilot_token

    def _refresh_token(self) -> None:
        """Refresh the Copilot token using the GitHub OAuth token."""
        import anyenv

        if not self._github_oauth_token:
            msg = "GitHub OAuth token not found in GITHUB_COPILOT_API_KEY env var"
            raise RuntimeError(msg)

        try:
            logger.debug("Fetching fresh GitHub Copilot token")
            data = anyenv.get_json_sync(
                "https://api.github.com/copilot_internal/v2/token",
                headers={
                    "authorization": f"token {self._github_oauth_token}",
                    "editor-version": EDITOR_VERSION,
                    "editor-plugin-version": EDITOR_PLUGIN_VERSION,
                    "user-agent": USER_AGENT,
                },
                timeout=30,
                return_type=dict,
            )
            # Extract the Copilot token
            self._copilot_token = data.get("token")
            if not self._copilot_token:
                msg = "No token found in response from Copilot API"
                raise RuntimeError(msg)  # noqa: TRY301

            # Extract expiration time
            expires_at = data.get("expires_at")
            if expires_at is not None:
                self._token_expires_at = datetime.fromtimestamp(expires_at)
            else:
                # Default expiry: 25 minutes if not specified
                self._token_expires_at = datetime.now() + timedelta(minutes=25)

            # Update API endpoint if provided
            endpoints = data.get("endpoints", {})
            if "api" in endpoints:
                self._api_endpoint = endpoints["api"]

            logger.debug(
                "Copilot token refreshed, valid until: %s",
                self._token_expires_at.isoformat(),
            )
        except Exception as e:
            logger.exception("Failed to refresh GitHub Copilot token")
            if not self._copilot_token:
                msg = "Failed to obtain GitHub Copilot token"
                raise RuntimeError(msg) from e

    def generate_headers(self) -> dict[str, str]:
        """Generate headers for GitHub Copilot API requests."""
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "editor-version": "Neovim/0.9.0",
            "Copilot-Integration-Id": "vscode-chat",
        }


token_manager = CopilotTokenManager()
