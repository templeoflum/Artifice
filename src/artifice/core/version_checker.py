"""
Version checking and update notification system.

Queries GitHub Releases API to check for newer versions and notifies
the user when updates are available.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSettings, Signal

from artifice import __version__

if TYPE_CHECKING:
    pass


@dataclass
class VersionInfo:
    """Information about a release version."""

    version: str
    release_url: str
    changelog: str
    published_at: str
    is_prerelease: bool


class VersionChecker(QObject):
    """
    Background version checker using GitHub Releases API.

    Signals:
        update_available: Emitted when a newer version is found.
        check_complete: Emitted when check finishes (success: bool, message: str).
        error_occurred: Emitted on error (message: str).
    """

    # Signals
    update_available = Signal(VersionInfo)
    check_complete = Signal(bool, str)
    error_occurred = Signal(str)

    # GitHub API configuration
    GITHUB_OWNER = "templeoflum"
    GITHUB_REPO = "Artifice"
    GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    RELEASES_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"

    # Settings
    CHECK_INTERVAL_HOURS = 24
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = QSettings("ArtificeEngine", "Artifice")
        self._checking = False

    def get_current_version(self) -> str:
        """Return the current application version."""
        return __version__

    def check_for_updates(self, force: bool = False) -> None:
        """
        Check for updates in a background thread.

        Args:
            force: If True, check even if recently checked.
        """
        if self._checking:
            return

        if not force and not self._should_check():
            self.check_complete.emit(True, "Recently checked")
            return

        self._checking = True
        thread = threading.Thread(target=self._do_check, args=(force,), daemon=True)
        thread.start()

    def _should_check(self) -> bool:
        """Determine if enough time has passed since last check."""
        last_check_str = self._settings.value("updates/last_check_time", "")
        if not last_check_str:
            return True

        try:
            last_check = datetime.fromisoformat(last_check_str)
            elapsed = datetime.now() - last_check
            return elapsed > timedelta(hours=self.CHECK_INTERVAL_HOURS)
        except (ValueError, TypeError):
            return True

    def _do_check(self, force: bool) -> None:
        """Perform the actual version check (runs in background thread)."""
        try:
            latest = self._fetch_latest_release()
            if latest is None:
                self._checking = False
                self._safe_emit(self.check_complete, False, "Could not fetch release info")
                return

            # Update last check time
            self._settings.setValue(
                "updates/last_check_time", datetime.now().isoformat()
            )

            current = self.get_current_version()
            if self._is_newer_version(current, latest.version):
                # Check if user dismissed this version
                dismissed = self._settings.value("updates/dismissed_version", "")
                if not force and dismissed == latest.version:
                    self._safe_emit(self.check_complete, True, "Update dismissed by user")
                else:
                    self._safe_emit(self.update_available, latest)
                    self._safe_emit(self.check_complete, True, f"Update available: {latest.version}")
            else:
                self._safe_emit(self.check_complete, True, "Up to date")

        except urllib.error.HTTPError as e:
            # HTTP errors (404, 403, etc.) - likely no releases yet
            if e.code == 404:
                self._safe_emit(self.check_complete, True, "No releases found")
            else:
                self._safe_emit(self.error_occurred, f"HTTP error {e.code}: {e.reason}")
                self._safe_emit(self.check_complete, False, f"HTTP error {e.code}")
        except urllib.error.URLError as e:
            self._safe_emit(self.error_occurred, f"Network error: {e.reason}")
            self._safe_emit(self.check_complete, False, "Network error")
        except Exception as e:
            self._safe_emit(self.error_occurred, f"Error checking for updates: {e}")
            self._safe_emit(self.check_complete, False, str(e))
        finally:
            self._checking = False

    def _safe_emit(self, signal, *args) -> None:
        """Safely emit a signal, catching errors if object is deleted."""
        try:
            signal.emit(*args)
        except RuntimeError:
            # Signal source was deleted, ignore
            pass

    def _fetch_latest_release(self) -> VersionInfo | None:
        """Fetch the latest release from GitHub API."""
        request = urllib.request.Request(
            self.GITHUB_API_URL,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": f"Artifice/{__version__}",
            },
        )

        with urllib.request.urlopen(request, timeout=self.REQUEST_TIMEOUT) as response:
            if response.status != 200:
                return None

            data = json.loads(response.read().decode("utf-8"))

        # Parse version from tag_name (strip 'v' prefix if present)
        tag = data.get("tag_name", "")
        version = tag.lstrip("v")

        return VersionInfo(
            version=version,
            release_url=data.get("html_url", self.RELEASES_URL),
            changelog=data.get("body", ""),
            published_at=data.get("published_at", ""),
            is_prerelease=data.get("prerelease", False),
        )

    def _is_newer_version(self, current: str, latest: str) -> bool:
        """
        Compare version strings semantically.

        Returns True if latest > current.
        """
        try:
            current_parts = self._parse_version(current)
            latest_parts = self._parse_version(latest)
            return latest_parts > current_parts
        except (ValueError, TypeError):
            # If parsing fails, do string comparison as fallback
            return latest > current

    @staticmethod
    def _parse_version(version_str: str) -> tuple[int, ...]:
        """
        Parse a version string into a tuple of integers.

        Handles formats like "0.1.0", "v0.1.0", "1.2.3-beta".
        """
        # Strip 'v' prefix and any suffix after hyphen
        clean = version_str.lstrip("v").split("-")[0]
        # Extract numeric parts
        parts = re.findall(r"\d+", clean)
        return tuple(int(p) for p in parts)

    def dismiss_version(self, version: str) -> None:
        """Mark a version as dismissed (user chose to skip it)."""
        self._settings.setValue("updates/dismissed_version", version)

    def clear_dismissed(self) -> None:
        """Clear the dismissed version."""
        self._settings.remove("updates/dismissed_version")

    def get_releases_url(self) -> str:
        """Get the URL to the GitHub releases page."""
        return self.RELEASES_URL

    def is_checking(self) -> bool:
        """Return True if currently checking for updates."""
        return self._checking
