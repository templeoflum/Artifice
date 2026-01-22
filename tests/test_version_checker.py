"""
Tests for the version checking system.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from artifice.core.version_checker import VersionChecker, VersionInfo


class TestVersionParsing:
    """Tests for version string parsing."""

    def test_parse_simple_version(self):
        """Test parsing simple version strings."""
        assert VersionChecker._parse_version("0.1.0") == (0, 1, 0)
        assert VersionChecker._parse_version("1.2.3") == (1, 2, 3)
        assert VersionChecker._parse_version("10.20.30") == (10, 20, 30)

    def test_parse_version_with_v_prefix(self):
        """Test parsing versions with 'v' prefix."""
        assert VersionChecker._parse_version("v0.1.0") == (0, 1, 0)
        assert VersionChecker._parse_version("v1.2.3") == (1, 2, 3)

    def test_parse_version_with_suffix(self):
        """Test parsing versions with suffix."""
        assert VersionChecker._parse_version("0.1.0-beta") == (0, 1, 0)
        assert VersionChecker._parse_version("1.2.3-rc1") == (1, 2, 3)
        assert VersionChecker._parse_version("v2.0.0-alpha") == (2, 0, 0)

    def test_parse_two_part_version(self):
        """Test parsing two-part version strings."""
        assert VersionChecker._parse_version("1.0") == (1, 0)
        assert VersionChecker._parse_version("2.5") == (2, 5)


class TestVersionComparison:
    """Tests for version comparison logic."""

    @pytest.fixture
    def checker(self):
        """Create a VersionChecker instance."""
        return VersionChecker()

    def test_newer_patch_version(self, checker):
        """Test detecting newer patch version."""
        assert checker._is_newer_version("0.1.0", "0.1.1") is True
        assert checker._is_newer_version("1.2.3", "1.2.4") is True

    def test_newer_minor_version(self, checker):
        """Test detecting newer minor version."""
        assert checker._is_newer_version("0.1.0", "0.2.0") is True
        assert checker._is_newer_version("1.2.3", "1.3.0") is True

    def test_newer_major_version(self, checker):
        """Test detecting newer major version."""
        assert checker._is_newer_version("0.1.0", "1.0.0") is True
        assert checker._is_newer_version("1.2.3", "2.0.0") is True

    def test_same_version(self, checker):
        """Test same version is not newer."""
        assert checker._is_newer_version("0.1.0", "0.1.0") is False
        assert checker._is_newer_version("1.2.3", "1.2.3") is False

    def test_older_version(self, checker):
        """Test older version is not newer."""
        assert checker._is_newer_version("0.2.0", "0.1.0") is False
        assert checker._is_newer_version("1.0.0", "0.99.99") is False

    def test_version_with_prefix(self, checker):
        """Test comparison works with v prefix."""
        assert checker._is_newer_version("v0.1.0", "v0.2.0") is True
        assert checker._is_newer_version("0.1.0", "v0.2.0") is True
        assert checker._is_newer_version("v0.1.0", "0.2.0") is True


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_version_info_creation(self):
        """Test creating VersionInfo."""
        info = VersionInfo(
            version="0.2.0",
            release_url="https://github.com/test/releases/tag/v0.2.0",
            changelog="- New feature\n- Bug fix",
            published_at="2025-01-20T12:00:00Z",
            is_prerelease=False,
        )

        assert info.version == "0.2.0"
        assert "v0.2.0" in info.release_url
        assert "New feature" in info.changelog
        assert info.is_prerelease is False


class TestVersionChecker:
    """Tests for VersionChecker functionality."""

    @pytest.fixture
    def checker(self):
        """Create a VersionChecker instance."""
        return VersionChecker()

    def test_get_current_version(self, checker):
        """Test getting current version."""
        from artifice import __version__
        assert checker.get_current_version() == __version__

    def test_get_releases_url(self, checker):
        """Test getting releases URL."""
        url = checker.get_releases_url()
        assert "github.com" in url
        assert "releases" in url

    def test_dismiss_version(self, checker):
        """Test dismissing a version."""
        checker.dismiss_version("0.2.0")
        # Check it was saved to settings
        from PySide6.QtCore import QSettings
        settings = QSettings("ArtificeEngine", "Artifice")
        assert settings.value("updates/dismissed_version") == "0.2.0"
        # Clean up
        checker.clear_dismissed()

    def test_is_checking_initial_state(self, checker):
        """Test is_checking returns False initially."""
        assert checker.is_checking() is False


class TestAPIResponseParsing:
    """Tests for GitHub API response parsing."""

    @pytest.fixture
    def checker(self):
        """Create a VersionChecker instance."""
        return VersionChecker()

    @pytest.fixture
    def mock_api_response(self):
        """Sample GitHub API response."""
        return {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/templeoflum/Artifice-Engine/releases/tag/v0.2.0",
            "body": "## Changes\n- Added feature X\n- Fixed bug Y",
            "published_at": "2025-01-20T12:00:00Z",
            "prerelease": False,
            "draft": False,
        }

    def test_parse_release_response(self, checker, mock_api_response):
        """Test parsing API response into VersionInfo."""
        # We test the internal method by mocking urlopen
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = json.dumps(mock_api_response).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            info = checker._fetch_latest_release()

            assert info is not None
            assert info.version == "0.2.0"
            assert "v0.2.0" in info.release_url
            assert "Added feature X" in info.changelog
            assert info.is_prerelease is False

    def test_parse_prerelease_response(self, checker, mock_api_response):
        """Test parsing prerelease API response."""
        mock_api_response["prerelease"] = True
        mock_api_response["tag_name"] = "v0.3.0-beta"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = json.dumps(mock_api_response).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            info = checker._fetch_latest_release()

            assert info is not None
            assert info.version == "0.3.0-beta"
            assert info.is_prerelease is True


class TestCheckForUpdatesSync:
    """Synchronous tests for update checking (no signal waiting)."""

    @pytest.fixture
    def checker(self):
        """Create a VersionChecker instance."""
        return VersionChecker()

    def test_fetch_returns_version_info(self, checker):
        """Test that _fetch_latest_release returns VersionInfo."""
        mock_response = {
            "tag_name": "v0.2.0",
            "html_url": "https://example.com/release",
            "body": "New release",
            "published_at": "2025-01-20T12:00:00Z",
            "prerelease": False,
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            info = checker._fetch_latest_release()

            assert info is not None
            assert info.version == "0.2.0"

    def test_fetch_handles_network_error(self, checker):
        """Test that network errors are handled gracefully."""
        import urllib.error

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Network error")

            # Should not raise
            try:
                checker._fetch_latest_release()
            except urllib.error.URLError:
                pass  # Expected behavior
