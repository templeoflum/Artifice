"""
Enhanced About dialog with version information and update checking.
"""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
)

from artifice import __version__

if TYPE_CHECKING:
    from artifice.core.version_checker import VersionInfo


class AboutDialog(QDialog):
    """
    Enhanced About dialog showing version info and update status.

    Displays current version, update availability, and provides
    buttons to check for updates and view releases.
    """

    def __init__(
        self,
        parent=None,
        update_info: VersionInfo | None = None,
        releases_url: str = "https://github.com/templeoflum/Artifice-Engine/releases",
    ) -> None:
        super().__init__(parent)
        self._update_info = update_info
        self._releases_url = releases_url
        self._check_callback = None
        self._setup_ui()

    def set_check_callback(self, callback) -> None:
        """Set callback for check updates button."""
        self._check_callback = callback

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        self.setWindowTitle("About Artifice")
        self.setFixedSize(400, 320)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title_label = QLabel("ARTIFICE")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Motto
        motto_label = QLabel("Converse with Chaos, Sculpt Emergence.")
        motto_label.setStyleSheet("color: gray; font-style: italic;")
        motto_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(motto_label)

        # Version
        version_label = QLabel(f"Version {__version__}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        layout.addSpacing(8)

        # Update notification (if available)
        if self._update_info:
            self._add_update_banner(layout)

        layout.addStretch()

        # Buttons row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self._check_btn = QPushButton("Check for Updates")
        self._check_btn.clicked.connect(self._on_check_updates)
        button_layout.addWidget(self._check_btn)

        releases_btn = QPushButton("View Releases")
        releases_btn.clicked.connect(self._on_view_releases)
        button_layout.addWidget(releases_btn)

        layout.addLayout(button_layout)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # Description
        desc_label = QLabel(
            "A node-based glitch art tool for building\n"
            "image processing pipelines."
        )
        desc_label.setStyleSheet("color: gray; font-size: 10px;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

    def _add_update_banner(self, layout: QVBoxLayout) -> None:
        """Add update available banner."""
        banner = QFrame()
        banner.setFrameStyle(QFrame.Shape.StyledPanel)
        banner.setStyleSheet(
            """
            QFrame {
                background-color: #2d5a27;
                border: 1px solid #3d7a37;
                border-radius: 4px;
                padding: 8px;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #4a8c44;
                border: none;
                border-radius: 3px;
                padding: 4px 12px;
                color: white;
            }
            QPushButton:hover {
                background-color: #5a9c54;
            }
            """
        )

        banner_layout = QVBoxLayout(banner)
        banner_layout.setSpacing(8)

        info_label = QLabel(
            f"A new version ({self._update_info.version}) is available!"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner_layout.addWidget(info_label)

        download_btn = QPushButton("Download Update")
        download_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        download_btn.clicked.connect(self._on_download_update)

        btn_container = QHBoxLayout()
        btn_container.addStretch()
        btn_container.addWidget(download_btn)
        btn_container.addStretch()
        banner_layout.addLayout(btn_container)

        layout.addWidget(banner)

    def _on_check_updates(self) -> None:
        """Handle check for updates button."""
        if self._check_callback:
            self._check_callback()
        self._check_btn.setText("Checking...")
        self._check_btn.setEnabled(False)

    def _on_view_releases(self) -> None:
        """Open GitHub releases page in browser."""
        webbrowser.open(self._releases_url)

    def _on_download_update(self) -> None:
        """Open the release page for the available update."""
        if self._update_info:
            webbrowser.open(self._update_info.release_url)

    def set_check_complete(self, message: str) -> None:
        """Update UI after check completes."""
        self._check_btn.setText("Check for Updates")
        self._check_btn.setEnabled(True)
