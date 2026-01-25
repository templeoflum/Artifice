"""Real-time GPU-accelerated preview widget.

Provides a QOpenGLWidget-based preview that displays GPU textures
directly without CPU roundtrips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QHBoxLayout

if TYPE_CHECKING:
    from artifice.core.stream_executor import StreamExecutor, ExecutorStats
    from artifice.gpu.backend import GPUBackend
    from artifice.gpu.texture import Texture


class RealtimePreviewWidget(QOpenGLWidget):
    """GPU-accelerated real-time preview widget.

    Displays frames from the StreamExecutor with minimal latency
    by rendering GPU textures directly without CPU copies.
    """

    # Signals
    fps_updated = Signal(float)
    frame_rendered = Signal()

    def __init__(self, parent: QWidget | None = None):
        # Set up OpenGL format before creating widget
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.TripleBuffer)
        fmt.setSwapInterval(1)  # VSync
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        # GPU resources
        self._backend: GPUBackend | None = None
        self._executor: StreamExecutor | None = None
        self._display_texture: Texture | None = None

        # Display state
        self._initialized = False
        self._needs_redraw = True

        # Fallback image for when no GPU texture
        self._fallback_image: np.ndarray | None = None

        # FPS counter
        self._frame_count = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)  # Update every second
        self._last_fps = 0.0

        # Refresh timer (backup in case signals don't fire)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.update)
        self._refresh_timer.start(16)  # ~60 FPS baseline

    def set_backend(self, backend: GPUBackend) -> None:
        """Set the GPU backend.

        Args:
            backend: Initialized GPU backend
        """
        self._backend = backend

    def set_executor(self, executor: StreamExecutor) -> None:
        """Connect to a stream executor.

        Args:
            executor: Stream executor to display frames from
        """
        if self._executor:
            self._executor.frame_ready.disconnect(self._on_frame_ready)

        self._executor = executor
        executor.frame_ready.connect(self._on_frame_ready)

    def set_texture(self, texture: Texture) -> None:
        """Set a texture to display directly.

        Args:
            texture: GPU texture to display
        """
        self._display_texture = texture
        self._needs_redraw = True
        self.update()

    def set_fallback_image(self, image: np.ndarray) -> None:
        """Set a fallback image for when no GPU texture is available.

        Args:
            image: NumPy array (H, W, C) in [0, 1]
        """
        self._fallback_image = image
        self._needs_redraw = True
        self.update()

    @Slot()
    def _on_frame_ready(self) -> None:
        """Handle new frame from executor."""
        if self._executor and self._executor.triple_buffer:
            self._display_texture = self._executor.triple_buffer.get_display_texture()
            self._needs_redraw = True
            self._frame_count += 1
            self.update()

    @Slot()
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self._last_fps = float(self._frame_count)
        self._frame_count = 0
        self.fps_updated.emit(self._last_fps)

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        # Note: The backend should be initialized before this widget
        # We just verify it's ready
        self._initialized = True

    def resizeGL(self, w: int, h: int) -> None:
        """Handle resize."""
        if self._backend and self._backend.is_initialized:
            # Update viewport will be handled in paintGL
            pass

    def paintGL(self) -> None:
        """Render the current frame."""
        if not self._initialized:
            return

        if not self._backend or not self._backend.is_initialized:
            # Fall back to basic clear
            from OpenGL.GL import glClearColor, glClear, GL_COLOR_BUFFER_BIT
            glClearColor(0.1, 0.1, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            return

        # Get display texture from executor if available
        if self._executor and self._executor.triple_buffer:
            self._display_texture = self._executor.triple_buffer.get_display_texture()

        if self._display_texture:
            # Render texture to screen
            viewport = (0, 0, self.width(), self.height())
            self._backend.blit_to_screen(
                self._display_texture,
                self._backend.get_blit_program(),
                viewport,
            )
        else:
            # Clear to dark gray
            from OpenGL.GL import glClearColor, glClear, GL_COLOR_BUFFER_BIT
            glClearColor(0.1, 0.1, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

        self._needs_redraw = False
        self.frame_rendered.emit()

    @property
    def fps(self) -> float:
        """Return current FPS."""
        return self._last_fps


class RealtimePreviewPanel(QWidget):
    """Complete preview panel with stats overlay.

    Wraps RealtimePreviewWidget with FPS display and controls.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Preview widget
        self._preview = RealtimePreviewWidget()
        layout.addWidget(self._preview, 1)

        # Stats bar
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(4, 2, 4, 2)

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("color: #888; font-size: 11px;")
        stats_layout.addWidget(self._fps_label)

        stats_layout.addStretch()

        self._resolution_label = QLabel("--x--")
        self._resolution_label.setStyleSheet("color: #888; font-size: 11px;")
        stats_layout.addWidget(self._resolution_label)

        layout.addLayout(stats_layout)

        # Connect signals
        self._preview.fps_updated.connect(self._on_fps_updated)

    @Slot(float)
    def _on_fps_updated(self, fps: float) -> None:
        """Update FPS display."""
        color = "#4f4" if fps >= 55 else "#ff4" if fps >= 30 else "#f44"
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._fps_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def set_backend(self, backend: GPUBackend) -> None:
        """Set the GPU backend."""
        self._preview.set_backend(backend)

    def set_executor(self, executor: StreamExecutor) -> None:
        """Connect to a stream executor."""
        self._preview.set_executor(executor)

        # Update resolution label
        if executor:
            w, h = executor._output_width, executor._output_height
            self._resolution_label.setText(f"{w}x{h}")

    def update_stats(self, stats: ExecutorStats) -> None:
        """Update stats display.

        Args:
            stats: Executor statistics
        """
        # FPS is updated via signal, but we can show more stats here
        pass

    @property
    def preview_widget(self) -> RealtimePreviewWidget:
        """Return the preview widget."""
        return self._preview


class FallbackPreviewWidget(QWidget):
    """Fallback preview for when OpenGL is not available.

    Uses standard Qt painting instead of OpenGL.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image: np.ndarray | None = None

        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a1a;")

    def set_image(self, image: np.ndarray) -> None:
        """Set image to display.

        Args:
            image: NumPy array (H, W, C) in [0, 1]
        """
        self._image = image
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the image."""
        from PySide6.QtGui import QPainter, QImage, QPixmap

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._image is not None:
            # Convert to QImage
            h, w = self._image.shape[:2]
            channels = self._image.shape[2] if self._image.ndim == 3 else 1

            # Convert to 8-bit RGB
            img_8bit = (self._image * 255).astype(np.uint8)

            if channels == 1:
                # Grayscale
                qimg = QImage(
                    img_8bit.tobytes(),
                    w, h,
                    w,
                    QImage.Format.Format_Grayscale8,
                )
            elif channels == 3:
                # RGB
                qimg = QImage(
                    img_8bit.tobytes(),
                    w, h,
                    w * 3,
                    QImage.Format.Format_RGB888,
                )
            else:
                # RGBA
                qimg = QImage(
                    img_8bit.tobytes(),
                    w, h,
                    w * 4,
                    QImage.Format.Format_RGBA8888,
                )

            # Scale to fit widget
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            # Center in widget
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            # No image - draw placeholder
            painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No preview available",
            )

        painter.end()
