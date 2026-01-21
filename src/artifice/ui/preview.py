"""
Preview panel for displaying image output.

Provides zoomable, pannable image preview with various display options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QWheelEvent,
    QMouseEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QSlider,
    QFrame,
    QSizePolicy,
)

if TYPE_CHECKING:
    from artifice.core.data_types import ImageBuffer


class ImageWidget(QWidget):
    """
    Widget for displaying an image with zoom and pan.

    Renders a numpy array or ImageBuffer as a QImage.
    """

    ZOOM_MIN = 0.1
    ZOOM_MAX = 10.0
    ZOOM_FACTOR = 1.2

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image: QImage | None = None
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)

        self._is_panning = False
        self._pan_start = QPointF()
        self._last_pan_pos = QPointF()

        # Display options
        self._show_checkerboard = True
        self._channel_mode = "RGB"  # RGB, R, G, B, A, L

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    @property
    def has_image(self) -> bool:
        """Check if an image is loaded."""
        return self._image is not None

    @property
    def image_size(self) -> tuple[int, int] | None:
        """Get image dimensions (width, height)."""
        if self._image:
            return (self._image.width(), self._image.height())
        return None

    def set_image(self, image: ImageBuffer | np.ndarray | None) -> None:
        """Set the image to display."""
        if image is None:
            self._image = None
            self._pixmap = None
            self.update()
            return

        # Convert to numpy array if ImageBuffer
        if hasattr(image, "data"):
            data = image.data
        else:
            data = image

        # Ensure float32 and [0, 1] range
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.max() > 1.0:
            data = data / 255.0

        # Convert CHW to HWC if needed
        if data.ndim == 3 and data.shape[0] <= 4:
            data = np.transpose(data, (1, 2, 0))

        # Handle different channel counts
        if data.ndim == 2:
            # Grayscale
            data = np.stack([data, data, data], axis=-1)
        elif data.shape[-1] == 1:
            # Single channel
            data = np.repeat(data, 3, axis=-1)
        elif data.shape[-1] == 4:
            # RGBA - keep as is
            pass
        elif data.shape[-1] == 3:
            # RGB - add alpha
            alpha = np.ones((*data.shape[:2], 1), dtype=np.float32)
            data = np.concatenate([data, alpha], axis=-1)

        # Convert to 8-bit
        data = np.clip(data * 255, 0, 255).astype(np.uint8)

        # Ensure contiguous array
        data = np.ascontiguousarray(data)

        # Create QImage
        h, w = data.shape[:2]
        if data.shape[-1] == 4:
            self._image = QImage(
                data.data, w, h, w * 4, QImage.Format.Format_RGBA8888
            ).copy()  # Copy to avoid dangling pointer
        else:
            self._image = QImage(
                data.data, w, h, w * 3, QImage.Format.Format_RGB888
            ).copy()

        self._pixmap = QPixmap.fromImage(self._image)
        self._fit_to_view()
        self.update()

    def clear(self) -> None:
        """Clear the image."""
        self._image = None
        self._pixmap = None
        self.update()

    def set_channel_mode(self, mode: str) -> None:
        """Set channel display mode."""
        self._channel_mode = mode
        # TODO: Apply channel filtering
        self.update()

    def set_checkerboard(self, show: bool) -> None:
        """Set whether to show checkerboard for transparency."""
        self._show_checkerboard = show
        self.update()

    def zoom_in(self) -> None:
        """Zoom in."""
        center = self.rect().center()
        self._zoom_at(QPointF(center.x(), center.y()), self.ZOOM_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out."""
        center = self.rect().center()
        self._zoom_at(QPointF(center.x(), center.y()), 1.0 / self.ZOOM_FACTOR)

    def zoom_reset(self) -> None:
        """Reset to 100% zoom."""
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self.update()

    def fit_to_view(self) -> None:
        """Fit image to view."""
        self._fit_to_view()
        self.update()

    def _fit_to_view(self) -> None:
        """Calculate zoom to fit image in view."""
        if not self._image:
            return

        view_w = self.width()
        view_h = self.height()
        img_w = self._image.width()
        img_h = self._image.height()

        if img_w == 0 or img_h == 0:
            return

        scale_x = view_w / img_w
        scale_y = view_h / img_h
        self._zoom = min(scale_x, scale_y) * 0.95  # Leave some margin

        # Center the image
        scaled_w = img_w * self._zoom
        scaled_h = img_h * self._zoom
        self._offset = QPointF(
            (view_w - scaled_w) / 2,
            (view_h - scaled_h) / 2
        )

    def _zoom_at(self, pos: QPointF, factor: float) -> None:
        """Zoom at a specific position."""
        new_zoom = self._zoom * factor
        if self.ZOOM_MIN <= new_zoom <= self.ZOOM_MAX:
            # Adjust offset to zoom towards position
            old_pos = (pos - self._offset) / self._zoom
            self._zoom = new_zoom
            new_pos = old_pos * self._zoom
            self._offset = pos - new_pos
            self.update()

    def paintEvent(self, event) -> None:
        """Paint the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._pixmap:
            # Draw checkerboard for transparency
            if self._show_checkerboard:
                self._draw_checkerboard(painter)

            # Calculate destination rect
            img_w = self._pixmap.width() * self._zoom
            img_h = self._pixmap.height() * self._zoom
            dest_rect = QRectF(
                self._offset.x(),
                self._offset.y(),
                img_w,
                img_h
            )

            # Draw image
            painter.drawPixmap(dest_rect, self._pixmap, QRectF(self._pixmap.rect()))

            # Draw border
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawRect(dest_rect)
        else:
            # Draw placeholder text
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No image"
            )

    def _draw_checkerboard(self, painter: QPainter) -> None:
        """Draw checkerboard pattern for transparency."""
        if not self._pixmap:
            return

        img_w = self._pixmap.width() * self._zoom
        img_h = self._pixmap.height() * self._zoom
        rect = QRectF(self._offset.x(), self._offset.y(), img_w, img_h)

        # Clip to image area
        painter.save()
        painter.setClipRect(rect)

        # Draw checkerboard
        size = 10
        colors = [QColor(60, 60, 60), QColor(80, 80, 80)]

        x_start = int(rect.left() / size) * size
        y_start = int(rect.top() / size) * size

        for y in range(int(y_start), int(rect.bottom()) + size, size):
            for x in range(int(x_start), int(rect.right()) + size, size):
                color_idx = ((x // size) + (y // size)) % 2
                painter.fillRect(x, y, size, size, colors[color_idx])

        painter.restore()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            factor = self.ZOOM_FACTOR
        else:
            factor = 1.0 / self.ZOOM_FACTOR

        self._zoom_at(event.position(), factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start = event.position()
            self._last_pan_pos = self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._offset = self._last_pan_pos + delta
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event) -> None:
        """Handle resize."""
        if self._image:
            self._fit_to_view()
        super().resizeEvent(event)


class PreviewPanel(QWidget):
    """
    Panel containing image preview with controls.

    Provides image display with zoom controls and channel selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Image widget
        self._image_widget = ImageWidget()
        layout.addWidget(self._image_widget, 1)

        # Controls
        controls = QFrame()
        controls.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 4, 4)

        # Channel selector
        self._channel_combo = QComboBox()
        self._channel_combo.addItems(["RGB", "Red", "Green", "Blue", "Alpha", "Luminance"])
        self._channel_combo.currentTextChanged.connect(self._on_channel_changed)
        controls_layout.addWidget(QLabel("Channel:"))
        controls_layout.addWidget(self._channel_combo)

        controls_layout.addStretch()

        # Checkerboard toggle
        self._checker_check = QCheckBox("Checkerboard")
        self._checker_check.setChecked(True)
        self._checker_check.toggled.connect(self._image_widget.set_checkerboard)
        controls_layout.addWidget(self._checker_check)

        # Zoom controls
        controls_layout.addWidget(QLabel("Zoom:"))

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 500)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(100)
        controls_layout.addWidget(self._zoom_slider)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(40)
        controls_layout.addWidget(self._zoom_label)

        layout.addWidget(controls)

    def set_image(self, image) -> None:
        """Set the image to preview."""
        self._image_widget.set_image(image)

    def clear(self) -> None:
        """Clear the preview."""
        self._image_widget.clear()

    def has_image(self) -> bool:
        """Check if an image is displayed."""
        return self._image_widget.has_image

    @property
    def image_size(self) -> tuple[int, int] | None:
        """Get the image size."""
        return self._image_widget.image_size

    def _on_channel_changed(self, text: str) -> None:
        """Handle channel selection change."""
        mode_map = {
            "RGB": "RGB",
            "Red": "R",
            "Green": "G",
            "Blue": "B",
            "Alpha": "A",
            "Luminance": "L",
        }
        self._image_widget.set_channel_mode(mode_map.get(text, "RGB"))
