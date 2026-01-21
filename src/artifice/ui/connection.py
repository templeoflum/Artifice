"""
Connection lines between node ports.

Provides bezier curve connections with dynamic updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
from PySide6.QtWidgets import QGraphicsPathItem, QStyleOptionGraphicsItem, QWidget

if TYPE_CHECKING:
    from artifice.ui.node_widget import PortWidget


class ConnectionItem(QGraphicsPathItem):
    """
    Visual connection between two ports.

    Draws a bezier curve from source to target port.
    """

    # Connection style
    LINE_WIDTH = 2.5
    SELECTED_WIDTH = 3.5
    DEFAULT_COLOR = QColor(200, 200, 200, 200)
    SELECTED_COLOR = QColor(255, 200, 100)

    def __init__(
        self,
        source_port: PortWidget,
        target_port: PortWidget,
        parent=None,
    ):
        super().__init__(parent)

        self._source_port = source_port
        self._target_port = target_port

        # Configure item
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)  # Draw behind nodes

        # Get color from port type
        self._color = source_port.TYPE_COLORS.get(
            source_port.port_type,
            self.DEFAULT_COLOR
        )

        # Mark ports as connected
        source_port.is_connected = True
        target_port.is_connected = True

        # Initial path
        self.update_path()

    @property
    def source_port(self) -> PortWidget:
        """Get the source port."""
        return self._source_port

    @property
    def target_port(self) -> PortWidget:
        """Get the target port."""
        return self._target_port

    def update_path(self) -> None:
        """Update the bezier path based on port positions."""
        start = self._source_port.center_scene_pos()
        end = self._target_port.center_scene_pos()

        path = self._create_bezier_path(start, end)
        self.setPath(path)

    def _create_bezier_path(self, start: QPointF, end: QPointF) -> QPainterPath:
        """Create a bezier curve path."""
        path = QPainterPath()
        path.moveTo(start)

        # Calculate control points
        dx = abs(end.x() - start.x())
        dy = abs(end.y() - start.y())

        # Control point offset - increases with distance
        offset = max(50, min(dx * 0.5, 200))

        # If end is to the left of start, curve more dramatically
        if end.x() < start.x():
            offset = max(100, dx * 0.7)

        ctrl1 = QPointF(start.x() + offset, start.y())
        ctrl2 = QPointF(end.x() - offset, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        return path

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the connection."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.isSelected():
            color = self.SELECTED_COLOR
            width = self.SELECTED_WIDTH
        else:
            color = self._color
            width = self.LINE_WIDTH

        # Draw glow/shadow
        if self.isSelected():
            glow_pen = QPen(color, width + 4)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(glow_pen)
            painter.setOpacity(0.3)
            painter.drawPath(self.path())
            painter.setOpacity(1.0)

        # Draw main line
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(self.path())

    def boundingRect(self) -> QRectF:
        """Get bounding rectangle with padding for selection."""
        rect = super().boundingRect()
        padding = self.SELECTED_WIDTH + 5
        return rect.adjusted(-padding, -padding, padding, padding)

    def shape(self) -> QPainterPath:
        """Get the shape for hit testing."""
        # Create a wider path for easier selection
        stroker = QPainterPath()
        path = self.path()

        # Create a polygon along the path
        from PySide6.QtGui import QPainterPathStroker

        ps = QPainterPathStroker()
        ps.setWidth(15)  # Click tolerance
        ps.setCapStyle(Qt.PenCapStyle.RoundCap)
        return ps.createStroke(path)


class TempConnectionItem(QGraphicsPathItem):
    """
    Temporary connection being drawn.

    Used during connection creation before completing.
    """

    LINE_WIDTH = 2
    COLOR = QColor(150, 150, 200, 180)
    DASH_PATTERN = [5, 3]

    def __init__(self, source_port: PortWidget, parent=None):
        super().__init__(parent)

        self._source_port = source_port
        self._end_pos = source_port.center_scene_pos()

        # Get color from port type
        self._color = source_port.TYPE_COLORS.get(
            source_port.port_type,
            self.COLOR
        )

        self.setZValue(-1)
        self.update_path()

    def set_end_pos(self, pos: QPointF) -> None:
        """Update the end position."""
        self._end_pos = pos
        self.update_path()

    def update_path(self) -> None:
        """Update the bezier path."""
        start = self._source_port.center_scene_pos()
        end = self._end_pos

        path = self._create_bezier_path(start, end)
        self.setPath(path)

    def _create_bezier_path(self, start: QPointF, end: QPointF) -> QPainterPath:
        """Create a bezier curve path."""
        path = QPainterPath()
        path.moveTo(start)

        # Calculate control points
        dx = abs(end.x() - start.x())
        offset = max(50, min(dx * 0.5, 200))

        if end.x() < start.x():
            offset = max(100, dx * 0.7)

        # Direction depends on whether source is input or output
        if self._source_port.is_input:
            ctrl1 = QPointF(start.x() - offset, start.y())
            ctrl2 = QPointF(end.x() + offset, end.y())
        else:
            ctrl1 = QPointF(start.x() + offset, start.y())
            ctrl2 = QPointF(end.x() - offset, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        return path

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the temporary connection."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self._color, self.LINE_WIDTH)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern(self.DASH_PATTERN)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(self.path())
