"""
Node widget for visual representation of nodes.

Each node is displayed as a rectangular widget with ports,
a title, and collapsed/expanded states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QBrush,
    QFont,
    QFontMetrics,
    QPainterPath,
    QLinearGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsDropShadowEffect,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from artifice.core.node import Node


# Color palette for different node categories
CATEGORY_COLORS = {
    "Input/Output": QColor(120, 180, 80),
    "Color": QColor(180, 120, 200),
    "Segmentation": QColor(80, 160, 200),
    "Prediction": QColor(200, 140, 80),
    "Quantization": QColor(140, 140, 200),
    "Transform": QColor(200, 100, 100),
    "Corruption": QColor(200, 80, 120),
    "Pipeline": QColor(100, 180, 180),
    "Utility": QColor(150, 150, 150),
    "default": QColor(100, 100, 120),
}


class PortWidget(QGraphicsObject):
    """
    Visual representation of a node port.

    Ports are clickable circles that can be connected to other ports.
    """

    RADIUS = 6
    HOVER_RADIUS = 8

    # Port type colors
    TYPE_COLORS = {
        "IMAGE": QColor(200, 200, 80),
        "ARRAY": QColor(80, 180, 200),
        "FLOAT": QColor(150, 200, 80),
        "INT": QColor(80, 200, 150),
        "BOOL": QColor(200, 100, 100),
        "STRING": QColor(200, 150, 100),
        "ANY": QColor(180, 180, 180),
    }

    pressed = Signal(object)  # PortWidget

    def __init__(
        self,
        node: Node,
        port_name: str,
        port_type: str,
        is_input: bool,
        parent: NodeWidget | None = None,
    ):
        super().__init__(parent)

        self._node = node
        self._port_name = port_name
        self._port_type = port_type
        self._is_input = is_input
        self._is_connected = False
        self._is_hovered = False

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    @property
    def node(self) -> Node:
        """Get the node this port belongs to."""
        return self._node

    @property
    def port_name(self) -> str:
        """Get the port name."""
        return self._port_name

    @property
    def port_type(self) -> str:
        """Get the port type."""
        return self._port_type

    @property
    def is_input(self) -> bool:
        """Check if this is an input port."""
        return self._is_input

    @property
    def is_connected(self) -> bool:
        """Check if port is connected."""
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """Set connected state."""
        self._is_connected = value
        self.update()

    def boundingRect(self) -> QRectF:
        """Get bounding rectangle."""
        r = self.HOVER_RADIUS + 2
        return QRectF(-r, -r, r * 2, r * 2)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the port."""
        color = self.TYPE_COLORS.get(self._port_type, self.TYPE_COLORS["ANY"])

        radius = self.HOVER_RADIUS if self._is_hovered else self.RADIUS

        # Draw port circle
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._is_connected or self._is_hovered:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(QBrush(color.darker(150)))

        painter.setPen(QPen(color.lighter(120), 1.5))
        painter.drawEllipse(QPointF(0, 0), radius, radius)

    def hoverEnterEvent(self, event) -> None:
        """Handle hover enter."""
        self._is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Handle hover leave."""
        self._is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed.emit(self)
        super().mousePressEvent(event)

    def center_scene_pos(self) -> QPointF:
        """Get the center position in scene coordinates."""
        return self.scenePos()


class NodeWidget(QGraphicsObject):
    """
    Visual representation of a node in the graph.

    Displays the node title, ports, and handles selection/movement.
    """

    # Dimensions
    MIN_WIDTH = 150
    HEADER_HEIGHT = 24
    PORT_SPACING = 20
    PORT_MARGIN = 12
    PADDING = 8
    CORNER_RADIUS = 6

    # Signals
    position_changed = Signal(object, QPointF, QPointF)  # widget, old, new
    port_pressed = Signal(object)  # PortWidget
    selected_changed = Signal(object, bool)  # widget, selected

    def __init__(self, node: Node, parent=None):
        super().__init__(parent)

        self._node = node
        self._width = self.MIN_WIDTH
        self._height = 0
        self._input_ports: dict[str, PortWidget] = {}
        self._output_ports: dict[str, PortWidget] = {}
        self._drag_start_pos: QPointF | None = None

        # Configure item
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)

        # Calculate size and create ports
        self._calculate_size()
        self._create_ports()

    @property
    def node(self) -> Node:
        """Get the underlying node."""
        return self._node

    def get_input_port(self, name: str) -> PortWidget | None:
        """Get an input port widget by name."""
        return self._input_ports.get(name)

    def get_output_port(self, name: str) -> PortWidget | None:
        """Get an output port widget by name."""
        return self._output_ports.get(name)

    def _calculate_size(self) -> None:
        """Calculate widget dimensions based on node ports."""
        # Calculate width from title and port names
        font = QFont("Segoe UI", 10)
        metrics = QFontMetrics(font)

        title_width = metrics.horizontalAdvance(self._node.name) + self.PADDING * 4

        max_input_width = 0
        max_output_width = 0

        for port_name in self._node.inputs:
            max_input_width = max(
                max_input_width,
                metrics.horizontalAdvance(port_name) + self.PORT_MARGIN * 2
            )

        for port_name in self._node.outputs:
            max_output_width = max(
                max_output_width,
                metrics.horizontalAdvance(port_name) + self.PORT_MARGIN * 2
            )

        self._width = max(
            self.MIN_WIDTH,
            title_width,
            max_input_width + max_output_width + self.PADDING * 2
        )

        # Calculate height
        num_ports = max(len(self._node.inputs), len(self._node.outputs))
        self._height = self.HEADER_HEIGHT + num_ports * self.PORT_SPACING + self.PADDING

    def _create_ports(self) -> None:
        """Create port widgets."""
        y = self.HEADER_HEIGHT + self.PORT_SPACING // 2

        # Input ports (left side)
        for port_name, port in self._node.inputs.items():
            port_widget = PortWidget(
                self._node,
                port_name,
                port.port_type.name,
                is_input=True,
                parent=self,
            )
            port_widget.setPos(0, y)
            port_widget.pressed.connect(self._on_port_pressed)
            self._input_ports[port_name] = port_widget
            y += self.PORT_SPACING

        # Reset y for outputs
        y = self.HEADER_HEIGHT + self.PORT_SPACING // 2

        # Output ports (right side)
        for port_name, port in self._node.outputs.items():
            port_widget = PortWidget(
                self._node,
                port_name,
                port.port_type.name,
                is_input=False,
                parent=self,
            )
            port_widget.setPos(self._width, y)
            port_widget.pressed.connect(self._on_port_pressed)
            self._output_ports[port_name] = port_widget
            y += self.PORT_SPACING

        # Update height if needed
        total_ports = max(len(self._input_ports), len(self._output_ports))
        self._height = max(self._height, self.HEADER_HEIGHT + total_ports * self.PORT_SPACING + self.PADDING)

    def boundingRect(self) -> QRectF:
        """Get bounding rectangle."""
        return QRectF(-2, -2, self._width + 4, self._height + 4)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the node."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get category color
        category = getattr(self._node, "category", "default")
        base_color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(0, 0, self._width, self._height),
            self.CORNER_RADIUS,
            self.CORNER_RADIUS
        )

        # Draw body
        body_color = QColor(50, 50, 55)
        if self.isSelected():
            body_color = body_color.lighter(120)

        painter.fillPath(path, QBrush(body_color))

        # Draw header gradient
        header_path = QPainterPath()
        header_path.addRoundedRect(
            QRectF(0, 0, self._width, self.HEADER_HEIGHT),
            self.CORNER_RADIUS,
            self.CORNER_RADIUS
        )
        # Clip to top rounded corners only
        header_path = header_path.intersected(path)

        gradient = QLinearGradient(0, 0, 0, self.HEADER_HEIGHT)
        gradient.setColorAt(0, base_color)
        gradient.setColorAt(1, base_color.darker(130))
        painter.fillPath(header_path, QBrush(gradient))

        # Draw border
        border_color = base_color if self.isSelected() else QColor(80, 80, 85)
        painter.setPen(QPen(border_color, 2 if self.isSelected() else 1))
        painter.drawPath(path)

        # Draw title
        painter.setPen(QPen(QColor(240, 240, 240)))
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(self.PADDING, 0, self._width - self.PADDING * 2, self.HEADER_HEIGHT),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter,
            self._node.name
        )

        # Draw port labels
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(180, 180, 180)))

        y = self.HEADER_HEIGHT + self.PORT_SPACING // 2

        # Input port labels
        for port_name in self._node.inputs:
            painter.drawText(
                QRectF(self.PORT_MARGIN, y - 8, self._width / 2, 16),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                port_name
            )
            y += self.PORT_SPACING

        y = self.HEADER_HEIGHT + self.PORT_SPACING // 2

        # Output port labels
        for port_name in self._node.outputs:
            painter.drawText(
                QRectF(self._width / 2, y - 8, self._width / 2 - self.PORT_MARGIN, 16),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                port_name
            )
            y += self.PORT_SPACING

    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Snap to grid
            new_pos = value
            grid_size = 10
            new_x = round(new_pos.x() / grid_size) * grid_size
            new_y = round(new_pos.y() / grid_size) * grid_size
            return QPointF(new_x, new_y)

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Notify of position change
            if self._drag_start_pos is not None:
                self.position_changed.emit(self, self._drag_start_pos, value)

        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.selected_changed.emit(self, value)

        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start_pos != self.pos():
                self.position_changed.emit(self, self._drag_start_pos, self.pos())
            self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _on_port_pressed(self, port: PortWidget) -> None:
        """Forward port press signal."""
        self.port_pressed.emit(port)
