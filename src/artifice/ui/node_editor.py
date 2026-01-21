"""
Node Editor canvas widget.

Provides a zoomable, pannable canvas for creating and editing node graphs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QMimeData
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QBrush,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
    QDragEnterEvent,
    QDropEvent,
)
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QApplication,
    QGraphicsRectItem,
    QMenu,
)

from artifice.core.graph import NodeGraph
from artifice.core.registry import get_registry
from artifice.ui.node_widget import NodeWidget, PortWidget
from artifice.ui.connection import ConnectionItem, TempConnectionItem
from artifice.ui.undo import (
    UndoStack,
    AddNodeCommand,
    RemoveNodeCommand,
    ConnectCommand,
    DisconnectCommand,
    MoveNodeCommand,
    CompositeCommand,
)

if TYPE_CHECKING:
    from artifice.core.node import Node


class NodeEditorScene(QGraphicsScene):
    """
    Graphics scene for the node editor.

    Manages node widgets and connections.
    """

    # Grid settings
    GRID_SIZE = 20
    GRID_COLOR_MAJOR = QColor(60, 60, 60)
    GRID_COLOR_MINOR = QColor(45, 45, 45)
    BACKGROUND_COLOR = QColor(35, 35, 35)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(self.BACKGROUND_COLOR))

        # Scene bounds
        self.setSceneRect(-10000, -10000, 20000, 20000)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw the grid background."""
        super().drawBackground(painter, rect)

        # Draw grid
        left = int(rect.left()) - (int(rect.left()) % self.GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % self.GRID_SIZE)

        # Minor grid lines
        painter.setPen(QPen(self.GRID_COLOR_MINOR, 0.5))
        for x in range(left, int(rect.right()), self.GRID_SIZE):
            if x % (self.GRID_SIZE * 5) != 0:
                painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top, int(rect.bottom()), self.GRID_SIZE):
            if y % (self.GRID_SIZE * 5) != 0:
                painter.drawLine(int(rect.left()), y, int(rect.right()), y)

        # Major grid lines
        painter.setPen(QPen(self.GRID_COLOR_MAJOR, 1.0))
        for x in range(left, int(rect.right()), self.GRID_SIZE * 5):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top, int(rect.bottom()), self.GRID_SIZE * 5):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)


class NodeEditorWidget(QGraphicsView):
    """
    Main node editor widget.

    A QGraphicsView-based canvas for creating and editing node graphs.
    Supports zooming, panning, node creation, and connection drawing.
    """

    # Signals
    node_selected = Signal(object)  # Node
    node_deselected = Signal()
    graph_modified = Signal()

    # Zoom settings
    ZOOM_MIN = 0.1
    ZOOM_MAX = 4.0
    ZOOM_FACTOR = 1.15

    def __init__(self, graph: NodeGraph, undo_stack: UndoStack, parent=None):
        super().__init__(parent)

        self._graph = graph
        self._undo_stack = undo_stack

        # Node widgets
        self._node_widgets: dict[str, NodeWidget] = {}  # node_id -> widget
        self._connection_items: list[ConnectionItem] = []

        # Interaction state
        self._temp_connection: TempConnectionItem | None = None
        self._source_port: PortWidget | None = None
        self._is_panning = False
        self._pan_start = QPointF()
        self._selection_rect: QGraphicsRectItem | None = None
        self._selection_start = QPointF()
        self._moving_nodes: list[tuple[NodeWidget, QPointF]] = []

        # Clipboard
        self._clipboard: dict | None = None

        # Set up scene
        self._scene = NodeEditorScene(self)
        self.setScene(self._scene)

        # Configure view
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Accept drops
        self.setAcceptDrops(True)

        # Initial zoom
        self._zoom = 1.0

        # Sync with graph
        self._sync_from_graph()

    @property
    def graph(self) -> NodeGraph:
        """Get the node graph."""
        return self._graph

    def set_graph(self, graph: NodeGraph) -> None:
        """Set a new graph."""
        self._graph = graph
        self.clear()
        self._sync_from_graph()

    def clear(self) -> None:
        """Clear all widgets."""
        # Clear mouse grabber to prevent "ungrabMouse without scene" errors
        if self._scene.mouseGrabberItem():
            self._scene.mouseGrabberItem().ungrabMouse()

        for widget in list(self._node_widgets.values()):
            widget.setSelected(False)
            self._scene.removeItem(widget)
        self._node_widgets.clear()

        for conn in list(self._connection_items):
            conn.setSelected(False)
            self._scene.removeItem(conn)
        self._connection_items.clear()

        if self._temp_connection:
            self._scene.removeItem(self._temp_connection)
            self._temp_connection = None

    def _sync_from_graph(self) -> None:
        """Sync widgets from graph state."""
        # Create widgets for existing nodes
        for node in self._graph.nodes.values():
            self._create_node_widget(node)

        # Create connection items
        for conn in self._graph.get_connections():
            self._create_connection_item(conn)

    def _create_node_widget(self, node: Node) -> NodeWidget:
        """Create a widget for a node."""
        widget = NodeWidget(node)
        self._scene.addItem(widget)
        self._node_widgets[node.id] = widget

        # Position
        pos = getattr(node, "position", (0, 0))
        widget.setPos(pos[0], pos[1])

        # Connect signals
        widget.position_changed.connect(self._on_node_moved)
        widget.port_pressed.connect(self._on_port_pressed)
        widget.selected_changed.connect(self._on_node_selection_changed)

        return widget

    def _create_connection_item(self, conn) -> ConnectionItem:
        """Create a connection item."""
        source_widget = self._node_widgets.get(conn.source_node_id)
        target_widget = self._node_widgets.get(conn.dest_node_id)

        if source_widget and target_widget:
            source_port = source_widget.get_output_port(conn.source_port)
            target_port = target_widget.get_input_port(conn.dest_port)

            if source_port and target_port:
                item = ConnectionItem(source_port, target_port)
                self._scene.addItem(item)
                self._connection_items.append(item)
                return item

        return None

    # --- Node Operations ---

    def add_node_at_position(self, node_type: str, x: float, y: float) -> Node | None:
        """Add a node at a specific position."""
        registry = get_registry()
        node_class = registry.get(node_type)
        if not node_class:
            return None

        node = node_class()
        node.position = (x, y)

        cmd = AddNodeCommand(self._graph, node, (x, y))
        self._undo_stack.push(cmd)

        # Create widget
        widget = self._create_node_widget(node)

        self.graph_modified.emit()
        return node

    def add_node_at_center(self, node_type: str) -> Node | None:
        """Add a node at the center of the view."""
        center = self.mapToScene(self.viewport().rect().center())
        return self.add_node_at_position(node_type, center.x(), center.y())

    def add_node(self, node_type: str, **kwargs) -> Node | None:
        """Add a node with optional parameters."""
        node = self.add_node_at_center(node_type)
        if node and kwargs:
            for key, value in kwargs.items():
                if key == "path" and hasattr(node, "parameters"):
                    node.set_parameter("path", value)
                elif hasattr(node, key):
                    setattr(node, key, value)
        return node

    def delete_node(self, node: Node) -> None:
        """Delete a node."""
        widget = self._node_widgets.get(node.id)
        if not widget:
            return

        # Clear mouse grabber to prevent "ungrabMouse without scene" errors
        if self._scene.mouseGrabberItem():
            self._scene.mouseGrabberItem().ungrabMouse()

        # Remove connections first
        for conn in list(self._connection_items):
            if conn.source_port.node == node or conn.target_port.node == node:
                conn.setSelected(False)
                self._scene.removeItem(conn)
                self._connection_items.remove(conn)

        # Remove widget
        widget.setSelected(False)
        self._scene.removeItem(widget)
        del self._node_widgets[node.id]

        # Remove from graph via undo command
        cmd = RemoveNodeCommand(self._graph, node)
        self._undo_stack.push(cmd)

        self.graph_modified.emit()

    def delete_selection(self) -> None:
        """Delete selected nodes."""
        selected = self.get_selected_nodes()
        if not selected:
            return

        # Clear mouse grabber to prevent "ungrabMouse without scene" errors
        if self._scene.mouseGrabberItem():
            self._scene.mouseGrabberItem().ungrabMouse()

        commands = []
        for node in selected:
            widget = self._node_widgets.get(node.id)
            if widget:
                # Remove connections
                for conn in list(self._connection_items):
                    if conn.source_port.node == node or conn.target_port.node == node:
                        conn.setSelected(False)
                        self._scene.removeItem(conn)
                        self._connection_items.remove(conn)

                # Remove widget
                widget.setSelected(False)
                self._scene.removeItem(widget)
                del self._node_widgets[node.id]

                commands.append(RemoveNodeCommand(self._graph, node))

        if commands:
            if len(commands) == 1:
                self._undo_stack.push(commands[0])
            else:
                self._undo_stack.push(CompositeCommand(commands, "Delete nodes"))

        self.graph_modified.emit()

    def get_selected_nodes(self) -> list[Node]:
        """Get currently selected nodes."""
        selected = []
        for node_id, widget in self._node_widgets.items():
            if widget.isSelected():
                node = self._graph.get_node(node_id)
                if node:
                    selected.append(node)
        return selected

    # --- Connection Operations ---

    def start_connection(self, port: PortWidget) -> None:
        """Start drawing a connection from a port."""
        # If clicking on a connected input port, disconnect and reconnect from source
        if port.is_input and port.is_connected:
            # Find and remove the existing connection
            for conn in list(self._connection_items):
                if conn.target_port == port:
                    source_port = conn.source_port
                    self.delete_connection(conn)
                    # Start a new connection from the source
                    self._source_port = source_port
                    self._temp_connection = TempConnectionItem(source_port)
                    self._scene.addItem(self._temp_connection)
                    self._highlight_compatible_ports(source_port)
                    return

        self._source_port = port
        self._temp_connection = TempConnectionItem(port)
        self._scene.addItem(self._temp_connection)
        self._highlight_compatible_ports(port)

    def _highlight_compatible_ports(self, source_port: PortWidget) -> None:
        """Highlight ports that can accept a connection from the source."""
        for widget in self._node_widgets.values():
            # Check input ports if source is output, and vice versa
            if source_port.is_input:
                ports_to_check = widget._output_ports.values()
            else:
                ports_to_check = widget._input_ports.values()

            for port in ports_to_check:
                # Don't highlight ports on the same node
                if port.node == source_port.node:
                    continue
                # Check type compatibility (same type or ANY)
                if port.port_type == source_port.port_type or port.port_type == "ANY" or source_port.port_type == "ANY":
                    port.is_compatible_target = True

    def _clear_port_highlights(self) -> None:
        """Clear all port compatibility highlights."""
        for widget in self._node_widgets.values():
            for port in widget._input_ports.values():
                port.is_compatible_target = False
            for port in widget._output_ports.values():
                port.is_compatible_target = False

    def update_temp_connection(self, pos: QPointF) -> None:
        """Update temporary connection endpoint."""
        if self._temp_connection:
            self._temp_connection.set_end_pos(pos)

    def complete_connection(self, target_port: PortWidget) -> bool:
        """Complete a connection to a target port."""
        if not self._source_port or not self._temp_connection:
            return False

        # Remove temp connection
        self._scene.removeItem(self._temp_connection)
        self._temp_connection = None

        # Validate connection
        source = self._source_port
        target = target_port

        # Ensure source is output and target is input
        if source.is_input:
            source, target = target, source

        if source.is_input or not target.is_input:
            self._source_port = None
            return False

        # Check if already connected
        for conn in self._graph.get_connections():
            if (conn.source_node_id == source.node.id and
                conn.source_port == source.port_name and
                conn.dest_node_id == target.node.id and
                conn.dest_port == target.port_name):
                self._source_port = None
                return False

        # Create connection
        cmd = ConnectCommand(
            self._graph,
            source.node,
            source.port_name,
            target.node,
            target.port_name,
        )
        self._undo_stack.push(cmd)

        # Create visual connection
        item = ConnectionItem(source, target)
        self._scene.addItem(item)
        self._connection_items.append(item)

        self._source_port = None
        self._clear_port_highlights()
        self.graph_modified.emit()
        return True

    def cancel_connection(self) -> None:
        """Cancel the current connection drawing."""
        if self._temp_connection:
            self._scene.removeItem(self._temp_connection)
            self._temp_connection = None
        self._source_port = None
        self._clear_port_highlights()

    def delete_connection(self, conn_item: ConnectionItem) -> None:
        """Delete a specific connection."""
        source_port = conn_item.source_port
        target_port = conn_item.target_port

        # Disconnect in the graph
        self._graph.disconnect(
            source_port.node,
            source_port.port_name,
            target_port.node,
            target_port.port_name,
        )

        # Update port visual state - check BEFORE removing from list
        # We need to exclude this connection when checking
        self._connection_items.remove(conn_item)
        source_port.is_connected = self._port_has_connections(source_port)
        target_port.is_connected = self._port_has_connections(target_port)

        # Clear mouse grabber to prevent "ungrabMouse without scene" errors
        if self._scene.mouseGrabberItem() == conn_item:
            conn_item.ungrabMouse()

        # Clear selection before removing
        conn_item.setSelected(False)

        # Remove visual connection
        self._scene.removeItem(conn_item)

        self.graph_modified.emit()

    def delete_selected_connections(self) -> None:
        """Delete all selected connections."""
        for conn in list(self._connection_items):
            if conn.isSelected():
                self.delete_connection(conn)

    def disconnect_port(self, port: PortWidget) -> None:
        """Disconnect all connections from a port."""
        connections_to_remove = []

        for conn in self._connection_items:
            if conn.source_port == port or conn.target_port == port:
                connections_to_remove.append(conn)

        for conn in connections_to_remove:
            self.delete_connection(conn)

    def _port_has_connections(self, port: PortWidget) -> bool:
        """Check if a port still has any connections."""
        for conn in self._connection_items:
            if conn.source_port == port or conn.target_port == port:
                return True
        return False

    def duplicate_node(self, node: Node) -> Node | None:
        """Duplicate a node."""
        # Get the node class
        registry = get_registry()
        node_class = registry.get(node.__class__.__name__)
        if not node_class:
            return None

        # Create a new node at offset position
        new_node = node_class()
        new_pos = (node.position[0] + 50, node.position[1] + 50)
        new_node.position = new_pos

        # Copy parameter values
        for name, param in node.parameters.items():
            if name in new_node.parameters:
                new_node.parameters[name].set(param.value)

        # Add to graph
        self._graph.add_node(new_node)

        # Create widget
        self._create_node_widget(new_node)

        self.graph_modified.emit()
        return new_node

    def connect(
        self,
        source_node: Node,
        source_port: str,
        target_node: Node,
        target_port: str,
    ) -> bool:
        """Programmatically create a connection."""
        cmd = ConnectCommand(
            self._graph,
            source_node,
            source_port,
            target_node,
            target_port,
        )
        self._undo_stack.push(cmd)

        # Create visual connection
        source_widget = self._node_widgets.get(source_node.id)
        target_widget = self._node_widgets.get(target_node.id)

        if source_widget and target_widget:
            source_port_widget = source_widget.get_output_port(source_port)
            target_port_widget = target_widget.get_input_port(target_port)

            if source_port_widget and target_port_widget:
                item = ConnectionItem(source_port_widget, target_port_widget)
                self._scene.addItem(item)
                self._connection_items.append(item)

        self.graph_modified.emit()
        return True

    # --- Selection ---

    def select_all(self) -> None:
        """Select all nodes."""
        for widget in self._node_widgets.values():
            widget.setSelected(True)

    def clear_selection(self) -> None:
        """Clear selection."""
        for widget in self._node_widgets.values():
            widget.setSelected(False)

    # --- Clipboard ---

    def copy_selection(self) -> None:
        """Copy selected nodes to clipboard."""
        selected = self.get_selected_nodes()
        if not selected:
            return

        # Serialize selected nodes
        self._clipboard = {
            "nodes": [node.to_dict() for node in selected],
            # TODO: Include connections between selected nodes
        }

    def cut_selection(self) -> None:
        """Cut selected nodes."""
        self.copy_selection()
        self.delete_selection()

    def paste(self) -> None:
        """Paste nodes from clipboard."""
        if not self._clipboard:
            return

        # TODO: Implement paste with position offset
        pass

    # --- Zoom/Pan ---

    def zoom_in(self) -> None:
        """Zoom in."""
        self._zoom_by(self.ZOOM_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out."""
        self._zoom_by(1.0 / self.ZOOM_FACTOR)

    def zoom_reset(self) -> None:
        """Reset zoom to 100%."""
        self.resetTransform()
        self._zoom = 1.0

    def fit_to_view(self) -> None:
        """Fit all nodes in view."""
        if not self._node_widgets:
            return

        # Get bounding rect of all nodes
        rects = [w.sceneBoundingRect() for w in self._node_widgets.values()]
        bounds = rects[0]
        for r in rects[1:]:
            bounds = bounds.united(r)

        # Add padding
        bounds.adjust(-50, -50, 50, 50)

        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)

        # Update zoom level
        self._zoom = self.transform().m11()

    def _zoom_by(self, factor: float) -> None:
        """Apply zoom factor."""
        new_zoom = self._zoom * factor
        if self.ZOOM_MIN <= new_zoom <= self.ZOOM_MAX:
            self.scale(factor, factor)
            self._zoom = new_zoom

    # --- Event Handlers ---

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self._zoom_by(self.ZOOM_FACTOR)
        else:
            self._zoom_by(1.0 / self.ZOOM_FACTOR)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self._is_panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on empty space
            item = self.itemAt(event.pos())
            if item is None:
                # Start selection rectangle
                self._selection_start = self.mapToScene(event.pos())
                self._selection_rect = QGraphicsRectItem()
                self._selection_rect.setPen(QPen(QColor(100, 150, 255), 1, Qt.PenStyle.DashLine))
                self._selection_rect.setBrush(QBrush(QColor(100, 150, 255, 30)))
                self._scene.addItem(self._selection_rect)

                # Clear selection unless shift is held
                if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self.clear_selection()
                    self.node_deselected.emit()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        if self._is_panning:
            # Pan the view
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        elif self._temp_connection:
            # Update temp connection
            self.update_temp_connection(self.mapToScene(event.pos()))
        elif self._selection_rect:
            # Update selection rectangle
            current = self.mapToScene(event.pos())
            rect = QRectF(self._selection_start, current).normalized()
            self._selection_rect.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._selection_rect:
                # Select nodes in rectangle
                rect = self._selection_rect.rect()
                for widget in self._node_widgets.values():
                    if rect.intersects(widget.sceneBoundingRect()):
                        widget.setSelected(True)

                self._scene.removeItem(self._selection_rect)
                self._selection_rect = None
            elif self._temp_connection:
                # Check if over a port
                item = self.itemAt(event.pos())
                if isinstance(item, PortWidget):
                    self.complete_connection(item)
                else:
                    self.cancel_connection()

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Delete selected connections first
            self.delete_selected_connections()
            # Then delete selected nodes
            self.delete_selection()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_connection()
            self.clear_selection()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        """Handle right-click context menu."""
        pos = event.pos()
        scene_pos = self.mapToScene(pos)
        item = self.itemAt(pos)

        menu = QMenu(self)

        if isinstance(item, ConnectionItem):
            # Context menu for connection
            disconnect_action = menu.addAction("Disconnect")
            disconnect_action.triggered.connect(lambda: self.delete_connection(item))
        elif isinstance(item, PortWidget):
            # Context menu for port - disconnect all connections
            if item.is_connected:
                disconnect_action = menu.addAction("Disconnect All")
                disconnect_action.triggered.connect(lambda: self.disconnect_port(item))
        elif isinstance(item, NodeWidget) or (item and item.parentItem() and isinstance(item.parentItem(), NodeWidget)):
            # Context menu for node
            node_widget = item if isinstance(item, NodeWidget) else item.parentItem()
            delete_action = menu.addAction("Delete Node")
            delete_action.triggered.connect(lambda: self.delete_node(node_widget.node))
            menu.addSeparator()
            duplicate_action = menu.addAction("Duplicate")
            duplicate_action.triggered.connect(lambda: self.duplicate_node(node_widget.node))
        else:
            # Context menu for empty space - add nodes
            add_menu = menu.addMenu("Add Node")
            registry = get_registry()

            # Group by category
            nodes_by_category: dict[str, list[tuple[str, str]]] = {}
            for name, node_class in registry.get_registry().items():
                category = getattr(node_class, "category", "Uncategorized")
                display_name = getattr(node_class, "name", name)
                if category not in nodes_by_category:
                    nodes_by_category[category] = []
                nodes_by_category[category].append((name, display_name))

            for category in sorted(nodes_by_category.keys()):
                cat_menu = add_menu.addMenu(category)
                for node_type, display_name in sorted(nodes_by_category[category], key=lambda x: x[1]):
                    action = cat_menu.addAction(display_name)
                    # Capture the variables in the lambda
                    action.triggered.connect(
                        lambda checked, nt=node_type, x=scene_pos.x(), y=scene_pos.y():
                            self.add_node_at_position(nt, x, y)
                    )

        if menu.actions():
            menu.exec(event.globalPos())

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter."""
        if event.mimeData().hasFormat("application/x-artifice-node"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        """Handle drag move - required for drop to work."""
        if event.mimeData().hasFormat("application/x-artifice-node"):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop."""
        if event.mimeData().hasFormat("application/x-artifice-node"):
            node_type = event.mimeData().data("application/x-artifice-node").data().decode()
            pos = self.mapToScene(event.position().toPoint())
            self.add_node_at_position(node_type, pos.x(), pos.y())
            event.acceptProposedAction()

    # --- Slots ---

    def _on_node_moved(self, widget: NodeWidget, old_pos: QPointF, new_pos: QPointF) -> None:
        """Handle node movement."""
        node = widget.node
        node.position = (new_pos.x(), new_pos.y())

        cmd = MoveNodeCommand(node, (old_pos.x(), old_pos.y()), (new_pos.x(), new_pos.y()))
        # Don't push to undo stack during drag, only on release
        # self._undo_stack.push(cmd)

        # Update connections
        for conn in self._connection_items:
            if conn.source_port.node == node or conn.target_port.node == node:
                conn.update_path()

        self.graph_modified.emit()

    def _on_port_pressed(self, port: PortWidget) -> None:
        """Handle port press to start connection."""
        self.start_connection(port)

    def _on_node_selection_changed(self, widget: NodeWidget, selected: bool) -> None:
        """Handle node selection change."""
        if selected:
            self.node_selected.emit(widget.node)
        elif not self.get_selected_nodes():
            self.node_deselected.emit()
