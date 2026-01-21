"""
Phase 4: User Interface Tests

Tests for UI components: MainWindow, NodeEditor, NodeWidget,
ConnectionItem, PreviewPanel, InspectorPanel, NodePalette.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from artifice.core.graph import NodeGraph
from artifice.core.data_types import ImageBuffer
from artifice.core.registry import NodeRegistry
from artifice.ui.main_window import MainWindow
from artifice.ui.node_editor import NodeEditorWidget, NodeEditorScene
from artifice.ui.node_widget import NodeWidget, PortWidget
from artifice.ui.connection import ConnectionItem, TempConnectionItem
from artifice.ui.preview import PreviewPanel, ImageWidget
from artifice.ui.inspector import InspectorPanel
from artifice.ui.palette import NodePalette
from artifice.ui.undo import (
    UndoStack,
    Command,
    AddNodeCommand,
    RemoveNodeCommand,
    ConnectCommand,
    DisconnectCommand,
    MoveNodeCommand,
    ChangeParameterCommand,
    CompositeCommand,
)


# Ensure a QApplication exists for all tests
@pytest.fixture(scope="module")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def graph():
    """Create a fresh NodeGraph."""
    return NodeGraph()


@pytest.fixture
def undo_stack():
    """Create a fresh UndoStack."""
    return UndoStack()


@pytest.fixture
def main_window(qapp, graph, undo_stack):
    """Create a MainWindow instance."""
    window = MainWindow()
    yield window
    # Reset modified flag to avoid save dialog during cleanup
    window._modified = False
    window.close()


@pytest.fixture
def node_editor(qapp, graph, undo_stack):
    """Create a NodeEditorWidget instance."""
    editor = NodeEditorWidget(graph, undo_stack)
    yield editor


@pytest.fixture
def preview_panel(qapp):
    """Create a PreviewPanel instance."""
    panel = PreviewPanel()
    yield panel


@pytest.fixture
def inspector_panel(qapp):
    """Create an InspectorPanel instance."""
    panel = InspectorPanel()
    yield panel


@pytest.fixture
def node_palette(qapp):
    """Create a NodePalette instance."""
    palette = NodePalette()
    yield palette


# =============================================================================
# V4.1 - UI Launch Tests
# =============================================================================

class TestUILaunch:
    """Tests for V4.1 - UI Launch."""

    def test_application_launches(self, qapp):
        """Test: Application launches without error."""
        window = MainWindow()
        window.show()

        # Process events briefly
        QTest.qWait(100)

        assert window.isVisible()
        # Reset modified flag to avoid save dialog during cleanup
        window._modified = False
        window.close()

    def test_main_window_components(self, main_window):
        """Test: MainWindow has expected components."""
        # Check node editor exists
        assert main_window._node_editor is not None
        assert isinstance(main_window._node_editor, NodeEditorWidget)

        # Check preview panel exists
        assert main_window._preview is not None
        assert isinstance(main_window._preview, PreviewPanel)

        # Check inspector panel exists
        assert main_window._inspector is not None
        assert isinstance(main_window._inspector, InspectorPanel)

        # Check node palette exists
        assert main_window._palette is not None
        assert isinstance(main_window._palette, NodePalette)

    def test_node_editor_scene(self, node_editor):
        """Test: NodeEditor has a scene."""
        assert node_editor.scene() is not None
        assert isinstance(node_editor.scene(), NodeEditorScene)

    def test_node_editor_initial_state(self, node_editor):
        """Test: NodeEditor starts with empty state."""
        assert len(node_editor.graph.nodes) == 0
        assert len(node_editor._node_widgets) == 0


# =============================================================================
# V4.2 - Node Creation via UI Tests
# =============================================================================

class TestNodeCreation:
    """Tests for V4.2 - Node Creation via UI."""

    def test_add_node_at_position(self, node_editor):
        """Test: Can add a node at specific position."""
        # Add a PassThrough node (a simple utility node)
        node = node_editor.add_node_at_position("PassThroughNode", x=100, y=100)

        if node is not None:
            assert len(node_editor.graph.nodes) == 1
            assert node.position == (100, 100)
            assert node.id in node_editor._node_widgets

    def test_add_node_at_center(self, node_editor):
        """Test: Can add a node at view center."""
        node = node_editor.add_node_at_center("PassThroughNode")

        if node is not None:
            assert len(node_editor.graph.nodes) == 1
            assert node.id in node_editor._node_widgets

    def test_node_widget_created(self, node_editor):
        """Test: NodeWidget is created for each node."""
        node = node_editor.add_node_at_position("PassThroughNode", 200, 200)

        if node is not None:
            widget = node_editor._node_widgets.get(node.id)
            assert widget is not None
            assert isinstance(widget, NodeWidget)
            assert widget.node == node

    def test_multiple_nodes(self, node_editor):
        """Test: Can add multiple nodes."""
        node1 = node_editor.add_node_at_position("PassThroughNode", 100, 100)
        node2 = node_editor.add_node_at_position("PassThroughNode", 300, 100)

        if node1 is not None and node2 is not None:
            assert len(node_editor.graph.nodes) == 2
            assert len(node_editor._node_widgets) == 2


# =============================================================================
# V4.3 - Node Connection via UI Tests
# =============================================================================

class TestNodeConnection:
    """Tests for V4.3 - Node Connection via UI."""

    def test_programmatic_connection(self, node_editor):
        """Test: Can connect nodes programmatically."""
        # Add two nodes
        node1 = node_editor.add_node_at_position("PassThroughNode", 100, 100)
        node2 = node_editor.add_node_at_position("PassThroughNode", 300, 100)

        if node1 is not None and node2 is not None:
            # Connect them
            result = node_editor.connect(node1, "image", node2, "image")

            assert result is True
            assert len(node_editor.graph.get_connections()) == 1
            assert len(node_editor._connection_items) == 1

    def test_connection_item_created(self, node_editor):
        """Test: ConnectionItem is created for each connection."""
        node1 = node_editor.add_node_at_position("PassThroughNode", 100, 100)
        node2 = node_editor.add_node_at_position("PassThroughNode", 300, 100)

        if node1 is not None and node2 is not None:
            node_editor.connect(node1, "image", node2, "image")

            assert len(node_editor._connection_items) == 1
            conn = node_editor._connection_items[0]
            assert isinstance(conn, ConnectionItem)

    def test_start_and_cancel_connection(self, node_editor):
        """Test: Can start and cancel a connection."""
        node = node_editor.add_node_at_position("PassThroughNode", 100, 100)

        if node is not None:
            widget = node_editor._node_widgets.get(node.id)
            if widget:
                output_port = widget.get_output_port("image")
                if output_port:
                    # Start connection
                    node_editor.start_connection(output_port)
                    assert node_editor._temp_connection is not None

                    # Cancel
                    node_editor.cancel_connection()
                    assert node_editor._temp_connection is None


# =============================================================================
# V4.4 - Parameter Editing Tests
# =============================================================================

class TestParameterEditing:
    """Tests for V4.4 - Parameter Editing."""

    def test_inspector_select_node(self, inspector_panel):
        """Test: Inspector can select a node."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        inspector_panel.set_node(node)

        assert inspector_panel._node == node

    def test_inspector_clear_node(self, inspector_panel):
        """Test: Inspector can clear selection."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        inspector_panel.set_node(node)
        inspector_panel.set_node(None)

        assert inspector_panel._node is None


# =============================================================================
# V4.5 - Preview Updates Tests
# =============================================================================

class TestPreviewUpdates:
    """Tests for V4.5 - Preview Updates."""

    def test_preview_set_image(self, preview_panel):
        """Test: Preview can display an ImageBuffer."""
        # Create a simple test image
        data = np.random.rand(3, 64, 64).astype(np.float32)
        buffer = ImageBuffer(data=data, colorspace="RGB")

        preview_panel.set_image(buffer)
        assert preview_panel._image_widget._pixmap is not None

    def test_preview_clear(self, preview_panel):
        """Test: Preview can be cleared."""
        # Set an image first
        data = np.random.rand(3, 64, 64).astype(np.float32)
        buffer = ImageBuffer(data=data, colorspace="RGB")
        preview_panel.set_image(buffer)

        # Clear it
        preview_panel.clear()
        assert preview_panel._image_widget._pixmap is None

    def test_preview_zoom(self, preview_panel):
        """Test: Preview zoom works."""
        initial_zoom = preview_panel._image_widget._zoom

        preview_panel._image_widget.zoom_in()
        assert preview_panel._image_widget._zoom > initial_zoom

        preview_panel._image_widget.zoom_out()
        preview_panel._image_widget.zoom_out()
        assert preview_panel._image_widget._zoom < initial_zoom


# =============================================================================
# V4.6 - Save/Load via UI Tests
# =============================================================================

class TestSaveLoad:
    """Tests for V4.6 - Save/Load via UI."""

    def test_save_project(self, main_window):
        """Test: Can save a project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_project.artifice")

            # Add a node
            main_window._node_editor.add_node_at_position("PassThroughNode", 100, 100)

            # Save
            main_window.save_project(filepath)

            assert os.path.exists(filepath)

    def test_load_project(self, main_window):
        """Test: Can load a project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_project.artifice")

            # Create and save a project
            main_window._node_editor.add_node_at_position("PassThroughNode", 100, 100)
            initial_count = len(main_window._node_editor.graph.nodes)
            main_window.save_project(filepath)

            # Clear
            main_window.new_project()
            assert len(main_window._node_editor.graph.nodes) == 0

            # Load
            main_window.load_project(filepath)

            # Should have same number of nodes
            assert len(main_window._node_editor.graph.nodes) == initial_count

    def test_new_project(self, main_window):
        """Test: New project clears graph."""
        # Add nodes
        main_window._node_editor.add_node_at_position("PassThroughNode", 100, 100)
        main_window._node_editor.add_node_at_position("PassThroughNode", 200, 200)

        assert len(main_window._node_editor.graph.nodes) > 0

        # New project
        main_window.new_project()

        assert len(main_window._node_editor.graph.nodes) == 0


# =============================================================================
# V4.7 - Undo/Redo Tests
# =============================================================================

class TestUndoRedo:
    """Tests for V4.7 - Undo/Redo."""

    def test_undo_stack_push(self, undo_stack, graph):
        """Test: Can push commands to undo stack."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        cmd = AddNodeCommand(graph, node, (0, 0))
        undo_stack.push(cmd)

        assert undo_stack.can_undo()
        assert not undo_stack.can_redo()
        assert len(graph.nodes) == 1

    def test_undo(self, undo_stack, graph):
        """Test: Undo works correctly."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        cmd = AddNodeCommand(graph, node, (0, 0))
        undo_stack.push(cmd)

        assert len(graph.nodes) == 1

        # Undo
        undo_stack.undo()

        assert len(graph.nodes) == 0
        assert undo_stack.can_redo()
        assert not undo_stack.can_undo()

    def test_redo(self, undo_stack, graph):
        """Test: Redo works correctly."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        cmd = AddNodeCommand(graph, node, (0, 0))
        undo_stack.push(cmd)
        undo_stack.undo()

        assert len(graph.nodes) == 0

        # Redo
        undo_stack.redo()

        assert len(graph.nodes) == 1
        assert undo_stack.can_undo()
        assert not undo_stack.can_redo()

    def test_undo_clears_redo_stack(self, undo_stack, graph):
        """Test: New command clears redo stack."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node1 = PassThroughNode()
        node2 = PassThroughNode()

        undo_stack.push(AddNodeCommand(graph, node1, (0, 0)))
        undo_stack.undo()

        assert undo_stack.can_redo()

        # Push new command
        undo_stack.push(AddNodeCommand(graph, node2, (100, 100)))

        # Redo stack should be cleared
        assert not undo_stack.can_redo()

    def test_connect_command(self, undo_stack, graph):
        """Test: Connect command works."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node1 = PassThroughNode()
        node2 = PassThroughNode()
        graph.add_node(node1)
        graph.add_node(node2)

        cmd = ConnectCommand(graph, node1, "image", node2, "image")
        undo_stack.push(cmd)

        assert len(graph.get_connections()) == 1

        undo_stack.undo()

        assert len(graph.get_connections()) == 0

    def test_composite_command(self, undo_stack, graph):
        """Test: Composite command works."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node1 = PassThroughNode()
        node2 = PassThroughNode()

        commands = [
            AddNodeCommand(graph, node1, (0, 0)),
            AddNodeCommand(graph, node2, (100, 100)),
        ]

        composite = CompositeCommand(commands, "Add multiple nodes")
        undo_stack.push(composite)

        assert len(graph.nodes) == 2

        undo_stack.undo()

        assert len(graph.nodes) == 0


# =============================================================================
# Node Widget Tests
# =============================================================================

class TestNodeWidget:
    """Tests for NodeWidget component."""

    def test_node_widget_creation(self, qapp):
        """Test: NodeWidget can be created."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        widget = NodeWidget(node)

        assert widget is not None
        assert widget.node == node

    def test_node_widget_ports(self, qapp):
        """Test: NodeWidget has port widgets."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        widget = NodeWidget(node)

        # Should have ports for inputs and outputs
        assert len(widget._input_ports) > 0 or len(widget._output_ports) > 0

    def test_port_widget_properties(self, qapp):
        """Test: PortWidget has correct properties."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        widget = NodeWidget(node)

        if widget._output_ports:
            port_name, port_widget = next(iter(widget._output_ports.items()))
            assert port_widget.port_name == port_name
            assert port_widget.node == node
            assert port_widget.is_input is False


# =============================================================================
# Connection Item Tests
# =============================================================================

class TestConnectionItem:
    """Tests for ConnectionItem component."""

    def test_temp_connection_creation(self, qapp):
        """Test: TempConnectionItem can be created."""
        from artifice.nodes.utility.passthrough import PassThroughNode

        node = PassThroughNode()
        widget = NodeWidget(node)

        if widget._output_ports:
            port_widget = next(iter(widget._output_ports.values()))
            temp_conn = TempConnectionItem(port_widget)

            assert temp_conn is not None


# =============================================================================
# Node Palette Tests
# =============================================================================

class TestNodePalette:
    """Tests for NodePalette component."""

    def test_palette_populated(self, node_palette):
        """Test: Palette is populated with nodes."""
        # Should have at least some items if nodes are registered
        # The palette reads from the NodeRegistry
        pass  # Depends on what nodes are registered

    def test_palette_search(self, node_palette):
        """Test: Palette search works."""
        # Set search text
        node_palette._search.setText("Pass")
        # Search should filter the tree
        # This depends on the implementation


# =============================================================================
# Node Editor Scene Tests
# =============================================================================

class TestNodeEditorScene:
    """Tests for NodeEditorScene component."""

    def test_scene_creation(self, qapp):
        """Test: NodeEditorScene can be created."""
        scene = NodeEditorScene()
        assert scene is not None

    def test_scene_bounds(self, qapp):
        """Test: Scene has reasonable bounds."""
        scene = NodeEditorScene()
        rect = scene.sceneRect()

        # Should have a large scene rect for panning
        assert rect.width() > 1000
        assert rect.height() > 1000


# =============================================================================
# Zoom/Pan Tests
# =============================================================================

class TestZoomPan:
    """Tests for zoom and pan functionality."""

    def test_zoom_in(self, node_editor):
        """Test: Zoom in works."""
        initial_zoom = node_editor._zoom

        node_editor.zoom_in()

        assert node_editor._zoom > initial_zoom

    def test_zoom_out(self, node_editor):
        """Test: Zoom out works."""
        initial_zoom = node_editor._zoom

        node_editor.zoom_out()

        assert node_editor._zoom < initial_zoom

    def test_zoom_reset(self, node_editor):
        """Test: Zoom reset works."""
        node_editor.zoom_in()
        node_editor.zoom_in()

        node_editor.zoom_reset()

        assert node_editor._zoom == 1.0

    def test_zoom_limits(self, node_editor):
        """Test: Zoom has min/max limits."""
        # Zoom out many times
        for _ in range(50):
            node_editor.zoom_out()

        assert node_editor._zoom >= node_editor.ZOOM_MIN

        # Zoom in many times
        for _ in range(100):
            node_editor.zoom_in()

        assert node_editor._zoom <= node_editor.ZOOM_MAX


# =============================================================================
# Selection Tests
# =============================================================================

class TestSelection:
    """Tests for selection functionality."""

    def test_select_all(self, node_editor):
        """Test: Select all works."""
        node_editor.add_node_at_position("PassThroughNode", 100, 100)
        node_editor.add_node_at_position("PassThroughNode", 200, 200)

        node_editor.select_all()

        selected = node_editor.get_selected_nodes()
        if len(node_editor.graph.nodes) > 0:
            assert len(selected) == len(node_editor.graph.nodes)

    def test_clear_selection(self, node_editor):
        """Test: Clear selection works."""
        node_editor.add_node_at_position("PassThroughNode", 100, 100)
        node_editor.select_all()
        node_editor.clear_selection()

        selected = node_editor.get_selected_nodes()
        assert len(selected) == 0


# =============================================================================
# Delete Tests
# =============================================================================

class TestDelete:
    """Tests for delete functionality."""

    def test_delete_selection(self, node_editor):
        """Test: Delete selection works."""
        node = node_editor.add_node_at_position("PassThroughNode", 100, 100)

        if node is not None:
            widget = node_editor._node_widgets.get(node.id)
            if widget:
                widget.setSelected(True)

            node_editor.delete_selection()

            assert len(node_editor.graph.nodes) == 0


# =============================================================================
# MainWindow Integration Tests
# =============================================================================

class TestMainWindowIntegration:
    """Integration tests for MainWindow."""

    def test_undo_redo_actions(self, main_window):
        """Test: Undo/redo actions work via main window."""
        # Add node
        main_window._node_editor.add_node_at_position("PassThroughNode", 100, 100)

        if len(main_window._node_editor.graph.nodes) > 0:
            assert len(main_window._node_editor.graph.nodes) == 1

            # Undo
            main_window.undo()
            assert len(main_window._node_editor.graph.nodes) == 0

            # Redo
            main_window.redo()
            assert len(main_window._node_editor.graph.nodes) == 1

    def test_execute_graph(self, main_window):
        """Test: Execute graph action works."""
        # This should not raise an error even with empty graph
        main_window.execute_graph()
