"""
Main application window.

The main window contains the node editor, preview panel, inspector,
and node palette arranged in a typical node-based application layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QMenuBar,
    QMenu,
    QToolBar,
    QFileDialog,
    QMessageBox,
    QApplication,
    QStatusBar,
)

from artifice.core.graph import NodeGraph
from artifice.ui.node_editor import NodeEditorWidget
from artifice.ui.preview import PreviewPanel
from artifice.ui.inspector import InspectorPanel
from artifice.ui.palette import NodePalette
from artifice.ui.undo import UndoStack

if TYPE_CHECKING:
    from artifice.core.node import Node


class MainWindow(QMainWindow):
    """
    Main application window for Artifice.

    Contains the node editor canvas as the central widget, with
    dockable panels for preview, inspector, and node palette.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Artifice")
        self.setMinimumSize(1200, 800)

        # Core components
        self._graph = NodeGraph()
        self._undo_stack = UndoStack()
        self._current_file: Path | None = None
        self._modified = False

        # Create UI components
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbars()
        self._setup_statusbar()
        self._setup_connections()

        # Load settings
        self._load_settings()

    @property
    def graph(self) -> NodeGraph:
        """Get the node graph."""
        return self._graph

    @property
    def node_editor(self) -> NodeEditorWidget:
        """Get the node editor widget."""
        return self._node_editor

    @property
    def preview(self) -> PreviewPanel:
        """Get the preview panel."""
        return self._preview

    @property
    def inspector(self) -> InspectorPanel:
        """Get the inspector panel."""
        return self._inspector

    @property
    def palette(self) -> NodePalette:
        """Get the node palette."""
        return self._palette

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Central widget - Node Editor
        self._node_editor = NodeEditorWidget(self._graph, self._undo_stack)
        self.setCentralWidget(self._node_editor)

        # Preview Panel (right dock)
        self._preview = PreviewPanel()
        self._preview_dock = QDockWidget("Preview", self)
        self._preview_dock.setWidget(self._preview)
        self._preview_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._preview_dock)

        # Inspector Panel (right dock, below preview)
        self._inspector = InspectorPanel()
        self._inspector_dock = QDockWidget("Inspector", self)
        self._inspector_dock.setWidget(self._inspector)
        self._inspector_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._inspector_dock)

        # Stack preview and inspector vertically
        self.splitDockWidget(
            self._preview_dock, self._inspector_dock, Qt.Orientation.Vertical
        )

        # Node Palette (left dock)
        self._palette = NodePalette()
        self._palette_dock = QDockWidget("Nodes", self)
        self._palette_dock.setWidget(self._palette)
        self._palette_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._palette_dock)

    def _setup_menus(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        self._action_new = QAction("&New", self)
        self._action_new.setShortcut(QKeySequence.StandardKey.New)
        self._action_new.triggered.connect(self.new_project)
        file_menu.addAction(self._action_new)

        self._action_open = QAction("&Open...", self)
        self._action_open.setShortcut(QKeySequence.StandardKey.Open)
        self._action_open.triggered.connect(self.open_project)
        file_menu.addAction(self._action_open)

        file_menu.addSeparator()

        self._action_save = QAction("&Save", self)
        self._action_save.setShortcut(QKeySequence.StandardKey.Save)
        self._action_save.triggered.connect(self.save_project)
        file_menu.addAction(self._action_save)

        self._action_save_as = QAction("Save &As...", self)
        self._action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._action_save_as.triggered.connect(self.save_project_as)
        file_menu.addAction(self._action_save_as)

        file_menu.addSeparator()

        self._action_exit = QAction("E&xit", self)
        self._action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self._action_exit.triggered.connect(self.close)
        file_menu.addAction(self._action_exit)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        self._action_undo = QAction("&Undo", self)
        self._action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._action_undo.triggered.connect(self.undo)
        edit_menu.addAction(self._action_undo)

        self._action_redo = QAction("&Redo", self)
        self._action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._action_redo.triggered.connect(self.redo)
        edit_menu.addAction(self._action_redo)

        edit_menu.addSeparator()

        self._action_cut = QAction("Cu&t", self)
        self._action_cut.setShortcut(QKeySequence.StandardKey.Cut)
        self._action_cut.triggered.connect(self._node_editor.cut_selection)
        edit_menu.addAction(self._action_cut)

        self._action_copy = QAction("&Copy", self)
        self._action_copy.setShortcut(QKeySequence.StandardKey.Copy)
        self._action_copy.triggered.connect(self._node_editor.copy_selection)
        edit_menu.addAction(self._action_copy)

        self._action_paste = QAction("&Paste", self)
        self._action_paste.setShortcut(QKeySequence.StandardKey.Paste)
        self._action_paste.triggered.connect(self._node_editor.paste)
        edit_menu.addAction(self._action_paste)

        self._action_delete = QAction("&Delete", self)
        self._action_delete.setShortcut(QKeySequence.StandardKey.Delete)
        self._action_delete.triggered.connect(self._node_editor.delete_selection)
        edit_menu.addAction(self._action_delete)

        edit_menu.addSeparator()

        self._action_select_all = QAction("Select &All", self)
        self._action_select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        self._action_select_all.triggered.connect(self._node_editor.select_all)
        edit_menu.addAction(self._action_select_all)

        # View menu
        view_menu = menubar.addMenu("&View")

        self._action_zoom_in = QAction("Zoom &In", self)
        self._action_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self._action_zoom_in.triggered.connect(self._node_editor.zoom_in)
        view_menu.addAction(self._action_zoom_in)

        self._action_zoom_out = QAction("Zoom &Out", self)
        self._action_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self._action_zoom_out.triggered.connect(self._node_editor.zoom_out)
        view_menu.addAction(self._action_zoom_out)

        self._action_zoom_reset = QAction("&Reset Zoom", self)
        self._action_zoom_reset.setShortcut(QKeySequence("Ctrl+0"))
        self._action_zoom_reset.triggered.connect(self._node_editor.zoom_reset)
        view_menu.addAction(self._action_zoom_reset)

        self._action_fit_view = QAction("&Fit to View", self)
        self._action_fit_view.setShortcut(QKeySequence("F"))
        self._action_fit_view.triggered.connect(self._node_editor.fit_to_view)
        view_menu.addAction(self._action_fit_view)

        view_menu.addSeparator()

        # Toggle dock widgets
        view_menu.addAction(self._preview_dock.toggleViewAction())
        view_menu.addAction(self._inspector_dock.toggleViewAction())
        view_menu.addAction(self._palette_dock.toggleViewAction())

        # Graph menu
        graph_menu = menubar.addMenu("&Graph")

        self._action_execute = QAction("&Execute", self)
        self._action_execute.setShortcut(QKeySequence("F5"))
        self._action_execute.triggered.connect(self.execute_graph)
        graph_menu.addAction(self._action_execute)

        self._action_clear = QAction("&Clear Graph", self)
        self._action_clear.triggered.connect(self._clear_graph)
        graph_menu.addAction(self._action_clear)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        self._action_about = QAction("&About", self)
        self._action_about.triggered.connect(self._show_about)
        help_menu.addAction(self._action_about)

    def _setup_toolbars(self) -> None:
        """Set up toolbars."""
        # Main toolbar
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self._action_new)
        toolbar.addAction(self._action_open)
        toolbar.addAction(self._action_save)
        toolbar.addSeparator()
        toolbar.addAction(self._action_undo)
        toolbar.addAction(self._action_redo)
        toolbar.addSeparator()
        toolbar.addAction(self._action_execute)

    def _setup_statusbar(self) -> None:
        """Set up the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _setup_connections(self) -> None:
        """Set up signal connections."""
        # Node selection -> Inspector
        self._node_editor.node_selected.connect(self._on_node_selected)
        self._node_editor.node_deselected.connect(self._on_node_deselected)

        # Palette -> Node Editor
        self._palette.node_requested.connect(self._on_node_requested)

        # Inspector parameter changes
        self._inspector.parameter_changed.connect(self._on_parameter_changed)

        # Graph changes
        self._node_editor.graph_modified.connect(self._on_graph_modified)

        # Undo stack changes
        self._undo_stack.can_undo_changed.connect(self._action_undo.setEnabled)
        self._undo_stack.can_redo_changed.connect(self._action_redo.setEnabled)

        # Initial state
        self._action_undo.setEnabled(False)
        self._action_redo.setEnabled(False)

    def _load_settings(self) -> None:
        """Load application settings."""
        settings = QSettings("ArtificeEngine", "Artifice")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

    def _save_settings(self) -> None:
        """Save application settings."""
        settings = QSettings("ArtificeEngine", "Artifice")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close."""
        if self._modified:
            result = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if result == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    event.ignore()
                    return
            elif result == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        self._save_settings()
        event.accept()

    # --- Slots ---

    def _on_node_selected(self, node: Node) -> None:
        """Handle node selection."""
        self._inspector.set_node(node)

    def _on_node_deselected(self) -> None:
        """Handle node deselection."""
        self._inspector.clear()

    def _on_node_requested(self, node_type: str) -> None:
        """Handle node creation request from palette."""
        self._node_editor.add_node_at_center(node_type)

    def _on_parameter_changed(self, node: Node, param_name: str, value) -> None:
        """Handle parameter change from inspector."""
        # The change is already applied; we just need to mark as modified
        # and potentially update the preview
        self._mark_modified()
        self._update_preview_if_needed(node)

    def _on_graph_modified(self) -> None:
        """Handle graph modification."""
        self._mark_modified()

    def _mark_modified(self) -> None:
        """Mark the project as modified."""
        if not self._modified:
            self._modified = True
            self._update_title()

    def _update_title(self) -> None:
        """Update the window title."""
        title = "Artifice"
        if self._current_file:
            title = f"{self._current_file.name} - {title}"
        if self._modified:
            title = f"*{title}"
        self.setWindowTitle(title)

    def _update_preview_if_needed(self, node: Node) -> None:
        """Update preview if the node affects it."""
        # For now, just update preview on any change
        # In the future, only update if the node is connected to preview
        pass

    # --- Public Methods ---

    def new_project(self) -> bool:
        """Create a new project."""
        if self._modified:
            result = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save first?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if result == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    return False
            elif result == QMessageBox.StandardButton.Cancel:
                return False

        self._graph.clear()
        self._node_editor.clear()
        self._undo_stack.clear()
        self._current_file = None
        self._modified = False
        self._update_title()
        self._statusbar.showMessage("New project created")
        return True

    def open_project(self) -> bool:
        """Open a project file."""
        if self._modified:
            result = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save first?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if result == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    return False
            elif result == QMessageBox.StandardButton.Cancel:
                return False

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "Artifice Projects (*.artifice);;All Files (*)",
        )
        if not file_path:
            return False

        return self.load_project(file_path)

    def load_project(self, file_path: str | Path) -> bool:
        """Load a project from file."""
        file_path = Path(file_path)
        try:
            self._graph = NodeGraph.load(file_path)
            self._node_editor.set_graph(self._graph)
            self._undo_stack.clear()
            self._current_file = file_path
            self._modified = False
            self._update_title()
            self._statusbar.showMessage(f"Loaded: {file_path.name}")
            return True
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Project",
                f"Failed to load project:\n{e}",
            )
            return False

    def save_project(self, file_path: str | Path | None = None) -> bool:
        """Save the current project."""
        if file_path:
            file_path = Path(file_path)
        elif self._current_file:
            file_path = self._current_file
        else:
            return self.save_project_as()

        try:
            self._graph.save(file_path)
            self._current_file = file_path
            self._modified = False
            self._update_title()
            self._statusbar.showMessage(f"Saved: {file_path.name}")
            return True
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Project",
                f"Failed to save project:\n{e}",
            )
            return False

    def save_project_as(self) -> bool:
        """Save the project to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "Artifice Projects (*.artifice);;All Files (*)",
        )
        if not file_path:
            return False

        if not file_path.endswith(".artifice"):
            file_path += ".artifice"

        return self.save_project(file_path)

    def execute_graph(self) -> None:
        """Execute the node graph."""
        try:
            self._statusbar.showMessage("Executing graph...")
            QApplication.processEvents()

            self._graph.execute()

            self._statusbar.showMessage("Execution complete")

            # Update preview with output
            self._update_preview_after_execution()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Execution Error",
                f"Error during graph execution:\n{e}",
            )
            self._statusbar.showMessage("Execution failed")

    def _update_preview_after_execution(self) -> None:
        """Update preview panel after graph execution."""
        # Get execution order and iterate in reverse to find the last node
        # with an image output (shows the end of the processing chain)
        execution_order = self._graph.get_execution_order()

        for node_id in reversed(execution_order):
            node = self._graph.nodes.get(node_id)
            if node is None:
                continue
            # Check if node has an image output with data
            for port_name, port in node.outputs.items():
                if port.port_type.name == "IMAGE":
                    value = port.get_value()
                    if value is not None:
                        self._preview.set_image(value)
                        return

    def _clear_graph(self) -> None:
        """Clear the graph."""
        if self._graph.nodes:
            result = QMessageBox.question(
                self,
                "Clear Graph",
                "Are you sure you want to clear the graph?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return

        self._graph.clear()
        self._node_editor.clear()
        self._mark_modified()
        self._statusbar.showMessage("Graph cleared")

    def undo(self) -> None:
        """Undo the last action."""
        self._undo_stack.undo()
        self._mark_modified()

    def redo(self) -> None:
        """Redo the last undone action."""
        self._undo_stack.redo()
        self._mark_modified()

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Artifice",
            "Artifice\n\n"
            "A node-based glitch art tool.\n\n"
            "Converse with Chaos, Sculpt Emergence.",
        )
