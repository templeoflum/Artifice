"""
Artifice User Interface.

Qt-based node editor with preview and parameter controls.
"""

from artifice.ui.main_window import MainWindow
from artifice.ui.node_editor import NodeEditorWidget
from artifice.ui.node_widget import NodeWidget
from artifice.ui.connection import ConnectionItem
from artifice.ui.preview import PreviewPanel
from artifice.ui.inspector import InspectorPanel
from artifice.ui.palette import NodePalette

__all__ = [
    "MainWindow",
    "NodeEditorWidget",
    "NodeWidget",
    "ConnectionItem",
    "PreviewPanel",
    "InspectorPanel",
    "NodePalette",
]
