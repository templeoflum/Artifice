"""
Node palette for selecting and creating nodes.

Displays available nodes organized by category with search functionality.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
)

from artifice.core.registry import get_registry


class NodePalette(QWidget):
    """
    Widget for displaying available nodes.

    Shows nodes organized by category with search/filter capability.
    Double-click or drag to create nodes.
    """

    node_requested = Signal(str)  # node_type

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()
        self._populate_nodes()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Search box
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes...")
        self._search.textChanged.connect(self._on_search_changed)
        layout.addWidget(self._search)

        # Tree view
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setDragEnabled(True)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree)

    def _populate_nodes(self) -> None:
        """Populate the tree with available nodes."""
        registry = get_registry()
        nodes_by_category: dict[str, list[tuple[str, str]]] = {}

        # Group nodes by category
        for name, node_class in registry.get_registry().items():
            category = getattr(node_class, "category", "Uncategorized")
            display_name = getattr(node_class, "name", name)
            description = getattr(node_class, "description", "")

            if category not in nodes_by_category:
                nodes_by_category[category] = []
            nodes_by_category[category].append((name, display_name, description))

        # Create tree items
        for category in sorted(nodes_by_category.keys()):
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(
                category_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
            )

            for node_name, display_name, description in sorted(
                nodes_by_category[category], key=lambda x: x[1]
            ):
                node_item = QTreeWidgetItem([display_name])
                node_item.setData(0, Qt.ItemDataRole.UserRole, node_name)
                node_item.setToolTip(0, description or display_name)
                category_item.addChild(node_item)

            self._tree.addTopLevelItem(category_item)

        # Expand all categories
        self._tree.expandAll()

    def _on_search_changed(self, text: str) -> None:
        """Filter nodes based on search text."""
        text = text.lower()

        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            category_visible = False

            for j in range(category_item.childCount()):
                node_item = category_item.child(j)
                node_name = node_item.text(0).lower()
                node_type = node_item.data(0, Qt.ItemDataRole.UserRole).lower()

                visible = not text or text in node_name or text in node_type
                node_item.setHidden(not visible)

                if visible:
                    category_visible = True

            category_item.setHidden(not category_visible)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click on a node item."""
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if node_type:
            self.node_requested.emit(node_type)

    def startDrag(self, supportedActions) -> None:
        """Start drag operation for creating nodes."""
        item = self._tree.currentItem()
        if not item:
            return

        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        if not node_type:
            return

        # Create drag data
        mime_data = QMimeData()
        mime_data.setData(
            "application/x-artifice-node",
            node_type.encode()
        )

        # Create drag
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)

    def refresh(self) -> None:
        """Refresh the node list."""
        self._tree.clear()
        self._populate_nodes()
