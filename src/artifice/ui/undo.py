"""
Undo/Redo system for the node editor.

Provides a command-based undo stack for all graph modifications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from artifice.core.graph import NodeGraph
    from artifice.core.node import Node


class Command(ABC):
    """Abstract base for undoable commands."""

    @abstractmethod
    def redo(self) -> None:
        """Execute or re-execute the command."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Reverse the command."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the command."""
        return self.__class__.__name__


class AddNodeCommand(Command):
    """Command to add a node to the graph."""

    def __init__(
        self,
        graph: NodeGraph,
        node: Node,
        position: tuple[float, float] = (0, 0),
    ):
        self._graph = graph
        self._node = node
        self._position = position
        self._node_id = node.id

    def redo(self) -> None:
        """Add the node."""
        self._graph.add_node(self._node)
        self._node.position = self._position

    def undo(self) -> None:
        """Remove the node."""
        self._graph.remove_node(self._node)

    @property
    def description(self) -> str:
        return f"Add {self._node.name}"


class RemoveNodeCommand(Command):
    """Command to remove a node from the graph."""

    def __init__(self, graph: NodeGraph, node: Node):
        self._graph = graph
        self._node = node
        self._position = getattr(node, "position", (0, 0))
        # Store connections that will be removed
        self._connections: list[tuple[str, str, str, str]] = []
        for conn in list(graph.get_connections()):
            if conn.source_node_id == node.id or conn.dest_node_id == node.id:
                self._connections.append((
                    conn.source_node_id,
                    conn.source_port,
                    conn.dest_node_id,
                    conn.dest_port,
                ))

    def redo(self) -> None:
        """Remove the node."""
        self._graph.remove_node(self._node)

    def undo(self) -> None:
        """Restore the node and its connections."""
        self._graph.add_node(self._node)
        self._node.position = self._position

        # Restore connections
        for src_id, src_port, tgt_id, tgt_port in self._connections:
            src_node = self._graph.get_node(src_id)
            tgt_node = self._graph.get_node(tgt_id)
            if src_node and tgt_node:
                self._graph.connect(src_node, src_port, tgt_node, tgt_port)

    @property
    def description(self) -> str:
        return f"Remove {self._node.name}"


class ConnectCommand(Command):
    """Command to connect two nodes."""

    def __init__(
        self,
        graph: NodeGraph,
        source_node: Node,
        source_port: str,
        target_node: Node,
        target_port: str,
    ):
        self._graph = graph
        self._source_id = source_node.id
        self._source_port = source_port
        self._target_id = target_node.id
        self._target_port = target_port

    def redo(self) -> None:
        """Create the connection."""
        source = self._graph.get_node(self._source_id)
        target = self._graph.get_node(self._target_id)
        if source and target:
            self._graph.connect(source, self._source_port, target, self._target_port)

    def undo(self) -> None:
        """Remove the connection."""
        source = self._graph.get_node(self._source_id)
        target = self._graph.get_node(self._target_id)
        if source and target:
            self._graph.disconnect(source, self._source_port, target, self._target_port)

    @property
    def description(self) -> str:
        return "Connect"


class DisconnectCommand(Command):
    """Command to disconnect two nodes."""

    def __init__(
        self,
        graph: NodeGraph,
        source_node: Node,
        source_port: str,
        target_node: Node,
        target_port: str,
    ):
        self._graph = graph
        self._source_id = source_node.id
        self._source_port = source_port
        self._target_id = target_node.id
        self._target_port = target_port

    def redo(self) -> None:
        """Remove the connection."""
        source = self._graph.get_node(self._source_id)
        target = self._graph.get_node(self._target_id)
        if source and target:
            self._graph.disconnect(source, self._source_port, target, self._target_port)

    def undo(self) -> None:
        """Restore the connection."""
        source = self._graph.get_node(self._source_id)
        target = self._graph.get_node(self._target_id)
        if source and target:
            self._graph.connect(source, self._source_port, target, self._target_port)

    @property
    def description(self) -> str:
        return "Disconnect"


class MoveNodeCommand(Command):
    """Command to move a node."""

    def __init__(self, node: Node, old_pos: tuple[float, float], new_pos: tuple[float, float]):
        self._node = node
        self._old_pos = old_pos
        self._new_pos = new_pos

    def redo(self) -> None:
        """Move to new position."""
        self._node.position = self._new_pos

    def undo(self) -> None:
        """Move to old position."""
        self._node.position = self._old_pos

    @property
    def description(self) -> str:
        return f"Move {self._node.name}"


class ChangeParameterCommand(Command):
    """Command to change a node parameter."""

    def __init__(self, node: Node, param_name: str, old_value: Any, new_value: Any):
        self._node = node
        self._param_name = param_name
        self._old_value = old_value
        self._new_value = new_value

    def redo(self) -> None:
        """Set to new value."""
        self._node.set_parameter(self._param_name, self._new_value)

    def undo(self) -> None:
        """Set to old value."""
        self._node.set_parameter(self._param_name, self._old_value)

    @property
    def description(self) -> str:
        return f"Change {self._param_name}"


class CompositeCommand(Command):
    """Command that groups multiple commands."""

    def __init__(self, commands: list[Command], description: str = "Multiple"):
        self._commands = commands
        self._description = description

    def redo(self) -> None:
        """Execute all commands."""
        for cmd in self._commands:
            cmd.redo()

    def undo(self) -> None:
        """Undo all commands in reverse order."""
        for cmd in reversed(self._commands):
            cmd.undo()

    @property
    def description(self) -> str:
        return self._description


class UndoStack(QObject):
    """
    Manages undo/redo operations.

    Maintains a stack of commands that can be undone and redone.
    """

    can_undo_changed = Signal(bool)
    can_redo_changed = Signal(bool)
    command_executed = Signal(str)  # Emits command description

    def __init__(self, max_size: int = 100):
        super().__init__()
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._max_size = max_size

    def push(self, command: Command) -> None:
        """
        Push and execute a command.

        This clears the redo stack.
        """
        command.redo()

        self._undo_stack.append(command)
        self._redo_stack.clear()

        # Limit stack size
        while len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)

        self._emit_state_changes()
        self.command_executed.emit(command.description)

    def undo(self) -> bool:
        """Undo the last command."""
        if not self._undo_stack:
            return False

        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)

        self._emit_state_changes()
        return True

    def redo(self) -> bool:
        """Redo the last undone command."""
        if not self._redo_stack:
            return False

        command = self._redo_stack.pop()
        command.redo()
        self._undo_stack.append(command)

        self._emit_state_changes()
        return True

    def clear(self) -> None:
        """Clear both stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_state_changes()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return bool(self._redo_stack)

    def undo_text(self) -> str:
        """Get description of command that would be undone."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""

    def redo_text(self) -> str:
        """Get description of command that would be redone."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return ""

    def _emit_state_changes(self) -> None:
        """Emit signals for state changes."""
        self.can_undo_changed.emit(self.can_undo())
        self.can_redo_changed.emit(self.can_redo())
