"""
Node graph and execution engine.

Manages collections of nodes, their connections, and handles
topological execution ordering with caching.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from artifice.core.port import InputPort, OutputPort, connect, disconnect

if TYPE_CHECKING:
    from artifice.core.node import Node


@dataclass
class Connection:
    """
    Represents a connection between two ports.

    Attributes:
        source_node_id: ID of the node with the output port
        source_port: Name of the output port
        dest_node_id: ID of the node with the input port
        dest_port: Name of the input port
    """

    source_node_id: str
    source_port: str
    dest_node_id: str
    dest_port: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {
            "source_node": self.source_node_id,
            "source_port": self.source_port,
            "dest_node": self.dest_node_id,
            "dest_port": self.dest_port,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Connection:
        """Deserialize from dictionary."""
        return cls(
            source_node_id=data["source_node"],
            source_port=data["source_port"],
            dest_node_id=data["dest_node"],
            dest_port=data["dest_port"],
        )


class NodeGraph:
    """
    A graph of connected nodes with execution capability.

    The graph manages node lifecycle, connections, and execution order.
    It performs topological sorting to ensure nodes execute in dependency
    order and caches results for unchanged nodes.

    Attributes:
        nodes: Dictionary of nodes by ID
        name: Optional name for the graph
        metadata: Optional metadata dictionary
    """

    def __init__(self, name: str = "Untitled") -> None:
        """Initialize an empty graph."""
        self.nodes: dict[str, Node] = {}
        self.name: str = name
        self.metadata: dict[str, Any] = {}
        self._execution_order: list[str] | None = None

    def add_node(self, node: Node) -> Node:
        """
        Add a node to the graph.

        Args:
            node: The node to add

        Returns:
            The added node (for chaining)
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with ID '{node.id}' already exists in graph")

        self.nodes[node.id] = node
        self._invalidate_order()
        return node

    def remove_node(self, node_or_id: Node | str) -> bool:
        """
        Remove a node from the graph.

        Disconnects all ports before removal.

        Args:
            node_or_id: Node instance or node ID

        Returns:
            True if removed, False if not found
        """
        node_id = node_or_id if isinstance(node_or_id, str) else node_or_id.id

        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        node.disconnect_all()
        del self.nodes[node_id]
        self._invalidate_order()
        return True

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def connect(
        self,
        source_node: Node | str,
        source_port: str,
        dest_node: Node | str,
        dest_port: str,
    ) -> bool:
        """
        Connect an output port to an input port.

        Args:
            source_node: Node with output (or its ID)
            source_port: Name of output port
            dest_node: Node with input (or its ID)
            dest_port: Name of input port

        Returns:
            True if connection was made, False if invalid
        """
        # Resolve node IDs
        src_id = source_node if isinstance(source_node, str) else source_node.id
        dst_id = dest_node if isinstance(dest_node, str) else dest_node.id

        src_node = self.nodes.get(src_id)
        dst_node = self.nodes.get(dst_id)

        if src_node is None or dst_node is None:
            return False

        # Get ports
        output = src_node.outputs.get(source_port)
        input_port = dst_node.inputs.get(dest_port)

        if output is None or input_port is None:
            return False

        # Check for cycles before connecting
        if self._would_create_cycle(src_node, dst_node):
            return False

        # Make connection
        if connect(output, input_port):
            dst_node.mark_dirty()
            self._invalidate_order()
            return True

        return False

    def disconnect(
        self,
        source_node: Node | str,
        source_port: str,
        dest_node: Node | str,
        dest_port: str,
    ) -> bool:
        """
        Disconnect an output port from an input port.

        Args:
            source_node: Node with output (or its ID)
            source_port: Name of output port
            dest_node: Node with input (or its ID)
            dest_port: Name of input port

        Returns:
            True if disconnection was made, False if not connected
        """
        # Resolve node IDs
        src_id = source_node if isinstance(source_node, str) else source_node.id
        dst_id = dest_node if isinstance(dest_node, str) else dest_node.id

        src_node = self.nodes.get(src_id)
        dst_node = self.nodes.get(dst_id)

        if src_node is None or dst_node is None:
            return False

        # Get ports
        output = src_node.outputs.get(source_port)
        input_port = dst_node.inputs.get(dest_port)

        if output is None or input_port is None:
            return False

        # Make disconnection
        if disconnect(output, input_port):
            dst_node.mark_dirty()
            self._invalidate_order()
            return True

        return False

    def get_connections(self) -> list[Connection]:
        """
        Get all connections in the graph.

        Returns:
            List of Connection objects
        """
        connections = []
        for node in self.nodes.values():
            for output_name, output_port in node.outputs.items():
                for input_port in output_port.connections:
                    if input_port.node is not None:
                        connections.append(
                            Connection(
                                source_node_id=node.id,
                                source_port=output_name,
                                dest_node_id=input_port.node.id,
                                dest_port=input_port.name,
                            )
                        )
        return connections

    def _would_create_cycle(self, source: Node, dest: Node) -> bool:
        """
        Check if connecting source->dest would create a cycle.

        Uses BFS from dest to see if we can reach source.
        """
        if source is dest:
            return True

        visited = set()
        queue = deque([dest])

        while queue:
            current = queue.popleft()
            if current.id in visited:
                continue
            visited.add(current.id)

            # Check all outputs of current node
            for output in current.outputs.values():
                for connected_input in output.connections:
                    if connected_input.node is not None:
                        if connected_input.node is source:
                            return True
                        queue.append(connected_input.node)

        return False

    def _invalidate_order(self) -> None:
        """Invalidate cached execution order."""
        self._execution_order = None

    def _compute_execution_order(self) -> list[str]:
        """
        Compute topological sort of nodes.

        Uses Kahn's algorithm to determine execution order.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If graph contains a cycle
        """
        # Count incoming edges for each node
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}

        for node in self.nodes.values():
            for output in node.outputs.values():
                for connected_input in output.connections:
                    if connected_input.node is not None:
                        in_degree[connected_input.node.id] += 1

        # Start with nodes that have no incoming edges
        queue = deque([nid for nid, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            node = self.nodes[node_id]
            for output in node.outputs.values():
                for connected_input in output.connections:
                    if connected_input.node is not None:
                        target_id = connected_input.node.id
                        in_degree[target_id] -= 1
                        if in_degree[target_id] == 0:
                            queue.append(target_id)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def get_execution_order(self) -> list[str]:
        """
        Get the execution order for nodes.

        Returns:
            List of node IDs in topological order
        """
        if self._execution_order is None:
            self._execution_order = self._compute_execution_order()
        return self._execution_order

    def execute(self, force: bool = False) -> dict[str, bool]:
        """
        Execute all nodes in the graph.

        Executes nodes in topological order. Nodes that are not dirty
        (and don't have dirty dependencies) are skipped unless force=True.

        Args:
            force: If True, execute all nodes regardless of dirty state

        Returns:
            Dictionary mapping node IDs to success status
        """
        results: dict[str, bool] = {}
        order = self.get_execution_order()

        for node_id in order:
            node = self.nodes[node_id]

            if force:
                node.mark_dirty()

            success = node.execute()
            results[node_id] = success

            if not success:
                # Mark downstream nodes as failed
                self._mark_downstream_failed(node, results)

        return results

    def _mark_downstream_failed(
        self, node: Node, results: dict[str, bool]
    ) -> None:
        """Mark all downstream nodes as failed."""
        for output in node.outputs.values():
            for connected_input in output.connections:
                if connected_input.node is not None:
                    downstream_id = connected_input.node.id
                    if downstream_id not in results:
                        results[downstream_id] = False
                        self._mark_downstream_failed(
                            connected_input.node, results
                        )

    def execute_to_node(self, node_or_id: Node | str) -> bool:
        """
        Execute nodes needed to produce output for a specific node.

        Args:
            node_or_id: Target node or its ID

        Returns:
            True if target node executed successfully
        """
        target_id = node_or_id if isinstance(node_or_id, str) else node_or_id.id

        if target_id not in self.nodes:
            return False

        # Find all upstream nodes
        upstream = self._get_upstream_nodes(target_id)
        order = self.get_execution_order()

        # Execute only upstream nodes in order
        for node_id in order:
            if node_id in upstream or node_id == target_id:
                node = self.nodes[node_id]
                if not node.execute():
                    return False

        return True

    def _get_upstream_nodes(self, node_id: str) -> set[str]:
        """Get all nodes upstream of a given node."""
        upstream = set()
        node = self.nodes.get(node_id)

        if node is None:
            return upstream

        for input_port in node.inputs.values():
            if input_port.connection is not None and input_port.connection.node is not None:
                upstream_id = input_port.connection.node.id
                upstream.add(upstream_id)
                upstream.update(self._get_upstream_nodes(upstream_id))

        return upstream

    def clear(self) -> None:
        """Remove all nodes from the graph."""
        for node in list(self.nodes.values()):
            node.disconnect_all()
        self.nodes.clear()
        self._invalidate_order()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize graph to dictionary.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.get_connections()],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        node_registry: dict[str, type] | None = None,
    ) -> NodeGraph:
        """
        Create graph from dictionary.

        Args:
            data: Serialized graph data
            node_registry: Mapping of node type names to classes

        Returns:
            New NodeGraph instance
        """
        from artifice.core.registry import NodeRegistry

        if node_registry is None:
            node_registry = NodeRegistry.get_registry()

        graph = cls(name=data.get("name", "Untitled"))
        graph.metadata = data.get("metadata", {})

        # Create nodes
        for node_data in data.get("nodes", []):
            node_type = node_data.get("type")
            if node_type not in node_registry:
                raise ValueError(f"Unknown node type: {node_type}")

            node_class = node_registry[node_type]
            node = node_class()
            node.id = node_data.get("id", node.id)
            node.position = tuple(node_data.get("position", [0.0, 0.0]))

            # Restore parameters
            for name, value in node_data.get("parameters", {}).items():
                if name in node.parameters:
                    node.parameters[name].set(value)

            graph.add_node(node)

        # Create connections
        for conn_data in data.get("connections", []):
            graph.connect(
                conn_data["source_node"],
                conn_data["source_port"],
                conn_data["dest_node"],
                conn_data["dest_port"],
            )

        return graph

    def save(self, path: str | Path) -> None:
        """
        Save graph to a JSON file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        node_registry: dict[str, type] | None = None,
    ) -> NodeGraph:
        """
        Load graph from a JSON file.

        Args:
            path: File path to load from
            node_registry: Mapping of node type names to classes

        Returns:
            Loaded NodeGraph instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data, node_registry)

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over nodes in execution order."""
        for node_id in self.get_execution_order():
            yield self.nodes[node_id]

    def __contains__(self, node_or_id: Node | str) -> bool:
        """Check if node is in graph."""
        node_id = node_or_id if isinstance(node_or_id, str) else node_or_id.id
        return node_id in self.nodes

    def __repr__(self) -> str:
        return f"NodeGraph(name={self.name!r}, nodes={len(self.nodes)})"
