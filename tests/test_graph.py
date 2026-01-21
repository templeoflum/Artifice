"""
Tests for NodeGraph class.

Validates V1.3, V1.4, V1.5, V1.6 from DELIVERABLES.md
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.core.graph import Connection, NodeGraph
from artifice.core.node import Node
from artifice.core.port import PortType
from artifice.core.registry import register_node
from artifice.nodes.io.loader import ImageLoaderNode
from artifice.nodes.io.saver import ImageSaverNode
from artifice.nodes.utility.passthrough import PassThroughNode


class TestGraphConnections:
    """V1.3 - Tests for graph connection management."""

    def test_add_nodes(self):
        """Test adding nodes to graph."""
        graph = NodeGraph()
        node1 = graph.add_node(PassThroughNode())
        node2 = graph.add_node(PassThroughNode())

        assert len(graph.nodes) == 2
        assert node1.id in graph.nodes
        assert node2.id in graph.nodes

    def test_connect_nodes(self):
        """V1.3 - Test connecting nodes in graph."""
        graph = NodeGraph()
        loader = graph.add_node(ImageLoaderNode())
        passthrough = graph.add_node(PassThroughNode())

        result = graph.connect(loader, "image", passthrough, "image")

        assert result
        connections = graph.get_connections()
        assert len(connections) == 1
        assert connections[0].source_node_id == loader.id
        assert connections[0].dest_node_id == passthrough.id

    def test_connect_chain(self):
        """Test connecting multiple nodes in a chain."""
        graph = NodeGraph()
        loader = graph.add_node(ImageLoaderNode())
        pass1 = graph.add_node(PassThroughNode())
        pass2 = graph.add_node(PassThroughNode())
        saver = graph.add_node(ImageSaverNode())

        assert graph.connect(loader, "image", pass1, "image")
        assert graph.connect(pass1, "image", pass2, "image")
        assert graph.connect(pass2, "image", saver, "image")

        assert len(graph.get_connections()) == 3

    def test_connect_invalid_port(self):
        """Test connecting to non-existent port fails."""
        graph = NodeGraph()
        loader = graph.add_node(ImageLoaderNode())
        passthrough = graph.add_node(PassThroughNode())

        result = graph.connect(loader, "nonexistent", passthrough, "image")

        assert not result

    def test_disconnect_nodes(self):
        """Test disconnecting nodes."""
        graph = NodeGraph()
        loader = graph.add_node(ImageLoaderNode())
        passthrough = graph.add_node(PassThroughNode())

        graph.connect(loader, "image", passthrough, "image")
        result = graph.disconnect(loader, "image", passthrough, "image")

        assert result
        assert len(graph.get_connections()) == 0

    def test_remove_node_disconnects(self):
        """Test removing node disconnects all ports."""
        graph = NodeGraph()
        loader = graph.add_node(ImageLoaderNode())
        passthrough = graph.add_node(PassThroughNode())
        saver = graph.add_node(ImageSaverNode())

        graph.connect(loader, "image", passthrough, "image")
        graph.connect(passthrough, "image", saver, "image")

        graph.remove_node(passthrough)

        assert len(graph.nodes) == 2
        assert len(graph.get_connections()) == 0

    def test_cycle_detection(self):
        """Test that cycles are prevented."""
        graph = NodeGraph()
        node1 = graph.add_node(PassThroughNode())
        node2 = graph.add_node(PassThroughNode())
        node3 = graph.add_node(PassThroughNode())

        # Create chain: node1 -> node2 -> node3
        graph.connect(node1, "image", node2, "image")
        graph.connect(node2, "image", node3, "image")

        # Try to create cycle: node3 -> node1
        result = graph.connect(node3, "image", node1, "image")

        assert not result

    def test_self_loop_prevented(self):
        """Test that self-loops are prevented."""
        graph = NodeGraph()
        node = graph.add_node(PassThroughNode())

        result = graph.connect(node, "image", node, "image")

        assert not result


class TestGraphExecution:
    """V1.4 - Tests for graph execution."""

    def test_execution_order(self):
        """V1.4 - Test graph executes in correct topological order."""
        graph = NodeGraph()

        # Create nodes that track execution order
        @register_node
        class OrderTracker(Node):
            name = "Order Tracker"
            _abstract = False
            execution_order = []

            def define_ports(self):
                self.add_input("in", PortType.IMAGE, required=False)
                self.add_output("out", PortType.IMAGE)

            def process(self):
                OrderTracker.execution_order.append(self.id)
                self.set_output_value("out", self.get_input_value("in"))

        OrderTracker.execution_order = []

        node1 = graph.add_node(OrderTracker())
        node2 = graph.add_node(OrderTracker())
        node3 = graph.add_node(OrderTracker())

        # node1 -> node2 -> node3
        graph.connect(node1, "out", node2, "in")
        graph.connect(node2, "out", node3, "in")

        graph.execute(force=True)

        # Verify order
        assert OrderTracker.execution_order.index(node1.id) < OrderTracker.execution_order.index(node2.id)
        assert OrderTracker.execution_order.index(node2.id) < OrderTracker.execution_order.index(node3.id)

    def test_execute_with_image(self, sample_image_path, temp_dir):
        """V1.4 - Test executing graph with actual image."""
        graph = NodeGraph()

        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", str(sample_image_path))

        passthrough = graph.add_node(PassThroughNode())

        output_path = temp_dir / "output.png"
        saver = graph.add_node(ImageSaverNode())
        saver.set_parameter("path", str(output_path))

        graph.connect(loader, "image", passthrough, "image")
        graph.connect(passthrough, "image", saver, "image")

        results = graph.execute()

        assert all(results.values())
        assert output_path.exists()

    def test_execute_partial(self, sample_image_buffer):
        """Test executing only up to a specific node."""
        graph = NodeGraph()

        node1 = graph.add_node(PassThroughNode())
        node1.inputs["image"].default = sample_image_buffer

        node2 = graph.add_node(PassThroughNode())
        node3 = graph.add_node(PassThroughNode())

        graph.connect(node1, "image", node2, "image")
        graph.connect(node2, "image", node3, "image")

        # Execute only up to node2
        result = graph.execute_to_node(node2)

        assert result
        assert not node2.is_dirty()
        # node3 should still be dirty (not executed)
        assert node3.is_dirty()


class TestNodeCaching:
    """V1.6 - Tests for node caching."""

    def test_unchanged_nodes_skip_execution(self, sample_image_buffer):
        """V1.6 - Test that unchanged nodes don't re-execute."""
        graph = NodeGraph()

        @register_node
        class CountingNode(Node):
            name = "Counting Node"
            _abstract = False
            execution_count = 0

            def define_ports(self):
                self.add_input("in", PortType.IMAGE, required=False)
                self.add_output("out", PortType.IMAGE)

            def process(self):
                CountingNode.execution_count += 1
                val = self.get_input_value("in")
                self.set_output_value("out", val)

        CountingNode.execution_count = 0

        counter = graph.add_node(CountingNode())
        counter.inputs["in"].default = sample_image_buffer

        # First execution
        graph.execute()
        assert CountingNode.execution_count == 1

        # Second execution (no changes)
        graph.execute()
        assert CountingNode.execution_count == 1  # Should not re-execute

    def test_parameter_change_triggers_reexecution(self, sample_image_buffer):
        """Test that parameter change triggers re-execution."""
        graph = NodeGraph()

        @register_node
        class ParamCountingNode(Node):
            name = "Param Counting Node"
            _abstract = False
            execution_count = 0

            def define_ports(self):
                self.add_input("in", PortType.IMAGE, required=False)
                self.add_output("out", PortType.IMAGE)

            def define_parameters(self):
                self.add_parameter("value", default=1.0)

            def process(self):
                ParamCountingNode.execution_count += 1
                self.set_output_value("out", self.get_input_value("in"))

        ParamCountingNode.execution_count = 0

        counter = graph.add_node(ParamCountingNode())
        counter.inputs["in"].default = sample_image_buffer

        graph.execute()
        assert ParamCountingNode.execution_count == 1

        # Change parameter
        counter.set_parameter("value", 2.0)
        graph.execute()
        assert ParamCountingNode.execution_count == 2


class TestGraphSerialization:
    """V1.5 - Tests for graph serialization."""

    def test_graph_to_dict(self):
        """Test serializing graph to dictionary."""
        graph = NodeGraph(name="Test Graph")
        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", "/test/image.png")
        loader.position = (100.0, 200.0)

        saver = graph.add_node(ImageSaverNode())
        graph.connect(loader, "image", saver, "image")

        data = graph.to_dict()

        assert data["name"] == "Test Graph"
        assert len(data["nodes"]) == 2
        assert len(data["connections"]) == 1

    def test_graph_from_dict(self):
        """Test deserializing graph from dictionary."""
        graph = NodeGraph(name="Test Graph")
        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", "/test/image.png")
        saver = graph.add_node(ImageSaverNode())
        graph.connect(loader, "image", saver, "image")

        data = graph.to_dict()
        restored = NodeGraph.from_dict(data)

        assert restored.name == "Test Graph"
        assert len(restored.nodes) == 2
        assert len(restored.get_connections()) == 1

    def test_graph_save_load(self, temp_dir):
        """V1.5 - Test saving and loading graph to/from file."""
        graph = NodeGraph(name="Test Graph")
        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", "/test/image.png")
        saver = graph.add_node(ImageSaverNode())
        graph.connect(loader, "image", saver, "image")

        # Save
        path = temp_dir / "test_graph.artifice"
        graph.save(path)

        assert path.exists()

        # Load
        loaded = NodeGraph.load(path)

        assert loaded.name == "Test Graph"
        assert len(loaded.nodes) == 2
        assert len(loaded.get_connections()) == 1

        # Verify parameters preserved
        loader_node = next(n for n in loaded.nodes.values() if isinstance(n, ImageLoaderNode))
        assert loader_node.get_parameter("path") == "/test/image.png"

    def test_graph_file_is_valid_json(self, temp_dir):
        """Test that saved graph file is valid JSON."""
        graph = NodeGraph()
        graph.add_node(PassThroughNode())

        path = temp_dir / "test.artifice"
        graph.save(path)

        # Should parse without error
        with open(path) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "connections" in data


class TestConnection:
    """Tests for Connection dataclass."""

    def test_connection_to_dict(self):
        """Test connection serialization."""
        conn = Connection(
            source_node_id="abc123",
            source_port="image",
            dest_node_id="def456",
            dest_port="input",
        )

        data = conn.to_dict()

        assert data["source_node"] == "abc123"
        assert data["source_port"] == "image"
        assert data["dest_node"] == "def456"
        assert data["dest_port"] == "input"

    def test_connection_from_dict(self):
        """Test connection deserialization."""
        data = {
            "source_node": "abc123",
            "source_port": "image",
            "dest_node": "def456",
            "dest_port": "input",
        }

        conn = Connection.from_dict(data)

        assert conn.source_node_id == "abc123"
        assert conn.dest_port == "input"


class TestGraphUtilities:
    """Tests for graph utility methods."""

    def test_clear(self):
        """Test clearing all nodes from graph."""
        graph = NodeGraph()
        graph.add_node(PassThroughNode())
        graph.add_node(PassThroughNode())

        graph.clear()

        assert len(graph.nodes) == 0

    def test_iteration(self):
        """Test iterating over graph in execution order."""
        graph = NodeGraph()
        node1 = graph.add_node(PassThroughNode())
        node2 = graph.add_node(PassThroughNode())
        graph.connect(node1, "image", node2, "image")

        nodes = list(graph)

        assert len(nodes) == 2
        assert nodes[0] is node1
        assert nodes[1] is node2

    def test_contains(self):
        """Test checking if node is in graph."""
        graph = NodeGraph()
        node = graph.add_node(PassThroughNode())
        other = PassThroughNode()

        assert node in graph
        assert node.id in graph
        assert other not in graph
