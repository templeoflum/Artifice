"""
Tests for Node base class and basic node implementations.

Validates V1.1 - Node Creation from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, Parameter, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import NodeRegistry, register_node
from artifice.nodes.io.loader import ImageLoaderNode
from artifice.nodes.io.saver import ImageSaverNode
from artifice.nodes.utility.passthrough import PassThroughNode


class TestNodeCreation:
    """Tests for basic node creation."""

    def test_passthrough_node_creation(self):
        """V1.1 - Test creating a node with typed ports."""
        node = PassThroughNode()

        assert node.name == "Pass Through"
        assert "image" in node.inputs
        assert "image" in node.outputs
        assert node.inputs["image"].port_type == PortType.IMAGE

    def test_loader_node_creation(self):
        """Test ImageLoaderNode creation."""
        node = ImageLoaderNode()

        assert node.name == "Image Loader"
        assert node.category == "I/O"
        assert "image" in node.outputs
        assert "path" in node.parameters

    def test_saver_node_creation(self):
        """Test ImageSaverNode creation."""
        node = ImageSaverNode()

        assert node.name == "Image Saver"
        assert "image" in node.inputs
        assert "path" in node.parameters
        assert "quality" in node.parameters

    def test_node_unique_ids(self):
        """Test that each node gets a unique ID."""
        node1 = PassThroughNode()
        node2 = PassThroughNode()

        assert node1.id != node2.id

    def test_node_position(self):
        """Test node position attribute."""
        node = PassThroughNode()

        assert node.position == (0.0, 0.0)

        node.position = (100.0, 200.0)
        assert node.position == (100.0, 200.0)


class TestNodeParameters:
    """Tests for node parameter system."""

    def test_parameter_creation(self):
        """Test parameter creation and defaults."""
        node = ImageSaverNode()
        quality = node.parameters["quality"]

        assert quality.param_type == ParameterType.INT
        assert quality.default == 95
        assert quality.value == 95
        assert quality.min_value == 1
        assert quality.max_value == 100

    def test_parameter_set(self):
        """Test setting parameter values."""
        node = ImageSaverNode()

        assert node.set_parameter("quality", 80)
        assert node.get_parameter("quality") == 80

    def test_parameter_validation_range(self):
        """Test parameter range validation."""
        node = ImageSaverNode()

        # Value should be clamped to range
        node.set_parameter("quality", 150)
        assert node.get_parameter("quality") == 100

        node.set_parameter("quality", -10)
        assert node.get_parameter("quality") == 1

    def test_parameter_type_coercion(self):
        """Test parameter type coercion."""
        node = ImageSaverNode()

        # Float should be converted to int
        node.set_parameter("quality", 75.5)
        assert node.get_parameter("quality") == 75
        assert isinstance(node.get_parameter("quality"), int)

    def test_parameter_marks_dirty(self):
        """Test that changing parameter marks node dirty."""
        node = PassThroughNode()
        node._dirty = False

        node.add_parameter("test", param_type=ParameterType.FLOAT, default=1.0)
        node.set_parameter("test", 2.0)

        assert node.is_dirty()


class TestNodeExecution:
    """Tests for node execution."""

    def test_can_execute_missing_required(self):
        """Test can_execute with missing required input."""
        node = PassThroughNode()

        can_exec, reason = node.can_execute()

        assert not can_exec
        assert "image" in reason

    def test_can_execute_with_connection(self, sample_image_buffer):
        """Test can_execute with connected input."""
        node = PassThroughNode()

        # Simulate connection with value
        node.inputs["image"].default = sample_image_buffer

        can_exec, _ = node.can_execute()
        assert can_exec

    def test_execute_success(self, sample_image_buffer):
        """Test successful execution."""
        node = PassThroughNode()
        node.inputs["image"].default = sample_image_buffer

        result = node.execute()

        assert result
        assert not node.is_dirty()
        assert node._error is None

    def test_execute_failure(self):
        """Test execution failure."""
        node = PassThroughNode()
        # No input connected

        result = node.execute()

        assert not result
        assert node._error is not None

    def test_dirty_propagation(self, sample_image_buffer):
        """Test that marking dirty propagates to downstream nodes."""
        from artifice.core.port import connect

        node1 = PassThroughNode()
        node2 = PassThroughNode()

        connect(node1.outputs["image"], node2.inputs["image"])

        node2._dirty = False
        node1.mark_dirty()

        assert node2.is_dirty()


class TestNodeSerialization:
    """Tests for node serialization."""

    def test_node_to_dict(self):
        """Test serializing node to dictionary."""
        node = ImageSaverNode()
        node.position = (100.0, 200.0)
        node.set_parameter("quality", 80)

        data = node.to_dict()

        assert data["type"] == "ImageSaverNode"
        assert data["id"] == node.id
        assert data["position"] == [100.0, 200.0]
        assert data["parameters"]["quality"] == 80

    def test_node_from_dict(self):
        """Test deserializing node from dictionary."""
        original = ImageSaverNode()
        original.position = (100.0, 200.0)
        original.set_parameter("quality", 80)

        data = original.to_dict()
        restored = Node.from_dict(data, ImageSaverNode)

        assert restored.id == original.id
        assert restored.position == (100.0, 200.0)
        assert restored.get_parameter("quality") == 80


class TestNodeRegistry:
    """Tests for node registry."""

    def test_nodes_registered(self):
        """Test that nodes are registered."""
        assert NodeRegistry.get("ImageLoaderNode") is not None
        assert NodeRegistry.get("ImageSaverNode") is not None
        assert NodeRegistry.get("PassThroughNode") is not None

    def test_create_node(self):
        """Test creating node through registry."""
        node = NodeRegistry.create("PassThroughNode")

        assert node is not None
        assert isinstance(node, PassThroughNode)

    def test_get_categories(self):
        """Test getting nodes by category."""
        categories = NodeRegistry.get_categories()

        assert "I/O" in categories
        assert "Utility" in categories

    def test_get_node_info(self):
        """Test getting node metadata."""
        info = NodeRegistry.get_node_info("ImageLoaderNode")

        assert info is not None
        assert info["name"] == "Image Loader"
        assert info["category"] == "I/O"

    def test_custom_node_registration(self):
        """Test registering a custom node."""

        @register_node
        class TestNode(Node):
            name = "Test Node"
            category = "Test"
            _abstract = False

            def define_ports(self):
                self.add_input("in", PortType.NUMBER)
                self.add_output("out", PortType.NUMBER)

            def process(self):
                self.set_output_value("out", self.get_input_value("in"))

        assert NodeRegistry.get("TestNode") is TestNode

        # Cleanup
        NodeRegistry.unregister("TestNode")
