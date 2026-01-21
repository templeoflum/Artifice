"""
Tests for port system.

Validates port creation, type checking, and connections.
"""

import pytest

from artifice.core.port import (
    InputPort,
    OutputPort,
    PortType,
    connect,
    disconnect,
    disconnect_all,
    types_compatible,
)


class TestPortTypes:
    """Tests for port type compatibility."""

    def test_same_type_compatible(self):
        """Test that same types are compatible."""
        assert types_compatible(PortType.IMAGE, PortType.IMAGE)
        assert types_compatible(PortType.NUMBER, PortType.NUMBER)
        assert types_compatible(PortType.STRING, PortType.STRING)

    def test_any_accepts_all(self):
        """Test that ANY type accepts any source."""
        assert types_compatible(PortType.IMAGE, PortType.ANY)
        assert types_compatible(PortType.NUMBER, PortType.ANY)
        assert types_compatible(PortType.STRING, PortType.ANY)

    def test_integer_to_number(self):
        """Test that INTEGER can connect to NUMBER."""
        assert types_compatible(PortType.INTEGER, PortType.NUMBER)

    def test_mask_to_image(self):
        """Test that MASK can connect to IMAGE."""
        assert types_compatible(PortType.MASK, PortType.IMAGE)

    def test_incompatible_types(self):
        """Test that incompatible types are rejected."""
        assert not types_compatible(PortType.IMAGE, PortType.NUMBER)
        assert not types_compatible(PortType.STRING, PortType.INTEGER)
        assert not types_compatible(PortType.BOOLEAN, PortType.IMAGE)


class TestInputPort:
    """Tests for InputPort class."""

    def test_creation(self):
        """Test input port creation."""
        port = InputPort(
            name="input1",
            port_type=PortType.IMAGE,
            description="Test input",
            default=None,
            required=True,
        )

        assert port.name == "input1"
        assert port.port_type == PortType.IMAGE
        assert port.required
        assert not port.is_connected

    def test_default_value(self):
        """Test input port default value."""
        port = InputPort(name="test", default=42.0)

        assert port.get_value() == 42.0

    def test_is_connected(self):
        """Test connection status tracking."""
        input_port = InputPort(name="in")
        output_port = OutputPort(name="out")

        assert not input_port.is_connected

        connect(output_port, input_port)
        assert input_port.is_connected

        disconnect(output_port, input_port)
        assert not input_port.is_connected


class TestOutputPort:
    """Tests for OutputPort class."""

    def test_creation(self):
        """Test output port creation."""
        port = OutputPort(
            name="output1",
            port_type=PortType.IMAGE,
            description="Test output",
        )

        assert port.name == "output1"
        assert port.port_type == PortType.IMAGE
        assert port.multi  # Outputs are always multi

    def test_value_caching(self):
        """Test output port value caching."""
        port = OutputPort(name="out")

        port.set_value(123)
        assert port.get_value() == 123
        assert port._cache_valid

        port.invalidate_cache()
        assert not port._cache_valid
        assert port.get_value() is None


class TestConnections:
    """Tests for port connection functions."""

    def test_basic_connection(self):
        """Test connecting compatible ports."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input_port = InputPort(name="in", port_type=PortType.IMAGE)

        result = connect(output, input_port)

        assert result
        assert input_port.connection is output
        assert input_port in output.connections

    def test_type_mismatch_rejected(self):
        """Test that incompatible types cannot connect."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input_port = InputPort(name="in", port_type=PortType.NUMBER)

        result = connect(output, input_port)

        assert not result
        assert not input_port.is_connected

    def test_multiple_outputs_to_inputs(self):
        """Test one output can connect to multiple inputs."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input1 = InputPort(name="in1", port_type=PortType.IMAGE)
        input2 = InputPort(name="in2", port_type=PortType.IMAGE)

        connect(output, input1)
        connect(output, input2)

        assert len(output.connections) == 2
        assert input1.is_connected
        assert input2.is_connected

    def test_single_input_replaces_connection(self):
        """Test non-multi input replaces existing connection."""
        output1 = OutputPort(name="out1", port_type=PortType.IMAGE)
        output2 = OutputPort(name="out2", port_type=PortType.IMAGE)
        input_port = InputPort(name="in", port_type=PortType.IMAGE, multi=False)

        connect(output1, input_port)
        assert input_port.connection is output1

        connect(output2, input_port)
        assert input_port.connection is output2
        assert input_port not in output1.connections

    def test_disconnect(self):
        """Test disconnecting ports."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input_port = InputPort(name="in", port_type=PortType.IMAGE)

        connect(output, input_port)
        result = disconnect(output, input_port)

        assert result
        assert not input_port.is_connected
        assert input_port not in output.connections

    def test_disconnect_not_connected(self):
        """Test disconnecting ports that aren't connected."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input_port = InputPort(name="in", port_type=PortType.IMAGE)

        result = disconnect(output, input_port)

        assert not result

    def test_disconnect_all_input(self):
        """Test disconnecting all from input port."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input_port = InputPort(name="in", port_type=PortType.IMAGE)

        connect(output, input_port)
        count = disconnect_all(input_port)

        assert count == 1
        assert not input_port.is_connected

    def test_disconnect_all_output(self):
        """Test disconnecting all from output port."""
        output = OutputPort(name="out", port_type=PortType.IMAGE)
        input1 = InputPort(name="in1", port_type=PortType.IMAGE)
        input2 = InputPort(name="in2", port_type=PortType.IMAGE)

        connect(output, input1)
        connect(output, input2)

        count = disconnect_all(output)

        assert count == 2
        assert not input1.is_connected
        assert not input2.is_connected
        assert len(output.connections) == 0

    def test_value_flows_through_connection(self):
        """Test that values flow from output to connected input."""
        output = OutputPort(name="out", port_type=PortType.NUMBER)
        input_port = InputPort(name="in", port_type=PortType.NUMBER, default=0)

        connect(output, input_port)
        output.set_value(42)

        assert input_port.get_value() == 42
