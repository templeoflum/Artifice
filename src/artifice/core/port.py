"""
Port system for node connections.

Defines typed input/output ports that enable type-safe connections
between nodes in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Type

if TYPE_CHECKING:
    from artifice.core.node import Node


class PortType(Enum):
    """Built-in port data types."""

    # Core types
    IMAGE = auto()  # ImageBuffer
    SEGMENTS = auto()  # SegmentList
    MASK = auto()  # Single-channel ImageBuffer (grayscale mask)

    # Primitive types
    NUMBER = auto()  # float or int
    INTEGER = auto()  # int only
    BOOLEAN = auto()  # bool
    STRING = auto()  # str

    # Collection types
    ARRAY = auto()  # numpy array (generic)
    LIST = auto()  # Python list

    # Special types
    ANY = auto()  # Accepts any type (use sparingly)
    TRIGGER = auto()  # Execution trigger (no data)


# Type compatibility rules: source -> allowed destinations
_TYPE_COMPATIBILITY: dict[PortType, set[PortType]] = {
    PortType.IMAGE: {PortType.IMAGE, PortType.ARRAY, PortType.ANY},
    PortType.SEGMENTS: {PortType.SEGMENTS, PortType.ANY},
    PortType.MASK: {PortType.MASK, PortType.IMAGE, PortType.ARRAY, PortType.ANY},
    PortType.NUMBER: {PortType.NUMBER, PortType.ANY},
    PortType.INTEGER: {PortType.INTEGER, PortType.NUMBER, PortType.ANY},
    PortType.BOOLEAN: {PortType.BOOLEAN, PortType.ANY},
    PortType.STRING: {PortType.STRING, PortType.ANY},
    PortType.ARRAY: {PortType.ARRAY, PortType.ANY},
    PortType.LIST: {PortType.LIST, PortType.ANY},
    PortType.ANY: {PortType.ANY},  # ANY only connects to ANY
    PortType.TRIGGER: {PortType.TRIGGER},
}


def types_compatible(source_type: PortType, dest_type: PortType) -> bool:
    """Check if source port type can connect to destination port type."""
    if dest_type == PortType.ANY:
        return True
    return dest_type in _TYPE_COMPATIBILITY.get(source_type, set())


# Color coding for port types (for UI)
PORT_COLORS: dict[PortType, str] = {
    PortType.IMAGE: "#4A90D9",  # Blue
    PortType.SEGMENTS: "#D9A54A",  # Orange
    PortType.MASK: "#7B68EE",  # Purple
    PortType.NUMBER: "#50C878",  # Green
    PortType.INTEGER: "#3CB371",  # Medium sea green
    PortType.BOOLEAN: "#FF6B6B",  # Red
    PortType.STRING: "#FFD93D",  # Yellow
    PortType.ARRAY: "#20B2AA",  # Light sea green
    PortType.LIST: "#DDA0DD",  # Plum
    PortType.ANY: "#808080",  # Gray
    PortType.TRIGGER: "#FFFFFF",  # White
}


@dataclass
class Port:
    """
    Base class for node ports.

    Ports are connection points on nodes. Each port has a name, type,
    and optional description. Ports track their connections and the
    node they belong to.

    Attributes:
        name: Unique identifier within the node
        port_type: Data type this port accepts/produces
        description: Human-readable description
        node: Reference to owning node (set when added to node)
        multi: Whether multiple connections are allowed
    """

    name: str
    port_type: PortType = PortType.ANY
    description: str = ""
    node: Node | None = field(default=None, repr=False)
    multi: bool = False

    @property
    def color(self) -> str:
        """Get the UI color for this port's type."""
        return PORT_COLORS.get(self.port_type, "#808080")

    @property
    def full_name(self) -> str:
        """Get fully qualified name (node.port)."""
        if self.node:
            return f"{self.node.id}.{self.name}"
        return self.name


@dataclass
class InputPort(Port):
    """
    Input port that receives data from connected output ports.

    Attributes:
        default: Default value if not connected
        required: Whether a connection is required for execution
        connection: The connected output port (None if disconnected)
        validator: Optional function to validate incoming data
    """

    default: Any = None
    required: bool = True
    connection: OutputPort | None = field(default=None, repr=False)
    validator: Callable[[Any], bool] | None = field(default=None, repr=False)

    @property
    def is_connected(self) -> bool:
        """Check if this input has a connection."""
        return self.connection is not None

    def get_value(self) -> Any:
        """
        Get the current value from connection or default.

        Returns:
            Value from connected output, or default if not connected.
        """
        if self.connection is not None:
            return self.connection.get_value()
        return self.default

    def validate(self, value: Any) -> bool:
        """
        Validate a value for this port.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        if self.validator is not None:
            return self.validator(value)
        return True

    def can_connect(self, output: OutputPort, allow_replace: bool = True) -> bool:
        """
        Check if an output port can connect to this input.

        Args:
            output: The output port to check
            allow_replace: If True, allows replacing existing connection

        Returns:
            True if connection is valid
        """
        # Check type compatibility
        if not types_compatible(output.port_type, self.port_type):
            return False

        # Check if already connected (and not multi and not allowing replace)
        if self.is_connected and not self.multi and not allow_replace:
            return False

        # Prevent self-loops
        if output.node is not None and output.node is self.node:
            return False

        return True


@dataclass
class OutputPort(Port):
    """
    Output port that sends data to connected input ports.

    Attributes:
        connections: List of connected input ports
        _cached_value: Cached output value from last execution
        _cache_valid: Whether the cache is valid
    """

    connections: list[InputPort] = field(default_factory=list, repr=False)
    _cached_value: Any = field(default=None, repr=False)
    _cache_valid: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize with multi=True for outputs (can connect to many inputs)."""
        self.multi = True

    @property
    def is_connected(self) -> bool:
        """Check if this output has any connections."""
        return len(self.connections) > 0

    def get_value(self) -> Any:
        """
        Get the cached output value.

        Returns:
            Cached value from last execution, or None if not cached.
        """
        return self._cached_value if self._cache_valid else None

    def set_value(self, value: Any) -> None:
        """
        Set the output value and update cache.

        Args:
            value: Value to cache
        """
        self._cached_value = value
        self._cache_valid = True

    def invalidate_cache(self) -> None:
        """Mark the cache as invalid (needs re-execution)."""
        self._cache_valid = False

    def can_connect(self, input_port: InputPort) -> bool:
        """
        Check if this output can connect to an input port.

        Args:
            input_port: The input port to check

        Returns:
            True if connection is valid
        """
        return input_port.can_connect(self)


def connect(output: OutputPort, input_port: InputPort) -> bool:
    """
    Connect an output port to an input port.

    Args:
        output: Source output port
        input_port: Destination input port

    Returns:
        True if connection was made, False if invalid
    """
    if not output.can_connect(input_port):
        return False

    # Disconnect existing connection on input (if not multi)
    if input_port.is_connected and not input_port.multi:
        disconnect(input_port.connection, input_port)

    # Make the connection
    output.connections.append(input_port)
    input_port.connection = output

    return True


def disconnect(output: OutputPort, input_port: InputPort) -> bool:
    """
    Disconnect an output port from an input port.

    Args:
        output: Source output port
        input_port: Destination input port

    Returns:
        True if disconnection was made, False if not connected
    """
    if input_port.connection is not output:
        return False

    if input_port not in output.connections:
        return False

    output.connections.remove(input_port)
    input_port.connection = None

    return True


def disconnect_all(port: Port) -> int:
    """
    Disconnect all connections from a port.

    Args:
        port: The port to disconnect

    Returns:
        Number of connections removed
    """
    count = 0

    if isinstance(port, InputPort):
        if port.connection is not None:
            disconnect(port.connection, port)
            count = 1

    elif isinstance(port, OutputPort):
        # Copy list since we're modifying it
        for input_port in list(port.connections):
            disconnect(port, input_port)
            count += 1

    return count
