"""
Base Node class and parameter system.

Nodes are the fundamental processing units in Artifice Engine.
Each node has typed input/output ports and parameters.
"""

from __future__ import annotations

import uuid
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Type

from artifice.core.port import InputPort, OutputPort, PortType, disconnect_all


class ParameterType(Enum):
    """Types of node parameters."""

    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    ENUM = auto()  # Choice from list
    COLOR = auto()  # RGB/RGBA color
    CURVE = auto()  # Spline curve data
    FILEPATH = auto()  # File path


@dataclass
class Parameter:
    """
    A configurable parameter on a node.

    Parameters are values that affect node behavior but aren't
    received through ports. They can be edited in the UI and
    serialized with the node.

    Attributes:
        name: Unique identifier within the node
        param_type: Data type of the parameter
        default: Default value
        value: Current value
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)
        step: Step size for UI sliders
        choices: Valid choices for ENUM type
        description: Human-readable description
        ui_hidden: Whether to hide in UI
        on_change: Callback when value changes
        file_filter: File filter string for FILEPATH type (e.g., "Images (*.png *.jpg)")
        is_save_path: If True, show save dialog; if False, show open dialog (for FILEPATH)
        default_directory: Default directory to open file dialogs in (for FILEPATH)
    """

    name: str
    param_type: ParameterType = ParameterType.FLOAT
    default: Any = None
    value: Any = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    choices: list[str] | None = None
    description: str = ""
    ui_hidden: bool = False
    on_change: Callable[[Any], None] | None = field(default=None, repr=False)
    file_filter: str | None = None
    is_save_path: bool = False
    default_directory: str | None = None

    def __post_init__(self) -> None:
        """Initialize value from default if not set."""
        if self.value is None:
            self.value = self.default

    def set(self, value: Any) -> bool:
        """
        Set the parameter value with validation.

        Args:
            value: New value to set

        Returns:
            True if value was set, False if invalid
        """
        # Type coercion
        if self.param_type == ParameterType.INT:
            try:
                value = int(value)
            except (ValueError, TypeError):
                return False
        elif self.param_type == ParameterType.FLOAT:
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False
        elif self.param_type == ParameterType.BOOL:
            value = bool(value)
        elif self.param_type == ParameterType.STRING:
            value = str(value)
        elif self.param_type == ParameterType.ENUM:
            if self.choices and value not in self.choices:
                return False

        # Range validation
        if self.min_value is not None and value < self.min_value:
            value = self.min_value
        if self.max_value is not None and value > self.max_value:
            value = self.max_value

        old_value = self.value
        self.value = value

        # Trigger callback
        if self.on_change is not None and old_value != value:
            self.on_change(value)

        return True

    def reset(self) -> None:
        """Reset to default value."""
        self.set(self.default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "param_type": self.param_type.name,
            "value": self.value,
            "default": self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "choices": self.choices,
            "description": self.description,
            "file_filter": self.file_filter,
            "is_save_path": self.is_save_path,
            "default_directory": self.default_directory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Parameter:
        """Deserialize from dictionary."""
        param_type = ParameterType[data.get("param_type", "FLOAT")]
        return cls(
            name=data["name"],
            param_type=param_type,
            default=data.get("default"),
            value=data.get("value"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            step=data.get("step"),
            choices=data.get("choices"),
            description=data.get("description", ""),
            file_filter=data.get("file_filter"),
            is_save_path=data.get("is_save_path", False),
            default_directory=data.get("default_directory"),
        )


class NodeMeta(ABCMeta):
    """
    Metaclass for nodes that handles registration.

    Inherits from ABCMeta to support abstract methods.
    Automatically registers node classes with the global registry
    when they are defined (unless they are abstract base classes).
    """

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> NodeMeta:
        cls = super().__new__(mcs, name, bases, namespace)

        # Don't register abstract base classes
        if not namespace.get("_abstract", False) and name != "Node":
            # Defer registration until registry is available
            if hasattr(cls, "_deferred_registration"):
                cls._deferred_registration.append(cls)
            else:
                cls._deferred_registration = [cls]

        return cls


class Node(metaclass=NodeMeta):
    """
    Abstract base class for all processing nodes.

    Subclasses must implement:
        - define_ports(): Set up input/output ports
        - process(): Perform the node's computation

    Class Attributes:
        name: Display name for UI
        category: Category for organization (e.g., "I/O", "Color")
        description: Human-readable description
        icon: Optional icon identifier

    Instance Attributes:
        id: Unique identifier for this node instance
        inputs: Dict of input ports by name
        outputs: Dict of output ports by name
        parameters: Dict of parameters by name
        position: (x, y) position in node editor
        _dirty: Whether node needs re-execution
    """

    # Class-level metadata (override in subclasses)
    name: ClassVar[str] = "Node"
    category: ClassVar[str] = "Utility"
    description: ClassVar[str] = ""
    icon: ClassVar[str | None] = None
    _abstract: ClassVar[bool] = True  # Don't register base class

    def __init__(self) -> None:
        """Initialize the node with a unique ID."""
        self.id: str = str(uuid.uuid4())[:8]
        self.inputs: dict[str, InputPort] = {}
        self.outputs: dict[str, OutputPort] = {}
        self.parameters: dict[str, Parameter] = {}
        self.position: tuple[float, float] = (0.0, 0.0)
        self._dirty: bool = True
        self._error: str | None = None

        # Set up ports and parameters
        self.define_ports()
        self.define_parameters()

        # Set node reference on all ports
        for port in self.inputs.values():
            port.node = self
        for port in self.outputs.values():
            port.node = self

    @abstractmethod
    def define_ports(self) -> None:
        """
        Define input and output ports.

        Override this method to add ports using add_input() and add_output().
        """
        pass

    def define_parameters(self) -> None:
        """
        Define node parameters.

        Override this method to add parameters using add_parameter().
        Default implementation does nothing.
        """
        pass

    def add_input(
        self,
        name: str,
        port_type: PortType = PortType.IMAGE,
        description: str = "",
        default: Any = None,
        required: bool = True,
    ) -> InputPort:
        """
        Add an input port to this node.

        Args:
            name: Unique port name
            port_type: Data type for the port
            description: Human-readable description
            default: Default value if not connected
            required: Whether connection is required

        Returns:
            The created InputPort
        """
        port = InputPort(
            name=name,
            port_type=port_type,
            description=description,
            default=default,
            required=required,
            node=self,
        )
        self.inputs[name] = port
        return port

    def add_output(
        self,
        name: str,
        port_type: PortType = PortType.IMAGE,
        description: str = "",
    ) -> OutputPort:
        """
        Add an output port to this node.

        Args:
            name: Unique port name
            port_type: Data type for the port
            description: Human-readable description

        Returns:
            The created OutputPort
        """
        port = OutputPort(
            name=name,
            port_type=port_type,
            description=description,
            node=self,
        )
        self.outputs[name] = port
        return port

    def add_parameter(
        self,
        name: str,
        param_type: ParameterType = ParameterType.FLOAT,
        default: Any = None,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        step: float | int | None = None,
        choices: list[str] | None = None,
        description: str = "",
        ui_hidden: bool = False,
        file_filter: str | None = None,
        is_save_path: bool = False,
        default_directory: str | None = None,
    ) -> Parameter:
        """
        Add a parameter to this node.

        Args:
            name: Unique parameter name
            param_type: Data type
            default: Default value
            min_value: Minimum (for numeric)
            max_value: Maximum (for numeric)
            step: UI step size
            choices: Valid choices (for ENUM)
            description: Human-readable description
            ui_hidden: Hide from UI
            file_filter: File filter for FILEPATH type (e.g., "Images (*.png *.jpg)")
            is_save_path: For FILEPATH, True = save dialog, False = open dialog
            default_directory: Default directory for file dialogs

        Returns:
            The created Parameter
        """
        param = Parameter(
            name=name,
            param_type=param_type,
            default=default,
            min_value=min_value,
            max_value=max_value,
            step=step,
            choices=choices,
            description=description,
            ui_hidden=ui_hidden,
            on_change=lambda _: self.mark_dirty(),
            file_filter=file_filter,
            is_save_path=is_save_path,
            default_directory=default_directory,
        )
        self.parameters[name] = param
        return param

    def get_parameter(self, name: str) -> Any:
        """Get a parameter value by name."""
        if name in self.parameters:
            return self.parameters[name].value
        raise KeyError(f"Parameter '{name}' not found on node '{self.name}'")

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value by name."""
        if name in self.parameters:
            return self.parameters[name].set(value)
        raise KeyError(f"Parameter '{name}' not found on node '{self.name}'")

    def get_input_value(self, name: str) -> Any:
        """Get the value from an input port."""
        if name in self.inputs:
            return self.inputs[name].get_value()
        raise KeyError(f"Input '{name}' not found on node '{self.name}'")

    def set_output_value(self, name: str, value: Any) -> None:
        """Set the value on an output port."""
        if name in self.outputs:
            self.outputs[name].set_value(value)
        else:
            raise KeyError(f"Output '{name}' not found on node '{self.name}'")

    def mark_dirty(self) -> None:
        """Mark this node as needing re-execution."""
        self._dirty = True
        # Also invalidate all output caches
        for output in self.outputs.values():
            output.invalidate_cache()
        # Propagate to downstream nodes
        for output in self.outputs.values():
            for connected_input in output.connections:
                if connected_input.node is not None:
                    connected_input.node.mark_dirty()

    def is_dirty(self) -> bool:
        """Check if node needs re-execution."""
        return self._dirty

    def can_execute(self) -> tuple[bool, str]:
        """
        Check if node can be executed.

        Returns:
            Tuple of (can_execute, reason_if_not)
        """
        # Check required inputs
        for name, port in self.inputs.items():
            if port.required and not port.is_connected and port.default is None:
                return False, f"Required input '{name}' not connected"

        return True, ""

    @abstractmethod
    def process(self) -> None:
        """
        Execute the node's computation.

        Override this method to implement the node's functionality.
        Use get_input_value() to read inputs and set_output_value()
        to write outputs.
        """
        pass

    def execute(self) -> bool:
        """
        Execute the node if dirty.

        Returns:
            True if execution succeeded, False on error
        """
        if not self._dirty:
            return True

        can_exec, reason = self.can_execute()
        if not can_exec:
            self._error = reason
            return False

        try:
            self._error = None
            self.process()
            self._dirty = False
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def disconnect_all(self) -> None:
        """Disconnect all ports on this node."""
        for port in self.inputs.values():
            disconnect_all(port)
        for port in self.outputs.values():
            disconnect_all(port)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize node to dictionary.

        Returns:
            Dictionary representation of the node
        """
        return {
            "type": self.__class__.__name__,
            "id": self.id,
            "position": list(self.position),
            "parameters": {
                name: param.value for name, param in self.parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node_class: Type[Node]) -> Node:
        """
        Create node from dictionary.

        Args:
            data: Serialized node data
            node_class: The node class to instantiate

        Returns:
            New node instance
        """
        node = node_class()
        node.id = data.get("id", node.id)
        node.position = tuple(data.get("position", [0.0, 0.0]))

        # Restore parameters
        for name, value in data.get("parameters", {}).items():
            if name in node.parameters:
                node.parameters[name].set(value)

        return node

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"
