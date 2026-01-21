"""
Pass-through node.

Simple utility node that passes input directly to output.
Useful for testing and as a connection point.
"""

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class PassThroughNode(Node):
    """
    Pass input image directly to output without modification.

    Useful for:
    - Testing graph execution
    - Creating named connection points
    - Debugging data flow
    """

    name = "Pass Through"
    category = "Utility"
    description = "Pass image through without modification"
    icon = "arrow-right"
    _abstract = False

    def define_ports(self) -> None:
        """Define input and output ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Output image (unchanged)",
        )

    def process(self) -> None:
        """Pass input to output."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        # Pass through (no copy needed since we don't modify)
        self.set_output_value("image", buffer)
