"""
Color space conversion node.

Converts images between all 16 GLIC color spaces.
"""

from artifice.core.data_types import ColorSpace, ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node
from artifice.nodes.color.conversions import convert_colorspace, list_colorspaces


@register_node
class ColorSpaceNode(Node):
    """
    Convert image between color spaces.

    Supports all 16 GLIC color spaces:
    - RGB (default)
    - HSB/HSV (Hue, Saturation, Brightness)
    - HWB (Hue, Whiteness, Blackness)
    - CMY (Cyan, Magenta, Yellow)
    - YUV, YCbCr, YPbPr, YDbDr (luminance-chrominance)
    - XYZ, LAB, LUV, HCL, YXY (perceptual/CIE)
    - OHTA, R-GGB-G (compact representations)
    - GREY/Greyscale

    Each color space produces different glitch characteristics when
    used with prediction and quantization nodes.
    """

    name = "Color Space"
    category = "Color"
    description = "Convert between color spaces"
    icon = "palette"
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
            description="Converted image",
        )

    def define_parameters(self) -> None:
        """Define color space selection parameter."""
        self.add_parameter(
            "target_space",
            param_type=ParameterType.ENUM,
            default="YCbCr",
            choices=list_colorspaces(),
            description="Target color space",
        )

    def process(self) -> None:
        """Convert the image to the target color space."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        target_space = self.get_parameter("target_space")

        # Get source color space from buffer
        source_space = buffer.colorspace
        if isinstance(source_space, ColorSpace):
            source_space = source_space.value

        # Convert
        converted_data = convert_colorspace(
            buffer.data,
            source_space,
            target_space,
        )

        # Create output buffer
        result = ImageBuffer(
            data=converted_data,
            colorspace=target_space,
            border_value=buffer.border_value,
            metadata=buffer.metadata.copy(),
        )
        result.metadata["source_colorspace"] = source_space

        self.set_output_value("image", result)
