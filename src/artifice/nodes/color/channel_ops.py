"""
Channel manipulation nodes.

Split, merge, and swap image channels.
"""

import numpy as np

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class ChannelSplitNode(Node):
    """
    Split image into individual channels.

    Outputs three single-channel images (or grayscale masks)
    that can be processed independently.
    """

    name = "Channel Split"
    category = "Color"
    description = "Split image into separate channels"
    icon = "layers"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image",
            required=True,
        )
        self.add_output(
            "channel_0",
            port_type=PortType.MASK,
            description="First channel (R/H/Y)",
        )
        self.add_output(
            "channel_1",
            port_type=PortType.MASK,
            description="Second channel (G/S/Cb)",
        )
        self.add_output(
            "channel_2",
            port_type=PortType.MASK,
            description="Third channel (B/B/Cr)",
        )

    def process(self) -> None:
        """Split the image into channels."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        # Extract channels
        for i in range(min(3, buffer.channels)):
            channel_data = buffer.data[i:i+1].copy()
            channel_buf = ImageBuffer(
                data=channel_data,
                colorspace=buffer.colorspace,
                border_value=(buffer.border_value[i],) if buffer.border_value else None,
                metadata=buffer.metadata.copy(),
            )
            channel_buf.metadata["channel_index"] = i
            self.set_output_value(f"channel_{i}", channel_buf)


@register_node
class ChannelMergeNode(Node):
    """
    Merge individual channels back into a single image.

    Takes three single-channel inputs and combines them.
    """

    name = "Channel Merge"
    category = "Color"
    description = "Combine channels into single image"
    icon = "layers"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "channel_0",
            port_type=PortType.MASK,
            description="First channel",
            required=True,
        )
        self.add_input(
            "channel_1",
            port_type=PortType.MASK,
            description="Second channel",
            required=True,
        )
        self.add_input(
            "channel_2",
            port_type=PortType.MASK,
            description="Third channel",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Merged image",
        )

    def process(self) -> None:
        """Merge channels into a single image."""
        ch0: ImageBuffer = self.get_input_value("channel_0")
        ch1: ImageBuffer = self.get_input_value("channel_1")
        ch2: ImageBuffer = self.get_input_value("channel_2")

        if ch0 is None or ch1 is None or ch2 is None:
            raise ValueError("All three channels required")

        # Verify dimensions match
        if ch0.shape[1:] != ch1.shape[1:] or ch0.shape[1:] != ch2.shape[1:]:
            raise ValueError("Channel dimensions must match")

        # Merge
        merged_data = np.concatenate([
            ch0.data[0:1],
            ch1.data[0:1],
            ch2.data[0:1],
        ], axis=0)

        # Use first channel's colorspace
        result = ImageBuffer(
            data=merged_data,
            colorspace=ch0.colorspace,
            border_value=(
                ch0.border_value[0] if ch0.border_value else 0.0,
                ch1.border_value[0] if ch1.border_value else 0.0,
                ch2.border_value[0] if ch2.border_value else 0.0,
            ),
            metadata=ch0.metadata.copy(),
        )

        self.set_output_value("image", result)


@register_node
class ChannelSwapNode(Node):
    """
    Swap or rearrange image channels.

    Allows arbitrary channel remapping for creative effects.
    """

    name = "Channel Swap"
    category = "Color"
    description = "Rearrange image channels"
    icon = "shuffle"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Output with swapped channels",
        )

    def define_parameters(self) -> None:
        """Define channel mapping parameters."""
        self.add_parameter(
            "channel_0_source",
            param_type=ParameterType.ENUM,
            default="0",
            choices=["0", "1", "2"],
            description="Source for output channel 0",
        )
        self.add_parameter(
            "channel_1_source",
            param_type=ParameterType.ENUM,
            default="1",
            choices=["0", "1", "2"],
            description="Source for output channel 1",
        )
        self.add_parameter(
            "channel_2_source",
            param_type=ParameterType.ENUM,
            default="2",
            choices=["0", "1", "2"],
            description="Source for output channel 2",
        )

    def process(self) -> None:
        """Swap channels according to mapping."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        src0 = int(self.get_parameter("channel_0_source"))
        src1 = int(self.get_parameter("channel_1_source"))
        src2 = int(self.get_parameter("channel_2_source"))

        # Remap channels
        swapped_data = np.stack([
            buffer.data[src0],
            buffer.data[src1],
            buffer.data[src2],
        ], axis=0)

        # Remap border values
        new_border = None
        if buffer.border_value:
            new_border = (
                buffer.border_value[src0],
                buffer.border_value[src1],
                buffer.border_value[src2],
            )

        result = ImageBuffer(
            data=swapped_data,
            colorspace=buffer.colorspace,
            border_value=new_border,
            metadata=buffer.metadata.copy(),
        )

        self.set_output_value("image", result)
