"""
Data-level corruption operations.

These operations manipulate the structure of image data - repeating,
dropping, or weaving sections to create glitch effects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def data_repeat(
    data: NDArray[np.float32],
    repeat_interval: int = 16,
    repeat_length: int = 8,
    direction: str = "horizontal",
) -> NDArray[np.float32]:
    """
    Repeat sections of data at regular intervals.

    Args:
        data: 2D (HW) or 3D (CHW) image array
        repeat_interval: Interval between repeated sections
        repeat_length: Length of section to repeat
        direction: 'horizontal' or 'vertical'

    Returns:
        Image with repeated sections
    """
    # Handle 2D arrays
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        was_2d = True
    else:
        was_2d = False

    result = data.copy()
    c, h, w = data.shape

    if direction == "horizontal":
        for x in range(0, w - repeat_length, repeat_interval):
            # Copy the section at position x to subsequent positions
            src = result[:, :, x:x + repeat_length]
            dest_x = x + repeat_length
            while dest_x + repeat_length <= w:
                result[:, :, dest_x:dest_x + repeat_length] = src
                dest_x += repeat_length
                if dest_x >= x + repeat_interval:
                    break
    else:  # vertical
        for y in range(0, h - repeat_length, repeat_interval):
            src = result[:, y:y + repeat_length, :]
            dest_y = y + repeat_length
            while dest_y + repeat_length <= h:
                result[:, dest_y:dest_y + repeat_length, :] = src
                dest_y += repeat_length
                if dest_y >= y + repeat_interval:
                    break

    if was_2d:
        result = result[0]

    return result


def data_drop(
    data: NDArray[np.float32],
    drop_interval: int = 32,
    drop_length: int = 8,
    direction: str = "horizontal",
    fill_mode: str = "shift",
) -> NDArray[np.float32]:
    """
    Drop sections of data at regular intervals.

    Args:
        data: 2D (HW) or 3D (CHW) image array
        drop_interval: Interval between dropped sections
        drop_length: Length of section to drop
        direction: 'horizontal' or 'vertical'
        fill_mode: 'shift' (shift remaining data), 'black', 'previous'

    Returns:
        Image with dropped sections
    """
    # Handle 2D arrays
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        was_2d = True
    else:
        was_2d = False

    result = data.copy()
    c, h, w = data.shape

    if direction == "horizontal":
        drop_positions = list(range(0, w, drop_interval))

        if fill_mode == "shift":
            # Collect non-dropped columns
            keep_mask = np.ones(w, dtype=bool)
            for pos in drop_positions:
                end = min(pos + drop_length, w)
                keep_mask[pos:end] = False

            kept_data = result[:, :, keep_mask]
            # Pad to original width
            new_w = kept_data.shape[2]
            if new_w < w:
                padding = np.zeros((c, h, w - new_w), dtype=np.float32)
                result = np.concatenate([kept_data, padding], axis=2)
            else:
                result = kept_data[:, :, :w]

        elif fill_mode == "black":
            for pos in drop_positions:
                end = min(pos + drop_length, w)
                result[:, :, pos:end] = 0.0

        elif fill_mode == "previous":
            for pos in drop_positions:
                if pos > 0:
                    end = min(pos + drop_length, w)
                    result[:, :, pos:end] = result[:, :, pos - 1:pos]

    else:  # vertical
        drop_positions = list(range(0, h, drop_interval))

        if fill_mode == "shift":
            keep_mask = np.ones(h, dtype=bool)
            for pos in drop_positions:
                end = min(pos + drop_length, h)
                keep_mask[pos:end] = False

            kept_data = result[:, keep_mask, :]
            new_h = kept_data.shape[1]
            if new_h < h:
                padding = np.zeros((c, h - new_h, w), dtype=np.float32)
                result = np.concatenate([kept_data, padding], axis=1)
            else:
                result = kept_data[:, :h, :]

        elif fill_mode == "black":
            for pos in drop_positions:
                end = min(pos + drop_length, h)
                result[:, pos:end, :] = 0.0

        elif fill_mode == "previous":
            for pos in drop_positions:
                if pos > 0:
                    end = min(pos + drop_length, h)
                    result[:, pos:end, :] = result[:, pos - 1:pos, :]

    if was_2d:
        result = result[0]

    return result


def data_weave(
    data1: NDArray[np.float32],
    data2: NDArray[np.float32],
    weave_width: int = 8,
    direction: str = "horizontal",
    blend: float = 0.0,
) -> NDArray[np.float32]:
    """
    Weave two images together by alternating sections.

    Args:
        data1: First image (2D HW or 3D CHW)
        data2: Second image (2D HW or 3D CHW)
        weave_width: Width of each alternating section
        direction: 'horizontal' or 'vertical'
        blend: Blend factor at boundaries (0=hard, 1=full blend)

    Returns:
        Woven image
    """
    # Handle 2D arrays
    was_2d = False
    if data1.ndim == 2:
        data1 = data1[np.newaxis, :, :]
        was_2d = True
    if data2.ndim == 2:
        data2 = data2[np.newaxis, :, :]

    # Ensure same shape
    c1, h1, w1 = data1.shape
    c2, h2, w2 = data2.shape
    c = min(c1, c2)
    h = min(h1, h2)
    w = min(w1, w2)

    d1 = data1[:c, :h, :w]
    d2 = data2[:c, :h, :w]

    result = np.zeros((c, h, w), dtype=np.float32)

    if direction == "horizontal":
        for x in range(0, w, weave_width * 2):
            # First section from data1
            end1 = min(x + weave_width, w)
            result[:, :, x:end1] = d1[:, :, x:end1]

            # Second section from data2
            start2 = end1
            end2 = min(start2 + weave_width, w)
            if start2 < w:
                result[:, :, start2:end2] = d2[:, :, start2:end2]

            # Apply blend at boundaries
            if blend > 0 and end1 < w:
                blend_width = max(1, int(weave_width * blend / 2))
                for bx in range(blend_width):
                    if end1 - 1 - bx >= x and end1 - 1 - bx < w:
                        alpha = (bx + 1) / (blend_width + 1)
                        result[:, :, end1 - 1 - bx] = (
                            d1[:, :, end1 - 1 - bx] * (1 - alpha) +
                            d2[:, :, end1 - 1 - bx] * alpha
                        )
    else:  # vertical
        for y in range(0, h, weave_width * 2):
            end1 = min(y + weave_width, h)
            result[:, y:end1, :] = d1[:, y:end1, :]

            start2 = end1
            end2 = min(start2 + weave_width, h)
            if start2 < h:
                result[:, start2:end2, :] = d2[:, start2:end2, :]

            if blend > 0 and end1 < h:
                blend_width = max(1, int(weave_width * blend / 2))
                for by in range(blend_width):
                    if end1 - 1 - by >= y and end1 - 1 - by < h:
                        alpha = (by + 1) / (blend_width + 1)
                        result[:, end1 - 1 - by, :] = (
                            d1[:, end1 - 1 - by, :] * (1 - alpha) +
                            d2[:, end1 - 1 - by, :] * alpha
                        )

    if was_2d:
        result = result[0]

    return result


def data_scramble(
    data: NDArray[np.float32],
    block_size: int = 16,
    scramble_ratio: float = 0.5,
) -> NDArray[np.float32]:
    """
    Scramble blocks of image data.

    Args:
        data: 2D (HW) or 3D (CHW) image array
        block_size: Size of blocks to scramble
        scramble_ratio: Ratio of blocks to scramble (0-1)

    Returns:
        Scrambled image
    """
    # Handle 2D arrays
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        was_2d = True
    else:
        was_2d = False

    result = data.copy()
    c, h, w = data.shape

    # Calculate number of blocks
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size

    if num_blocks_y < 2 or num_blocks_x < 2:
        return result

    # Create list of block positions
    positions = [
        (by, bx)
        for by in range(num_blocks_y)
        for bx in range(num_blocks_x)
    ]

    # Select blocks to scramble
    num_to_scramble = int(len(positions) * scramble_ratio)
    if num_to_scramble < 2:
        return result

    scramble_indices = np.random.choice(len(positions), num_to_scramble, replace=False)
    scramble_positions = [positions[i] for i in scramble_indices]

    # Shuffle and apply
    shuffled = scramble_positions.copy()
    np.random.shuffle(shuffled)

    # Store blocks
    blocks = {}
    for by, bx in scramble_positions:
        y_start = by * block_size
        x_start = bx * block_size
        blocks[(by, bx)] = result[:, y_start:y_start + block_size, x_start:x_start + block_size].copy()

    # Place shuffled blocks
    for orig, shuffled_pos in zip(scramble_positions, shuffled):
        by, bx = orig
        y_start = by * block_size
        x_start = bx * block_size
        result[:, y_start:y_start + block_size, x_start:x_start + block_size] = blocks[shuffled_pos]

    if was_2d:
        result = result[0]

    return result


@register_node
class DataRepeatNode(Node):
    """
    Repeat sections of image data.

    Creates stuttering/echo effects by duplicating sections
    of the image at regular intervals.
    """

    name = "Data Repeat"
    category = "Corruption"
    description = "Repeat sections of data"
    icon = "copy"
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
            description="Output image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "repeat_interval",
            param_type=ParameterType.INT,
            default=32,
            min_value=4,
            max_value=256,
            description="Interval between repeated sections",
        )
        self.add_parameter(
            "repeat_length",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=128,
            description="Length of section to repeat",
        )
        self.add_parameter(
            "direction",
            param_type=ParameterType.ENUM,
            default="horizontal",
            choices=["horizontal", "vertical"],
            description="Repeat direction",
        )

    def process(self) -> None:
        """Apply data repeat."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        interval = self.get_parameter("repeat_interval")
        length = self.get_parameter("repeat_length")
        direction = self.get_parameter("direction")

        result_data = data_repeat(buffer.data, interval, length, direction)

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "data_repeated": True},
        )

        self.set_output_value("image", result)


@register_node
class DataDropNode(Node):
    """
    Drop sections of image data.

    Creates gap/displacement effects by removing sections
    and optionally shifting remaining data.
    """

    name = "Data Drop"
    category = "Corruption"
    description = "Drop sections of data"
    icon = "minus-square"
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
            description="Output image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "drop_interval",
            param_type=ParameterType.INT,
            default=32,
            min_value=4,
            max_value=256,
            description="Interval between dropped sections",
        )
        self.add_parameter(
            "drop_length",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=128,
            description="Length of section to drop",
        )
        self.add_parameter(
            "direction",
            param_type=ParameterType.ENUM,
            default="horizontal",
            choices=["horizontal", "vertical"],
            description="Drop direction",
        )
        self.add_parameter(
            "fill_mode",
            param_type=ParameterType.ENUM,
            default="shift",
            choices=["shift", "black", "previous"],
            description="How to fill dropped regions",
        )

    def process(self) -> None:
        """Apply data drop."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        interval = self.get_parameter("drop_interval")
        length = self.get_parameter("drop_length")
        direction = self.get_parameter("direction")
        fill_mode = self.get_parameter("fill_mode")

        result_data = data_drop(buffer.data, interval, length, direction, fill_mode)

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "data_dropped": True},
        )

        self.set_output_value("image", result)


@register_node
class DataWeaveNode(Node):
    """
    Weave two images together.

    Creates interlaced effects by alternating sections from
    two input images.
    """

    name = "Data Weave"
    category = "Corruption"
    description = "Weave two images together"
    icon = "layers"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image_a",
            port_type=PortType.IMAGE,
            description="First input image",
            required=True,
        )
        self.add_input(
            "image_b",
            port_type=PortType.IMAGE,
            description="Second input image",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Woven image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "weave_width",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=128,
            description="Width of each alternating section",
        )
        self.add_parameter(
            "direction",
            param_type=ParameterType.ENUM,
            default="horizontal",
            choices=["horizontal", "vertical"],
            description="Weave direction",
        )
        self.add_parameter(
            "blend",
            param_type=ParameterType.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            description="Blend factor at boundaries",
        )

    def process(self) -> None:
        """Apply data weave."""
        buffer_a: ImageBuffer = self.get_input_value("image_a")
        buffer_b: ImageBuffer = self.get_input_value("image_b")

        if buffer_a is None:
            raise ValueError("No first input image")
        if buffer_b is None:
            raise ValueError("No second input image")

        weave_width = self.get_parameter("weave_width")
        direction = self.get_parameter("direction")
        blend = self.get_parameter("blend")

        result_data = data_weave(
            buffer_a.data, buffer_b.data, weave_width, direction, blend
        )

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer_a.colorspace,
            border_value=buffer_a.border_value,
            metadata={**buffer_a.metadata, "data_woven": True},
        )

        self.set_output_value("image", result)


@register_node
class DataScrambleNode(Node):
    """
    Scramble blocks of image data.

    Randomly reorders blocks of the image to create
    fragmented/mosaic glitch effects.
    """

    name = "Data Scramble"
    category = "Corruption"
    description = "Scramble blocks of data"
    icon = "grid"
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
            description="Scrambled image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "block_size",
            param_type=ParameterType.INT,
            default=16,
            min_value=4,
            max_value=128,
            description="Size of blocks to scramble",
        )
        self.add_parameter(
            "scramble_ratio",
            param_type=ParameterType.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            description="Ratio of blocks to scramble",
        )

    def process(self) -> None:
        """Apply data scramble."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        block_size = self.get_parameter("block_size")
        scramble_ratio = self.get_parameter("scramble_ratio")

        result_data = data_scramble(buffer.data, block_size, scramble_ratio)

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "data_scrambled": True},
        )

        self.set_output_value("image", result)
