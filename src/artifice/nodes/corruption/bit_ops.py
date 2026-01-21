"""
Bit-level corruption operations.

These operations work at the bit/byte level to create glitch effects
by manipulating the raw data representation of images.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def bit_shift(
    data: NDArray,
    shift: int = 1,
    direction: str = "left",
    wrap: bool = True,
) -> NDArray:
    """
    Shift bits in data.

    Args:
        data: Input array (will be converted to uint8)
        shift: Number of positions to shift
        direction: 'left' or 'right'
        wrap: If True, wrap shifted bits; if False, fill with zeros

    Returns:
        Shifted data as uint8
    """
    # Convert to uint8
    if data.dtype == np.float32 or data.dtype == np.float64:
        uint_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    else:
        uint_data = data.astype(np.uint8)

    shift = shift % 8  # Ensure shift is within byte range

    if direction == "left":
        if wrap:
            result = (uint_data << shift) | (uint_data >> (8 - shift))
        else:
            result = uint_data << shift
    else:  # right
        if wrap:
            result = (uint_data >> shift) | (uint_data << (8 - shift))
        else:
            result = uint_data >> shift

    return result.astype(np.uint8)


def bit_flip(
    data: NDArray,
    bit: int = 0,
    probability: float = 1.0,
) -> NDArray:
    """
    Flip specific bit(s) in data.

    Args:
        data: Input array
        bit: Which bit to flip (0=LSB, 7=MSB), or -1 for random
        probability: Probability of flipping each pixel (0-1)

    Returns:
        Data with flipped bits
    """
    # Convert to uint8
    if data.dtype == np.float32 or data.dtype == np.float64:
        uint_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    else:
        uint_data = data.astype(np.uint8)

    result = uint_data.copy()

    if probability < 1.0:
        mask = np.random.random(uint_data.shape) < probability
    else:
        mask = np.ones(uint_data.shape, dtype=bool)

    if bit == -1:
        # Random bit for each pixel
        random_bits = np.random.randint(0, 8, uint_data.shape)
        for b in range(8):
            bit_mask = (random_bits == b) & mask
            result[bit_mask] ^= (1 << b)
    else:
        bit = max(0, min(7, bit))
        result[mask] ^= (1 << bit)

    return result


def byte_swap(
    data: NDArray,
    swap_type: str = "adjacent",
    stride: int = 1,
) -> NDArray:
    """
    Swap bytes in data.

    Args:
        data: Input array
        swap_type: 'adjacent' (swap pairs), 'reverse' (reverse byte order),
                   'shuffle' (random permutation)
        stride: Distance between swapped bytes

    Returns:
        Data with swapped bytes
    """
    # Convert to uint8 and flatten
    if data.dtype == np.float32 or data.dtype == np.float64:
        uint_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    else:
        uint_data = data.astype(np.uint8)

    original_shape = uint_data.shape
    flat = uint_data.ravel().copy()

    if swap_type == "adjacent":
        # Swap pairs of bytes
        for i in range(0, len(flat) - stride, stride * 2):
            if i + stride < len(flat):
                flat[i], flat[i + stride] = flat[i + stride], flat[i]

    elif swap_type == "reverse":
        # Reverse groups of bytes
        group_size = max(2, stride * 2)
        for i in range(0, len(flat) - group_size + 1, group_size):
            flat[i:i + group_size] = flat[i:i + group_size][::-1]

    elif swap_type == "shuffle":
        # Random permutation within groups
        group_size = max(2, stride * 2)
        for i in range(0, len(flat) - group_size + 1, group_size):
            np.random.shuffle(flat[i:i + group_size])

    return flat.reshape(original_shape)


def xor_noise(
    data: NDArray,
    noise_intensity: float = 0.1,
    pattern: str = "random",
) -> NDArray:
    """
    XOR data with noise pattern.

    Args:
        data: Input array
        noise_intensity: How much noise to apply (0-1)
        pattern: 'random', 'stripes', 'blocks', 'gradient'

    Returns:
        XORed data
    """
    # Convert to uint8
    if data.dtype == np.float32 or data.dtype == np.float64:
        uint_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    else:
        uint_data = data.astype(np.uint8)

    h, w = uint_data.shape[-2:]

    if pattern == "random":
        noise = np.random.randint(0, 256, uint_data.shape, dtype=np.uint8)
    elif pattern == "stripes":
        # Horizontal stripes
        stripe = np.arange(h, dtype=np.uint8)[:, np.newaxis]
        noise = np.broadcast_to(stripe, (h, w))
        if uint_data.ndim == 3:
            noise = np.broadcast_to(noise, uint_data.shape)
        noise = noise.copy()
    elif pattern == "blocks":
        # Checkerboard pattern
        block_size = max(8, min(h, w) // 8)
        y, x = np.mgrid[0:h, 0:w]
        noise = ((y // block_size + x // block_size) % 2 * 255).astype(np.uint8)
        if uint_data.ndim == 3:
            noise = np.broadcast_to(noise, uint_data.shape).copy()
    elif pattern == "gradient":
        # Horizontal gradient
        noise = np.linspace(0, 255, w, dtype=np.uint8)[np.newaxis, :]
        noise = np.broadcast_to(noise, (h, w))
        if uint_data.ndim == 3:
            noise = np.broadcast_to(noise, uint_data.shape)
        noise = noise.copy()
    else:
        noise = np.zeros_like(uint_data)

    # Apply intensity by scaling noise
    noise_scaled = (noise * noise_intensity).astype(np.uint8)

    result = uint_data ^ noise_scaled

    return result


@register_node
class BitShiftNode(Node):
    """
    Shift bits in image data.

    Creates glitch effects by shifting the bits of each pixel value,
    producing characteristic color banding and contrast changes.
    """

    name = "Bit Shift"
    category = "Corruption"
    description = "Shift bits in pixel values"
    icon = "chevrons-left"
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
            description="Shifted image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "shift",
            param_type=ParameterType.INT,
            default=1,
            min_value=0,
            max_value=7,
            description="Number of bits to shift",
        )
        self.add_parameter(
            "direction",
            param_type=ParameterType.ENUM,
            default="left",
            choices=["left", "right"],
            description="Shift direction",
        )
        self.add_parameter(
            "wrap",
            param_type=ParameterType.BOOL,
            default=True,
            description="Wrap shifted bits",
        )

    def process(self) -> None:
        """Apply bit shift."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        shift = self.get_parameter("shift")
        direction = self.get_parameter("direction")
        wrap = self.get_parameter("wrap")

        shifted = bit_shift(buffer.data, shift, direction, wrap)

        # Convert back to float
        result_data = shifted.astype(np.float32) / 255.0

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "bit_shifted": True},
        )

        self.set_output_value("image", result)


@register_node
class BitFlipNode(Node):
    """
    Flip specific bits in image data.

    Creates noise-like glitch effects by toggling individual bits
    in pixel values.
    """

    name = "Bit Flip"
    category = "Corruption"
    description = "Flip bits in pixel values"
    icon = "toggle-left"
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
            "bit",
            param_type=ParameterType.INT,
            default=0,
            min_value=-1,
            max_value=7,
            description="Bit to flip (0=LSB, 7=MSB, -1=random)",
        )
        self.add_parameter(
            "probability",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Probability of flipping each pixel",
        )

    def process(self) -> None:
        """Apply bit flip."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        bit = self.get_parameter("bit")
        probability = self.get_parameter("probability")

        flipped = bit_flip(buffer.data, bit, probability)

        result_data = flipped.astype(np.float32) / 255.0

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "bit_flipped": True},
        )

        self.set_output_value("image", result)


@register_node
class ByteSwapNode(Node):
    """
    Swap bytes in image data.

    Reorders bytes in the raw image data, creating distinctive
    color channel mixing and displacement effects.
    """

    name = "Byte Swap"
    category = "Corruption"
    description = "Swap bytes in image data"
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
            description="Output image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "swap_type",
            param_type=ParameterType.ENUM,
            default="adjacent",
            choices=["adjacent", "reverse", "shuffle"],
            description="Type of byte swap",
        )
        self.add_parameter(
            "stride",
            param_type=ParameterType.INT,
            default=1,
            min_value=1,
            max_value=16,
            description="Distance between swapped bytes",
        )

    def process(self) -> None:
        """Apply byte swap."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        swap_type = self.get_parameter("swap_type")
        stride = self.get_parameter("stride")

        swapped = byte_swap(buffer.data, swap_type, stride)

        result_data = swapped.astype(np.float32) / 255.0

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "byte_swapped": True},
        )

        self.set_output_value("image", result)


@register_node
class XORNoiseNode(Node):
    """
    XOR image with noise pattern.

    Creates glitch effects by XORing image data with various
    noise patterns.
    """

    name = "XOR Noise"
    category = "Corruption"
    description = "XOR with noise pattern"
    icon = "zap"
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
            "intensity",
            param_type=ParameterType.FLOAT,
            default=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Noise intensity",
        )
        self.add_parameter(
            "pattern",
            param_type=ParameterType.ENUM,
            default="random",
            choices=["random", "stripes", "blocks", "gradient"],
            description="Noise pattern",
        )

    def process(self) -> None:
        """Apply XOR noise."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        intensity = self.get_parameter("intensity")
        pattern = self.get_parameter("pattern")

        xored = xor_noise(buffer.data, intensity, pattern)

        result_data = xored.astype(np.float32) / 255.0

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={**buffer.metadata, "xor_noise": True},
        )

        self.set_output_value("image", result)
