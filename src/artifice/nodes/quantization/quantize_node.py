"""
Quantization nodes for GLIC-style processing.

QuantizeNode: Reduce precision of values (for compression)
DequantizeNode: Restore values from quantized form
"""

import numpy as np

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def quantize_value(value: np.ndarray, bits: int, signed: bool = True) -> np.ndarray:
    """
    Quantize floating point values to specified bit depth.

    For signed mode (residuals): maps [-1, 1] to integer range
    For unsigned mode: maps [0, 1] to integer range

    Args:
        value: Float values to quantize
        bits: Number of bits for quantization (1-16)
        signed: Whether to use signed quantization (for residuals)

    Returns:
        Quantized integer values
    """
    bits = max(1, min(16, bits))

    if signed:
        # Signed: map [-1, 1] to [-(2^(bits-1)), 2^(bits-1)-1]
        max_val = (1 << (bits - 1)) - 1
        min_val = -(1 << (bits - 1))
        # Clamp to [-1, 1] then scale
        clamped = np.clip(value, -1.0, 1.0)
        quantized = np.round(clamped * max_val).astype(np.int32)
        quantized = np.clip(quantized, min_val, max_val)
    else:
        # Unsigned: map [0, 1] to [0, 2^bits - 1]
        max_val = (1 << bits) - 1
        clamped = np.clip(value, 0.0, 1.0)
        quantized = np.round(clamped * max_val).astype(np.int32)

    return quantized


def dequantize_value(
    quantized: np.ndarray, bits: int, signed: bool = True
) -> np.ndarray:
    """
    Dequantize integer values back to floating point.

    Args:
        quantized: Quantized integer values
        bits: Number of bits used in quantization
        signed: Whether signed quantization was used

    Returns:
        Floating point values in original range
    """
    bits = max(1, min(16, bits))

    if signed:
        max_val = (1 << (bits - 1)) - 1
        dequantized = quantized.astype(np.float32) / max_val
        dequantized = np.clip(dequantized, -1.0, 1.0)
    else:
        max_val = (1 << bits) - 1
        dequantized = quantized.astype(np.float32) / max_val
        dequantized = np.clip(dequantized, 0.0, 1.0)

    return dequantized


def adaptive_quantize(
    data: np.ndarray, min_bits: int = 2, max_bits: int = 8, threshold: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptively quantize based on local variance.

    Areas with low variance use fewer bits.

    Args:
        data: Input data
        min_bits: Minimum bits for low-variance regions
        max_bits: Maximum bits for high-variance regions
        threshold: Variance threshold for bit allocation

    Returns:
        Tuple of (quantized data, bit map showing bits used per region)
    """
    # For simplicity, use global variance to select bits
    variance = np.var(data)

    if variance < threshold * 0.1:
        bits = min_bits
    elif variance < threshold:
        bits = (min_bits + max_bits) // 2
    else:
        bits = max_bits

    quantized = quantize_value(data, bits, signed=True)
    bit_map = np.full(data.shape, bits, dtype=np.uint8)

    return quantized, bit_map


@register_node
class QuantizeNode(Node):
    """
    Quantize image values to reduced bit depth.

    Used in GLIC-style compression to reduce the precision of
    residual values before encoding. Supports both uniform and
    adaptive quantization modes.
    """

    name = "Quantize"
    category = "Quantization"
    description = "Quantize values to reduced bit depth"
    icon = "layers"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image (typically residuals)",
            required=True,
        )
        self.add_output(
            "quantized",
            port_type=PortType.IMAGE,
            description="Quantized image",
        )
        self.add_output(
            "quantized_int",
            port_type=PortType.ARRAY,
            description="Quantized integer values (for encoding)",
        )

    def define_parameters(self) -> None:
        """Define quantization parameters."""
        self.add_parameter(
            "bits",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=16,
            description="Quantization bit depth",
        )
        self.add_parameter(
            "signed",
            param_type=ParameterType.BOOL,
            default=True,
            description="Use signed quantization (for residuals)",
        )
        self.add_parameter(
            "mode",
            param_type=ParameterType.ENUM,
            default="uniform",
            choices=["uniform", "adaptive"],
            description="Quantization mode",
        )

    def process(self) -> None:
        """Perform quantization."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        bits = self.get_parameter("bits")
        signed = self.get_parameter("signed")
        mode = self.get_parameter("mode")

        if mode == "adaptive":
            # Adaptive quantization per channel
            quantized_data = np.zeros_like(buffer.data, dtype=np.float32)
            quantized_int = np.zeros_like(buffer.data, dtype=np.int32)

            for c in range(buffer.channels):
                q_int, _ = adaptive_quantize(buffer.data[c], min_bits=2, max_bits=bits)
                quantized_int[c] = q_int
                # Dequantize for preview
                actual_bits = bits  # simplified
                quantized_data[c] = dequantize_value(q_int, actual_bits, signed)
        else:
            # Uniform quantization
            quantized_int = quantize_value(buffer.data, bits, signed)
            quantized_data = dequantize_value(quantized_int, bits, signed)

        # Create output buffer with dequantized preview
        result = ImageBuffer(
            data=quantized_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={
                **buffer.metadata,
                "quantization_bits": bits,
                "quantization_signed": signed,
                "quantization_mode": mode,
            },
        )

        self.set_output_value("quantized", result)
        self.set_output_value("quantized_int", quantized_int)


@register_node
class DequantizeNode(Node):
    """
    Dequantize integer values back to floating point.

    Used to restore values from their quantized integer form.
    """

    name = "Dequantize"
    category = "Quantization"
    description = "Restore values from quantized form"
    icon = "layers"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "quantized_int",
            port_type=PortType.ARRAY,
            description="Quantized integer values",
            required=True,
        )
        self.add_input(
            "reference",
            port_type=PortType.IMAGE,
            description="Reference image for metadata",
            required=False,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Dequantized image",
        )

    def define_parameters(self) -> None:
        """Define dequantization parameters."""
        self.add_parameter(
            "bits",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=16,
            description="Quantization bit depth used",
        )
        self.add_parameter(
            "signed",
            param_type=ParameterType.BOOL,
            default=True,
            description="Was signed quantization used",
        )

    def process(self) -> None:
        """Perform dequantization."""
        quantized_int = self.get_input_value("quantized_int")
        reference: ImageBuffer | None = self.get_input_value("reference")

        if quantized_int is None:
            raise ValueError("No quantized input")

        bits = self.get_parameter("bits")
        signed = self.get_parameter("signed")

        dequantized = dequantize_value(quantized_int, bits, signed)

        # Get metadata from reference if available
        if reference is not None:
            colorspace = reference.colorspace
            border_value = reference.border_value
            metadata = dict(reference.metadata)
        else:
            colorspace = "RGB"
            border_value = None
            metadata = {}

        metadata["dequantized"] = True

        result = ImageBuffer(
            data=dequantized,
            colorspace=colorspace,
            border_value=border_value,
            metadata=metadata,
        )

        self.set_output_value("image", result)
