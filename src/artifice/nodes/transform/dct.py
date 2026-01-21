"""
Discrete Cosine Transform (DCT) nodes.

DCT transforms image data to frequency domain, commonly used in
JPEG compression. DCT operates on 8x8 blocks by default.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import fftpack

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def dct_2d(
    data: NDArray[np.float32],
    block_size: int = 8,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """
    Apply 2D DCT to image data.

    Args:
        data: 2D or 3D (CHW) array
        block_size: Size of DCT blocks (0 for full image)
        normalize: Normalize coefficients to [0, 1] range

    Returns:
        DCT coefficients
    """
    if data.ndim == 3:
        # Process each channel
        result = np.zeros_like(data)
        for c in range(data.shape[0]):
            result[c] = _dct_2d_channel(data[c], block_size, normalize)
        return result
    else:
        return _dct_2d_channel(data, block_size, normalize)


def _dct_2d_channel(
    data: NDArray[np.float32],
    block_size: int,
    normalize: bool,
) -> NDArray[np.float32]:
    """Apply DCT to a single 2D channel."""
    h, w = data.shape

    if block_size == 0 or (block_size >= h and block_size >= w):
        # Full image DCT
        coeffs = fftpack.dct(fftpack.dct(data.T, norm='ortho').T, norm='ortho')
    else:
        # Block-based DCT (like JPEG)
        coeffs = np.zeros_like(data)

        # Pad if necessary
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(data, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            padded = data

        ph, pw = padded.shape

        # Process each block
        for y in range(0, ph, block_size):
            for x in range(0, pw, block_size):
                block = padded[y:y+block_size, x:x+block_size]
                dct_block = fftpack.dct(
                    fftpack.dct(block.T, norm='ortho').T, norm='ortho'
                )
                # Only copy back the valid region
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                coeffs[y:y_end, x:x_end] = dct_block[:y_end-y, :x_end-x]

    if normalize:
        # Normalize to [0, 1] for visualization
        coeffs_min = coeffs.min()
        coeffs_max = coeffs.max()
        if coeffs_max > coeffs_min:
            coeffs = (coeffs - coeffs_min) / (coeffs_max - coeffs_min)

    return coeffs.astype(np.float32)


def idct_2d(
    coeffs: NDArray[np.float32],
    block_size: int = 8,
    denormalize: bool = True,
    original_range: tuple[float, float] | None = None,
) -> NDArray[np.float32]:
    """
    Apply inverse 2D DCT.

    Args:
        coeffs: DCT coefficients
        block_size: Size of DCT blocks (0 for full image)
        denormalize: Assume coeffs are normalized
        original_range: Original min/max for denormalization

    Returns:
        Reconstructed image data
    """
    if coeffs.ndim == 3:
        result = np.zeros_like(coeffs)
        for c in range(coeffs.shape[0]):
            result[c] = _idct_2d_channel(coeffs[c], block_size, denormalize, original_range)
        return result
    else:
        return _idct_2d_channel(coeffs, block_size, denormalize, original_range)


def _idct_2d_channel(
    coeffs: NDArray[np.float32],
    block_size: int,
    denormalize: bool,
    original_range: tuple[float, float] | None,
) -> NDArray[np.float32]:
    """Apply IDCT to a single 2D channel."""
    h, w = coeffs.shape

    # For inverse, we don't need to denormalize if we just want reconstruction
    # The coefficients should be used as-is for proper reconstruction

    if block_size == 0 or (block_size >= h and block_size >= w):
        # Full image IDCT
        result = fftpack.idct(fftpack.idct(coeffs.T, norm='ortho').T, norm='ortho')
    else:
        # Block-based IDCT
        result = np.zeros_like(coeffs)

        # Pad if necessary
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(coeffs, ((0, pad_h), (0, pad_w)), mode='constant')
        else:
            padded = coeffs

        ph, pw = padded.shape

        for y in range(0, ph, block_size):
            for x in range(0, pw, block_size):
                block = padded[y:y+block_size, x:x+block_size]
                idct_block = fftpack.idct(
                    fftpack.idct(block.T, norm='ortho').T, norm='ortho'
                )
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                result[y:y_end, x:x_end] = idct_block[:y_end-y, :x_end-x]

    return result.astype(np.float32)


def quantize_dct(
    coeffs: NDArray[np.float32],
    quality: int = 50,
    block_size: int = 8,
) -> NDArray[np.float32]:
    """
    Quantize DCT coefficients (JPEG-style).

    Args:
        coeffs: DCT coefficients
        quality: Quality factor (1-100, higher = better)
        block_size: Block size

    Returns:
        Quantized coefficients
    """
    # Standard JPEG luminance quantization matrix (8x8)
    jpeg_quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=np.float32)

    # Scale quantization matrix by quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    quant_matrix = np.floor((jpeg_quant * scale + 50) / 100)
    quant_matrix = np.clip(quant_matrix, 1, 255)

    # Resize to match block size if needed
    if block_size != 8:
        from scipy.ndimage import zoom
        quant_matrix = zoom(quant_matrix, block_size / 8, order=1)

    def quantize_channel(channel: NDArray) -> NDArray:
        h, w = channel.shape
        result = np.zeros_like(channel)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                block = channel[y:y_end, x:x_end]

                # Quantize
                qh, qw = block.shape
                q_block = np.round(block / quant_matrix[:qh, :qw])
                result[y:y_end, x:x_end] = q_block * quant_matrix[:qh, :qw]

        return result

    if coeffs.ndim == 3:
        result = np.zeros_like(coeffs)
        for c in range(coeffs.shape[0]):
            result[c] = quantize_channel(coeffs[c])
        return result
    else:
        return quantize_channel(coeffs)


@register_node
class DCTNode(Node):
    """
    Apply Discrete Cosine Transform to an image.

    DCT transforms the image to frequency domain. Can operate on
    the full image or in blocks (like JPEG).
    """

    name = "DCT"
    category = "Transform"
    description = "Discrete Cosine Transform"
    icon = "bar-chart-2"
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
            "coefficients",
            port_type=PortType.IMAGE,
            description="DCT coefficients (as image)",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "block_size",
            param_type=ParameterType.INT,
            default=8,
            min_value=0,
            max_value=64,
            description="Block size (0 for full image)",
        )
        self.add_parameter(
            "normalize",
            param_type=ParameterType.BOOL,
            default=True,
            description="Normalize output to [0, 1]",
        )

    def process(self) -> None:
        """Perform DCT."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        block_size = self.get_parameter("block_size")
        normalize = self.get_parameter("normalize")

        coeffs = dct_2d(buffer.data, block_size, normalize)

        result = ImageBuffer(
            data=coeffs,
            colorspace=buffer.colorspace,
            metadata={
                **buffer.metadata,
                "dct_block_size": block_size,
                "dct_normalized": normalize,
            },
        )

        self.set_output_value("coefficients", result)


@register_node
class InverseDCTNode(Node):
    """
    Apply inverse Discrete Cosine Transform.

    Reconstructs an image from DCT coefficients.
    """

    name = "Inverse DCT"
    category = "Transform"
    description = "Inverse Discrete Cosine Transform"
    icon = "bar-chart-2"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "coefficients",
            port_type=PortType.IMAGE,
            description="DCT coefficients",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Reconstructed image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "block_size",
            param_type=ParameterType.INT,
            default=8,
            min_value=0,
            max_value=64,
            description="Block size (must match forward DCT)",
        )

    def process(self) -> None:
        """Perform inverse DCT."""
        buffer: ImageBuffer = self.get_input_value("coefficients")

        if buffer is None:
            raise ValueError("No coefficients input")

        block_size = self.get_parameter("block_size")

        # Get block size from metadata if available
        if "dct_block_size" in buffer.metadata:
            block_size = buffer.metadata["dct_block_size"]

        result_data = idct_2d(buffer.data, block_size, denormalize=False)

        # Clip to valid range
        result_data = np.clip(result_data, 0.0, 1.0)

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            metadata={"reconstructed_from": "dct"},
        )

        self.set_output_value("image", result)
