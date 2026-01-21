"""
Wavelet transform nodes.

Provides forward and inverse wavelet transforms using PyWavelets.
Supports 68+ wavelet families including Haar, Daubechies, Symlets, etc.
Two modes: FWT (Fast Wavelet Transform) and WPT (Wavelet Packet Transform).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pywt
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def list_wavelets() -> list[str]:
    """Get list of all available wavelet names."""
    return pywt.wavelist()


def wavelet_transform(
    data: NDArray[np.float32],
    wavelet: str = "haar",
    mode: str = "fwt",
    level: int | None = None,
) -> dict[str, Any]:
    """
    Perform wavelet transform on image data.

    Args:
        data: 2D or 3D (CHW) array
        wavelet: Wavelet name (e.g., 'haar', 'db4', 'sym4')
        mode: 'fwt' for standard DWT or 'wpt' for wavelet packet
        level: Decomposition level (None for max)

    Returns:
        Dictionary containing coefficients and metadata
    """
    if wavelet not in pywt.wavelist():
        raise ValueError(f"Unknown wavelet: {wavelet}")

    is_3d = data.ndim == 3

    if is_3d:
        # Process each channel
        channels = data.shape[0]
        coeffs_list = []
        for c in range(channels):
            coeffs = _wavelet_transform_2d(data[c], wavelet, mode, level)
            coeffs_list.append(coeffs)

        return {
            "coeffs": coeffs_list,
            "wavelet": wavelet,
            "mode": mode,
            "shape": data.shape,
            "is_3d": True,
        }
    else:
        coeffs = _wavelet_transform_2d(data, wavelet, mode, level)
        return {
            "coeffs": coeffs,
            "wavelet": wavelet,
            "mode": mode,
            "shape": data.shape,
            "is_3d": False,
        }


def _wavelet_transform_2d(
    data: NDArray[np.float32],
    wavelet: str,
    mode: str,
    level: int | None,
) -> Any:
    """Perform 2D wavelet transform."""
    if mode == "wpt":
        # Wavelet Packet Transform
        wp = pywt.WaveletPacket2D(data, wavelet, maxlevel=level)
        return wp
    else:
        # Standard Fast Wavelet Transform (DWT)
        if level is None:
            level = pywt.dwt_max_level(min(data.shape), wavelet)
        coeffs = pywt.wavedec2(data, wavelet, level=level)
        return coeffs


def inverse_wavelet(
    coeffs_data: dict[str, Any],
    wavelet: str | None = None,
    mode: str | None = None,
) -> NDArray[np.float32]:
    """
    Perform inverse wavelet transform.

    Args:
        coeffs_data: Coefficients from wavelet_transform
        wavelet: Override wavelet (uses stored if None)
        mode: Override mode (uses stored if None)

    Returns:
        Reconstructed image data
    """
    wavelet = wavelet or coeffs_data["wavelet"]
    mode = mode or coeffs_data["mode"]
    is_3d = coeffs_data["is_3d"]

    if is_3d:
        coeffs_list = coeffs_data["coeffs"]
        channels = len(coeffs_list)
        shape = coeffs_data["shape"]

        result = np.zeros(shape, dtype=np.float32)
        for c in range(channels):
            result[c] = _inverse_wavelet_2d(coeffs_list[c], wavelet, mode)

        return result
    else:
        return _inverse_wavelet_2d(coeffs_data["coeffs"], wavelet, mode)


def _inverse_wavelet_2d(
    coeffs: Any,
    wavelet: str,
    mode: str,
) -> NDArray[np.float32]:
    """Perform inverse 2D wavelet transform."""
    if mode == "wpt":
        # Wavelet Packet inverse
        if isinstance(coeffs, pywt.WaveletPacket2D):
            return coeffs.reconstruct(update=False).astype(np.float32)
        else:
            raise ValueError("WPT coefficients must be WaveletPacket2D object")
    else:
        # Standard IDWT
        result = pywt.waverec2(coeffs, wavelet)
        return result.astype(np.float32)


def compress_coefficients(
    coeffs_data: dict[str, Any],
    threshold: float = 0.1,
    keep_approximation: bool = True,
) -> dict[str, Any]:
    """
    Compress wavelet coefficients by zeroing small values.

    Args:
        coeffs_data: Coefficients from wavelet_transform
        threshold: Values below this are zeroed
        keep_approximation: If True, don't threshold the approximation coeffs

    Returns:
        Modified coefficients dictionary
    """
    import copy
    result = copy.deepcopy(coeffs_data)

    if result["mode"] == "wpt":
        # WPT compression
        if result["is_3d"]:
            for c, wp in enumerate(result["coeffs"]):
                _compress_wpt(wp, threshold)
        else:
            _compress_wpt(result["coeffs"], threshold)
    else:
        # DWT compression
        if result["is_3d"]:
            for c in range(len(result["coeffs"])):
                result["coeffs"][c] = _compress_dwt(
                    result["coeffs"][c], threshold, keep_approximation
                )
        else:
            result["coeffs"] = _compress_dwt(
                result["coeffs"], threshold, keep_approximation
            )

    return result


def _compress_dwt(
    coeffs: list,
    threshold: float,
    keep_approximation: bool,
) -> list:
    """Compress DWT coefficients."""
    result = []
    for i, c in enumerate(coeffs):
        if i == 0 and keep_approximation:
            # Keep approximation coefficients unchanged
            result.append(c)
        elif isinstance(c, tuple):
            # Detail coefficients (LH, HL, HH)
            result.append(tuple(
                np.where(np.abs(arr) < threshold, 0, arr) for arr in c
            ))
        else:
            # Approximation or single array
            if keep_approximation and i == 0:
                result.append(c)
            else:
                result.append(np.where(np.abs(c) < threshold, 0, c))
    return result


def _compress_wpt(wp: pywt.WaveletPacket2D, threshold: float) -> None:
    """Compress WPT coefficients in-place."""
    for node in wp.get_level(wp.maxlevel, 'natural'):
        if node.data is not None:
            node.data = np.where(np.abs(node.data) < threshold, 0, node.data)


@register_node
class WaveletTransformNode(Node):
    """
    Perform forward wavelet transform on an image.

    Decomposes an image into wavelet coefficients at multiple scales.
    Supports 68+ wavelet families and two transform modes (FWT/WPT).

    The output coefficients can be modified (e.g., thresholded, scaled)
    and then reconstructed using InverseWaveletNode.
    """

    name = "Wavelet Transform"
    category = "Transform"
    description = "Forward wavelet transform"
    icon = "activity"
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
            port_type=PortType.ARRAY,
            description="Wavelet coefficients",
        )
        self.add_output(
            "visualization",
            port_type=PortType.IMAGE,
            description="Coefficient visualization",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        # Popular wavelets for quick selection
        wavelet_choices = [
            "haar", "db2", "db4", "db8",
            "sym2", "sym4", "sym8",
            "coif2", "coif4",
            "bior2.2", "bior4.4",
            "rbio2.2",
        ]
        self.add_parameter(
            "wavelet",
            param_type=ParameterType.ENUM,
            default="haar",
            choices=wavelet_choices,
            description="Wavelet type",
        )
        self.add_parameter(
            "mode",
            param_type=ParameterType.ENUM,
            default="fwt",
            choices=["fwt", "wpt"],
            description="Transform mode (FWT=standard, WPT=packet)",
        )
        self.add_parameter(
            "level",
            param_type=ParameterType.INT,
            default=3,
            min_value=1,
            max_value=10,
            description="Decomposition level",
        )

    def process(self) -> None:
        """Perform wavelet transform."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        wavelet = self.get_parameter("wavelet")
        mode = self.get_parameter("mode")
        level = self.get_parameter("level")

        coeffs = wavelet_transform(buffer.data, wavelet, mode, level)

        self.set_output_value("coefficients", coeffs)

        # Create visualization
        viz = self._create_visualization(coeffs, buffer)
        self.set_output_value("visualization", viz)

    def _create_visualization(
        self,
        coeffs_data: dict,
        buffer: ImageBuffer,
    ) -> ImageBuffer:
        """Create a visualization of wavelet coefficients."""
        # For simplicity, show the first channel's approximation
        if coeffs_data["mode"] == "wpt":
            # WPT - show the full reconstruction as grayscale
            if coeffs_data["is_3d"]:
                wp = coeffs_data["coeffs"][0]
            else:
                wp = coeffs_data["coeffs"]

            approx = wp['a' * wp.maxlevel].data
            if approx is not None:
                viz_data = approx.astype(np.float32)
                # Normalize to [0, 1]
                if viz_data.max() > viz_data.min():
                    viz_data = (viz_data - viz_data.min()) / (viz_data.max() - viz_data.min())

                # Expand to 3 channels
                h, w = viz_data.shape
                result = np.zeros((3, h, w), dtype=np.float32)
                result[0] = result[1] = result[2] = viz_data
            else:
                result = np.zeros((3, 8, 8), dtype=np.float32)
        else:
            # DWT - show approximation coefficients
            if coeffs_data["is_3d"]:
                coeffs = coeffs_data["coeffs"][0]  # First channel
            else:
                coeffs = coeffs_data["coeffs"]

            approx = coeffs[0]  # Approximation at lowest level
            if approx is not None:
                viz_data = approx.astype(np.float32)
                # Normalize to [0, 1]
                if viz_data.max() > viz_data.min():
                    viz_data = (viz_data - viz_data.min()) / (viz_data.max() - viz_data.min())

                h, w = viz_data.shape
                result = np.zeros((3, h, w), dtype=np.float32)
                result[0] = result[1] = result[2] = viz_data
            else:
                result = np.zeros((3, 8, 8), dtype=np.float32)

        return ImageBuffer(
            data=result,
            colorspace="RGB",
            metadata={"source": "wavelet_viz"},
        )


@register_node
class InverseWaveletNode(Node):
    """
    Perform inverse wavelet transform.

    Reconstructs an image from wavelet coefficients.
    """

    name = "Inverse Wavelet"
    category = "Transform"
    description = "Inverse wavelet transform"
    icon = "activity"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "coefficients",
            port_type=PortType.ARRAY,
            description="Wavelet coefficients",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Reconstructed image",
        )

    def process(self) -> None:
        """Perform inverse transform."""
        coeffs = self.get_input_value("coefficients")

        if coeffs is None:
            raise ValueError("No coefficients input")

        data = inverse_wavelet(coeffs)

        # Clip to valid range
        data = np.clip(data, 0.0, 1.0)

        result = ImageBuffer(
            data=data,
            colorspace="RGB",
            metadata={"reconstructed_from": "wavelet"},
        )

        self.set_output_value("image", result)


@register_node
class WaveletCompressNode(Node):
    """
    Compress wavelet coefficients by thresholding.

    Zeroes coefficients below a threshold, creating lossy compression
    with characteristic wavelet artifacts.
    """

    name = "Wavelet Compress"
    category = "Transform"
    description = "Threshold wavelet coefficients"
    icon = "minimize-2"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "coefficients",
            port_type=PortType.ARRAY,
            description="Wavelet coefficients",
            required=True,
        )
        self.add_output(
            "coefficients",
            port_type=PortType.ARRAY,
            description="Compressed coefficients",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "threshold",
            param_type=ParameterType.FLOAT,
            default=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Coefficient threshold",
        )
        self.add_parameter(
            "keep_approximation",
            param_type=ParameterType.BOOL,
            default=True,
            description="Keep approximation coefficients unchanged",
        )

    def process(self) -> None:
        """Compress coefficients."""
        coeffs = self.get_input_value("coefficients")

        if coeffs is None:
            raise ValueError("No coefficients input")

        threshold = self.get_parameter("threshold")
        keep_approx = self.get_parameter("keep_approximation")

        compressed = compress_coefficients(coeffs, threshold, keep_approx)

        self.set_output_value("coefficients", compressed)
