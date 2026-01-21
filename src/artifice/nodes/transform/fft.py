"""
Fast Fourier Transform (FFT) nodes.

FFT transforms image data to frequency domain with both magnitude
and phase information. Unlike DCT, FFT produces complex coefficients.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def fft_2d(
    data: NDArray[np.float32],
    shift: bool = True,
) -> dict[str, NDArray]:
    """
    Apply 2D FFT to image data.

    Args:
        data: 2D or 3D (CHW) array
        shift: Shift zero frequency to center

    Returns:
        Dictionary with 'magnitude', 'phase', and 'complex' arrays
    """
    if data.ndim == 3:
        channels = data.shape[0]
        mag_list = []
        phase_list = []
        complex_list = []

        for c in range(channels):
            result = _fft_2d_channel(data[c], shift)
            mag_list.append(result["magnitude"])
            phase_list.append(result["phase"])
            complex_list.append(result["complex"])

        return {
            "magnitude": np.stack(mag_list),
            "phase": np.stack(phase_list),
            "complex": np.stack(complex_list),
            "shift": shift,
            "shape": data.shape,
        }
    else:
        result = _fft_2d_channel(data, shift)
        result["shape"] = data.shape
        return result


def _fft_2d_channel(
    data: NDArray[np.float32],
    shift: bool,
) -> dict[str, NDArray]:
    """Apply FFT to a single 2D channel."""
    # Forward FFT
    fft_result = np.fft.fft2(data)

    if shift:
        fft_result = np.fft.fftshift(fft_result)

    # Separate magnitude and phase
    magnitude = np.abs(fft_result).astype(np.float32)
    phase = np.angle(fft_result).astype(np.float32)

    return {
        "magnitude": magnitude,
        "phase": phase,
        "complex": fft_result,
        "shift": shift,
    }


def ifft_2d(
    fft_data: dict[str, NDArray],
    use_magnitude_phase: bool = False,
) -> NDArray[np.float32]:
    """
    Apply inverse 2D FFT.

    Args:
        fft_data: Dictionary from fft_2d or with 'complex' key
        use_magnitude_phase: Reconstruct from magnitude and phase instead of complex

    Returns:
        Reconstructed image data
    """
    shift = fft_data.get("shift", True)

    if "complex" in fft_data and fft_data["complex"].ndim == 3:
        # Multi-channel
        channels = fft_data["complex"].shape[0]
        result = np.zeros(fft_data["shape"], dtype=np.float32)

        for c in range(channels):
            if use_magnitude_phase:
                complex_data = fft_data["magnitude"][c] * np.exp(
                    1j * fft_data["phase"][c]
                )
            else:
                complex_data = fft_data["complex"][c]

            result[c] = _ifft_2d_channel(complex_data, shift)

        return result
    else:
        if use_magnitude_phase:
            complex_data = fft_data["magnitude"] * np.exp(1j * fft_data["phase"])
        else:
            complex_data = fft_data["complex"]

        return _ifft_2d_channel(complex_data, shift)


def _ifft_2d_channel(
    complex_data: NDArray[np.complex128],
    shift: bool,
) -> NDArray[np.float32]:
    """Apply inverse FFT to a single channel."""
    if shift:
        complex_data = np.fft.ifftshift(complex_data)

    result = np.fft.ifft2(complex_data)
    return np.real(result).astype(np.float32)


def log_magnitude(magnitude: NDArray[np.float32], epsilon: float = 1e-10) -> NDArray[np.float32]:
    """
    Convert magnitude to log scale for visualization.

    Args:
        magnitude: FFT magnitude
        epsilon: Small value to avoid log(0)

    Returns:
        Log-scaled magnitude normalized to [0, 1]
    """
    log_mag = np.log(magnitude + epsilon)
    # Normalize to [0, 1]
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + epsilon)
    return log_mag.astype(np.float32)


def modify_magnitude(
    fft_data: dict[str, NDArray],
    scale: float = 1.0,
    high_pass: float = 0.0,
    low_pass: float = 1.0,
) -> dict[str, NDArray]:
    """
    Modify FFT magnitude with filtering.

    Args:
        fft_data: Dictionary from fft_2d
        scale: Scale factor for magnitude
        high_pass: High-pass filter radius (0-1, 0=none)
        low_pass: Low-pass filter radius (0-1, 1=none)

    Returns:
        Modified fft_data
    """
    import copy
    result = copy.deepcopy(fft_data)

    def filter_channel(mag: NDArray) -> NDArray:
        h, w = mag.shape
        cy, cx = h // 2, w // 2

        # Create distance from center
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)
        normalized_dist = distance / max_dist

        # Create filter
        mask = np.ones_like(mag)

        if high_pass > 0:
            mask *= (normalized_dist >= high_pass).astype(np.float32)

        if low_pass < 1.0:
            mask *= (normalized_dist <= low_pass).astype(np.float32)

        return mag * mask * scale

    if result["magnitude"].ndim == 3:
        for c in range(result["magnitude"].shape[0]):
            result["magnitude"][c] = filter_channel(result["magnitude"][c])
            # Update complex representation
            result["complex"][c] = result["magnitude"][c] * np.exp(
                1j * result["phase"][c]
            )
    else:
        result["magnitude"] = filter_channel(result["magnitude"])
        result["complex"] = result["magnitude"] * np.exp(1j * result["phase"])

    return result


@register_node
class FFTNode(Node):
    """
    Apply Fast Fourier Transform to an image.

    Transforms the image to frequency domain, producing magnitude
    and phase information. Useful for frequency filtering and analysis.
    """

    name = "FFT"
    category = "Transform"
    description = "Fast Fourier Transform"
    icon = "radio"
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
            "fft_data",
            port_type=PortType.ARRAY,
            description="FFT data (complex)",
        )
        self.add_output(
            "magnitude",
            port_type=PortType.IMAGE,
            description="Magnitude visualization",
        )
        self.add_output(
            "phase",
            port_type=PortType.IMAGE,
            description="Phase visualization",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "shift",
            param_type=ParameterType.BOOL,
            default=True,
            description="Shift zero frequency to center",
        )
        self.add_parameter(
            "log_scale",
            param_type=ParameterType.BOOL,
            default=True,
            description="Use log scale for magnitude display",
        )

    def process(self) -> None:
        """Perform FFT."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        shift = self.get_parameter("shift")
        log_scale = self.get_parameter("log_scale")

        fft_result = fft_2d(buffer.data, shift)

        self.set_output_value("fft_data", fft_result)

        # Create magnitude visualization
        mag = fft_result["magnitude"]
        if log_scale:
            mag_viz = log_magnitude(mag)
        else:
            # Normalize to [0, 1]
            mag_viz = mag / (mag.max() + 1e-10)

        mag_buffer = ImageBuffer(
            data=mag_viz,
            colorspace="RGB",
            metadata={"source": "fft_magnitude"},
        )
        self.set_output_value("magnitude", mag_buffer)

        # Create phase visualization
        phase = fft_result["phase"]
        # Normalize phase from [-pi, pi] to [0, 1]
        phase_viz = (phase + np.pi) / (2 * np.pi)
        phase_buffer = ImageBuffer(
            data=phase_viz.astype(np.float32),
            colorspace="RGB",
            metadata={"source": "fft_phase"},
        )
        self.set_output_value("phase", phase_buffer)


@register_node
class InverseFFTNode(Node):
    """
    Apply inverse Fast Fourier Transform.

    Reconstructs an image from FFT data.
    """

    name = "Inverse FFT"
    category = "Transform"
    description = "Inverse Fast Fourier Transform"
    icon = "radio"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "fft_data",
            port_type=PortType.ARRAY,
            description="FFT data",
            required=True,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Reconstructed image",
        )

    def process(self) -> None:
        """Perform inverse FFT."""
        fft_data = self.get_input_value("fft_data")

        if fft_data is None:
            raise ValueError("No FFT data input")

        result_data = ifft_2d(fft_data)

        # Clip to valid range
        result_data = np.clip(result_data, 0.0, 1.0)

        result = ImageBuffer(
            data=result_data,
            colorspace="RGB",
            metadata={"reconstructed_from": "fft"},
        )

        self.set_output_value("image", result)


@register_node
class FFTFilterNode(Node):
    """
    Apply frequency domain filtering to FFT data.

    Supports high-pass, low-pass, and band-pass filtering.
    """

    name = "FFT Filter"
    category = "Transform"
    description = "Frequency domain filtering"
    icon = "filter"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "fft_data",
            port_type=PortType.ARRAY,
            description="FFT data",
            required=True,
        )
        self.add_output(
            "fft_data",
            port_type=PortType.ARRAY,
            description="Filtered FFT data",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "high_pass",
            param_type=ParameterType.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="High-pass cutoff (0=none)",
        )
        self.add_parameter(
            "low_pass",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Low-pass cutoff (1=none)",
        )
        self.add_parameter(
            "magnitude_scale",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            description="Magnitude scale factor",
        )

    def process(self) -> None:
        """Apply filter."""
        fft_data = self.get_input_value("fft_data")

        if fft_data is None:
            raise ValueError("No FFT data input")

        high_pass = self.get_parameter("high_pass")
        low_pass = self.get_parameter("low_pass")
        scale = self.get_parameter("magnitude_scale")

        filtered = modify_magnitude(fft_data, scale, high_pass, low_pass)

        self.set_output_value("fft_data", filtered)
