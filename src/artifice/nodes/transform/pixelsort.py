"""
Pixel sorting node.

Classic glitch art effect that sorts pixels within rows or columns
based on various criteria (brightness, hue, saturation, etc.).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


class SortCriteria(Enum):
    """Criteria for determining pixel sort key."""
    BRIGHTNESS = "brightness"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    HUE = "hue"
    SATURATION = "saturation"
    VALUE = "value"
    ALPHA = "alpha"


class ThresholdMode(Enum):
    """Mode for threshold-based span detection."""
    BRIGHTNESS = "brightness"
    MASK = "mask"
    RANDOM = "random"
    NONE = "none"


def pixel_sort(
    data: NDArray[np.float32],
    threshold_low: float = 0.25,
    threshold_high: float = 0.8,
    direction: str = "horizontal",
    sort_by: str = "brightness",
    reverse: bool = False,
    threshold_mode: str = "brightness",
    mask: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """
    Apply pixel sorting effect to image data.

    Args:
        data: 3D (CHW) image array
        threshold_low: Lower threshold for span detection
        threshold_high: Upper threshold for span detection
        direction: 'horizontal' or 'vertical'
        sort_by: Criteria for sorting (brightness, red, green, blue, hue, saturation, value)
        reverse: Reverse sort order
        threshold_mode: How to detect spans ('brightness', 'mask', 'random', 'none')
        mask: Optional mask for 'mask' threshold mode

    Returns:
        Sorted image data
    """
    # Transpose for vertical sorting
    if direction == "vertical":
        data = np.transpose(data, (0, 2, 1))
        if mask is not None:
            mask = mask.T

    result = _sort_rows(
        data, threshold_low, threshold_high, sort_by, reverse, threshold_mode, mask
    )

    # Transpose back
    if direction == "vertical":
        result = np.transpose(result, (0, 2, 1))

    return result


def _sort_rows(
    data: NDArray[np.float32],
    threshold_low: float,
    threshold_high: float,
    sort_by: str,
    reverse: bool,
    threshold_mode: str,
    mask: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Sort pixels in each row."""
    c, h, w = data.shape
    result = data.copy()

    # Calculate sort key for each pixel
    sort_keys = _calculate_sort_keys(data, sort_by)

    # Calculate threshold values for span detection
    if threshold_mode == "brightness":
        threshold_values = _calculate_brightness(data)
    elif threshold_mode == "mask" and mask is not None:
        threshold_values = mask
    elif threshold_mode == "random":
        threshold_values = np.random.rand(h, w).astype(np.float32)
    else:  # none - sort entire rows
        threshold_values = np.ones((h, w), dtype=np.float32) * 0.5

    for y in range(h):
        # Find spans to sort based on threshold
        spans = _find_spans(threshold_values[y], threshold_low, threshold_high)

        for start, end in spans:
            if end - start > 1:
                # Get sort indices for this span
                span_keys = sort_keys[y, start:end]
                indices = np.argsort(span_keys)
                if reverse:
                    indices = indices[::-1]

                # Reorder pixels in span
                for channel in range(c):
                    result[channel, y, start:end] = result[channel, y, start:end][indices]

    return result


def _calculate_sort_keys(data: NDArray[np.float32], sort_by: str) -> NDArray[np.float32]:
    """Calculate sort key for each pixel."""
    c, h, w = data.shape

    if sort_by == "brightness":
        if c >= 3:
            return 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
        else:
            return data[0]
    elif sort_by == "red":
        return data[0]
    elif sort_by == "green":
        return data[1] if c > 1 else data[0]
    elif sort_by == "blue":
        return data[2] if c > 2 else data[0]
    elif sort_by == "hue":
        return _calculate_hue(data)
    elif sort_by == "saturation":
        return _calculate_saturation(data)
    elif sort_by == "value":
        return np.max(data, axis=0)
    else:
        return _calculate_brightness(data)


def _calculate_brightness(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate brightness for each pixel."""
    c = data.shape[0]
    if c >= 3:
        return 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
    else:
        return data[0]


def _calculate_hue(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate hue for each pixel (0-1)."""
    if data.shape[0] < 3:
        return np.zeros(data.shape[1:], dtype=np.float32)

    r, g, b = data[0], data[1], data[2]

    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val

    hue = np.zeros_like(max_val)

    # Red is max
    mask_r = (max_val == r) & (delta > 0)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6

    # Green is max
    mask_g = (max_val == g) & (delta > 0)
    hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2

    # Blue is max
    mask_b = (max_val == b) & (delta > 0)
    hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4

    return (hue / 6.0).astype(np.float32)


def _calculate_saturation(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate saturation for each pixel."""
    if data.shape[0] < 3:
        return np.zeros(data.shape[1:], dtype=np.float32)

    max_val = np.max(data[:3], axis=0)
    min_val = np.min(data[:3], axis=0)

    saturation = np.zeros_like(max_val)
    mask = max_val > 0
    saturation[mask] = (max_val[mask] - min_val[mask]) / max_val[mask]

    return saturation


def _find_spans(
    values: NDArray[np.float32],
    threshold_low: float,
    threshold_high: float,
) -> list[tuple[int, int]]:
    """Find spans of pixels to sort based on threshold."""
    w = len(values)
    spans = []

    in_span = False
    span_start = 0

    for x in range(w):
        # Check if pixel is within threshold range
        in_range = threshold_low <= values[x] <= threshold_high

        if in_range and not in_span:
            # Start new span
            span_start = x
            in_span = True
        elif not in_range and in_span:
            # End current span
            spans.append((span_start, x))
            in_span = False

    # Handle span that extends to end
    if in_span:
        spans.append((span_start, w))

    return spans


@register_node
class PixelSortNode(Node):
    """
    Apply pixel sorting glitch effect.

    Sorts pixels within rows or columns based on brightness or other
    criteria. Spans to sort are determined by threshold values.

    Classic glitch art technique that creates distinctive streaked patterns.
    """

    name = "Pixel Sort"
    category = "Transform"
    description = "Classic pixel sorting effect"
    icon = "sliders"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image",
            required=True,
        )
        self.add_input(
            "mask",
            port_type=PortType.MASK,
            description="Optional mask for threshold",
            required=False,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Sorted image",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "direction",
            param_type=ParameterType.ENUM,
            default="horizontal",
            choices=["horizontal", "vertical"],
            description="Sort direction",
        )
        self.add_parameter(
            "sort_by",
            param_type=ParameterType.ENUM,
            default="brightness",
            choices=["brightness", "red", "green", "blue", "hue", "saturation", "value"],
            description="Sort criteria",
        )
        self.add_parameter(
            "threshold_low",
            param_type=ParameterType.FLOAT,
            default=0.25,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Lower threshold for span detection",
        )
        self.add_parameter(
            "threshold_high",
            param_type=ParameterType.FLOAT,
            default=0.8,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Upper threshold for span detection",
        )
        self.add_parameter(
            "threshold_mode",
            param_type=ParameterType.ENUM,
            default="brightness",
            choices=["brightness", "mask", "random", "none"],
            description="How to detect spans",
        )
        self.add_parameter(
            "reverse",
            param_type=ParameterType.BOOL,
            default=False,
            description="Reverse sort order",
        )

    def process(self) -> None:
        """Apply pixel sorting."""
        buffer: ImageBuffer = self.get_input_value("image")
        mask = self.get_input_value("mask")

        if buffer is None:
            raise ValueError("No input image")

        direction = self.get_parameter("direction")
        sort_by = self.get_parameter("sort_by")
        threshold_low = self.get_parameter("threshold_low")
        threshold_high = self.get_parameter("threshold_high")
        threshold_mode = self.get_parameter("threshold_mode")
        reverse = self.get_parameter("reverse")

        # Get mask data if provided
        mask_data = None
        if mask is not None:
            if isinstance(mask, ImageBuffer):
                mask_data = mask.data[0]  # Use first channel
            else:
                mask_data = mask

        result_data = pixel_sort(
            buffer.data,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            direction=direction,
            sort_by=sort_by,
            reverse=reverse,
            threshold_mode=threshold_mode,
            mask=mask_data,
        )

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={
                **buffer.metadata,
                "pixel_sorted": True,
                "sort_direction": direction,
                "sort_by": sort_by,
            },
        )

        self.set_output_value("image", result)
