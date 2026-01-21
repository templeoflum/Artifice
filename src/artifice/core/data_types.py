"""
Core data types for Artifice Engine.

Provides ImageBuffer (multi-channel float arrays with border handling),
Segment, and SegmentList for quadtree-based region processing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


class ColorSpace(str, Enum):
    """Supported color spaces."""

    RGB = "RGB"
    HSB = "HSB"
    HSV = "HSV"  # Alias for HSB
    HWB = "HWB"
    CMY = "CMY"
    YUV = "YUV"
    YCBCR = "YCbCr"
    YPBPR = "YPbPr"
    YDBDR = "YDbDr"
    XYZ = "XYZ"
    LAB = "LAB"
    LUV = "LUV"
    HCL = "HCL"
    YXY = "YXY"
    OHTA = "OHTA"
    RGGBG = "R-GGB-G"
    GREY = "GREY"
    GRAYSCALE = "GRAYSCALE"  # Alias for GREY


@dataclass
class ImageBuffer:
    """
    Multi-channel float32 image buffer with border handling.

    This is the primary data structure for passing image data between nodes.
    Inspired by GLIC's Planes class but optimized for NumPy operations.

    Attributes:
        data: The image data as a float32 array with shape (C, H, W)
        colorspace: Current color space of the data
        border_value: Value used for out-of-bounds access (per channel)
        metadata: Optional metadata dict for carrying auxiliary info

    Shape Convention:
        - Channel-first format: (channels, height, width)
        - Typical shapes: (3, H, W) for color, (1, H, W) for grayscale
        - Float32 values typically in [0, 1] range for RGB, but can vary by colorspace
    """

    data: NDArray[np.float32]
    colorspace: ColorSpace | str = ColorSpace.RGB
    border_value: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the buffer after creation."""
        # Ensure float32
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

        # Ensure 3D (C, H, W)
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, :, :]
        elif self.data.ndim != 3:
            raise ValueError(f"ImageBuffer data must be 2D or 3D, got {self.data.ndim}D")

        # Normalize colorspace to enum if string
        if isinstance(self.colorspace, str):
            try:
                self.colorspace = ColorSpace(self.colorspace)
            except ValueError:
                # Keep as string for custom colorspaces
                pass

        # Set default border value if not provided
        if self.border_value is None:
            self.border_value = tuple(0.0 for _ in range(self.channels))
        elif len(self.border_value) != self.channels:
            raise ValueError(
                f"border_value length ({len(self.border_value)}) "
                f"must match channels ({self.channels})"
            )

    @property
    def channels(self) -> int:
        """Number of channels."""
        return self.data.shape[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as (channels, height, width)."""
        return self.data.shape  # type: ignore

    @property
    def size(self) -> tuple[int, int]:
        """Size as (width, height) - common image convention."""
        return (self.width, self.height)

    def get(self, channel: int, y: int, x: int) -> float:
        """
        Get pixel value with border handling.

        Like GLIC's RefColor system - returns border_value for out-of-bounds access.

        Args:
            channel: Channel index
            y: Row (y coordinate)
            x: Column (x coordinate)

        Returns:
            Pixel value, or border_value if out of bounds
        """
        if 0 <= y < self.height and 0 <= x < self.width and 0 <= channel < self.channels:
            return float(self.data[channel, y, x])
        elif 0 <= channel < self.channels:
            return self.border_value[channel]
        else:
            return 0.0

    def get_region(
        self, channel: int, y: int, x: int, height: int, width: int
    ) -> NDArray[np.float32]:
        """
        Get a rectangular region with border handling.

        Useful for prediction algorithms that need neighbor access.

        Args:
            channel: Channel index
            y: Top-left row
            x: Top-left column
            height: Region height
            width: Region width

        Returns:
            2D array of the requested region with border fill
        """
        result = np.full((height, width), self.border_value[channel], dtype=np.float32)

        # Calculate valid source region
        src_y_start = max(0, y)
        src_y_end = min(self.height, y + height)
        src_x_start = max(0, x)
        src_x_end = min(self.width, x + width)

        # Calculate destination offsets
        dst_y_start = src_y_start - y
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = src_x_start - x
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy valid region
        if src_y_end > src_y_start and src_x_end > src_x_start:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = self.data[
                channel, src_y_start:src_y_end, src_x_start:src_x_end
            ]

        return result

    def set(self, channel: int, y: int, x: int, value: float) -> None:
        """Set a single pixel value (no-op if out of bounds)."""
        if 0 <= y < self.height and 0 <= x < self.width and 0 <= channel < self.channels:
            self.data[channel, y, x] = value

    def set_region(
        self, channel: int, y: int, x: int, values: NDArray[np.float32]
    ) -> None:
        """
        Set a rectangular region (clips to bounds).

        Args:
            channel: Channel index
            y: Top-left row
            x: Top-left column
            values: 2D array of values to set
        """
        h, w = values.shape

        # Calculate valid destination region
        dst_y_start = max(0, y)
        dst_y_end = min(self.height, y + h)
        dst_x_start = max(0, x)
        dst_x_end = min(self.width, x + w)

        # Calculate source offsets
        src_y_start = dst_y_start - y
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        src_x_start = dst_x_start - x
        src_x_end = src_x_start + (dst_x_end - dst_x_start)

        # Copy valid region
        if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
            self.data[channel, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = values[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]

    def copy(self) -> ImageBuffer:
        """Create a deep copy of this buffer."""
        return ImageBuffer(
            data=self.data.copy(),
            colorspace=self.colorspace,
            border_value=self.border_value,
            metadata=copy.deepcopy(self.metadata),
        )

    def clone_empty(self) -> ImageBuffer:
        """Create a zeroed buffer with same shape and properties."""
        return ImageBuffer(
            data=np.zeros_like(self.data),
            colorspace=self.colorspace,
            border_value=self.border_value,
            metadata=copy.deepcopy(self.metadata),
        )

    def to_hwc(self) -> NDArray[np.float32]:
        """Convert to height-width-channel format (H, W, C) for OpenCV/PIL."""
        return np.transpose(self.data, (1, 2, 0))

    @classmethod
    def from_hwc(
        cls,
        data: NDArray,
        colorspace: ColorSpace | str = ColorSpace.RGB,
        border_value: tuple[float, ...] | None = None,
    ) -> ImageBuffer:
        """Create from height-width-channel format (H, W, C)."""
        if data.ndim == 2:
            chw_data = data[np.newaxis, :, :]
        else:
            chw_data = np.transpose(data, (2, 0, 1))
        return cls(data=chw_data, colorspace=colorspace, border_value=border_value)

    def to_uint8(self) -> NDArray[np.uint8]:
        """Convert to uint8 [0, 255] range, HWC format for display/saving."""
        hwc = self.to_hwc()
        clipped = np.clip(hwc * 255.0, 0, 255)
        return clipped.astype(np.uint8)

    @classmethod
    def from_uint8(
        cls,
        data: NDArray[np.uint8],
        colorspace: ColorSpace | str = ColorSpace.RGB,
        border_value: tuple[float, ...] | None = None,
    ) -> ImageBuffer:
        """Create from uint8 [0, 255] HWC data."""
        float_data = data.astype(np.float32) / 255.0
        return cls.from_hwc(float_data, colorspace, border_value)

    def __add__(self, other: ImageBuffer | float | NDArray) -> ImageBuffer:
        """Add two buffers or a scalar."""
        if isinstance(other, ImageBuffer):
            return ImageBuffer(
                data=self.data + other.data,
                colorspace=self.colorspace,
                border_value=self.border_value,
            )
        return ImageBuffer(
            data=self.data + other,
            colorspace=self.colorspace,
            border_value=self.border_value,
        )

    def __sub__(self, other: ImageBuffer | float | NDArray) -> ImageBuffer:
        """Subtract two buffers or a scalar."""
        if isinstance(other, ImageBuffer):
            return ImageBuffer(
                data=self.data - other.data,
                colorspace=self.colorspace,
                border_value=self.border_value,
            )
        return ImageBuffer(
            data=self.data - other,
            colorspace=self.colorspace,
            border_value=self.border_value,
        )

    def __mul__(self, other: ImageBuffer | float | NDArray) -> ImageBuffer:
        """Multiply two buffers or a scalar."""
        if isinstance(other, ImageBuffer):
            return ImageBuffer(
                data=self.data * other.data,
                colorspace=self.colorspace,
                border_value=self.border_value,
            )
        return ImageBuffer(
            data=self.data * other,
            colorspace=self.colorspace,
            border_value=self.border_value,
        )

    def __truediv__(self, other: ImageBuffer | float | NDArray) -> ImageBuffer:
        """Divide two buffers or a scalar."""
        if isinstance(other, ImageBuffer):
            return ImageBuffer(
                data=self.data / other.data,
                colorspace=self.colorspace,
                border_value=self.border_value,
            )
        return ImageBuffer(
            data=self.data / other,
            colorspace=self.colorspace,
            border_value=self.border_value,
        )

    def __repr__(self) -> str:
        return (
            f"ImageBuffer(shape={self.shape}, colorspace={self.colorspace}, "
            f"dtype={self.data.dtype})"
        )


@dataclass
class Segment:
    """
    A rectangular region from quadtree segmentation.

    Mirrors GLIC's Segment class - stores position, size, and metadata
    for prediction and processing.

    Attributes:
        x: Left column of segment
        y: Top row of segment
        size: Width and height (segments are square)
        pred_type: Index of predictor used (optional, set during prediction)
        angle: Angle for angle-based predictor (optional)
        ref_type: Reference region type for REF predictor (optional)
        ref_x: Reference x coordinate (optional)
        ref_y: Reference y coordinate (optional)
        channel: Which channel this segment belongs to (optional)
    """

    x: int
    y: int
    size: int
    pred_type: int | None = None
    angle: float | None = None
    ref_type: int | None = None
    ref_x: int | None = None
    ref_y: int | None = None
    channel: int | None = None

    @property
    def x2(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.size

    @property
    def y2(self) -> int:
        """Bottom edge (exclusive)."""
        return self.y + self.size

    @property
    def center(self) -> tuple[int, int]:
        """Center point of segment."""
        return (self.x + self.size // 2, self.y + self.size // 2)

    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.size * self.size

    def contains(self, x: int, y: int) -> bool:
        """Check if point is within segment."""
        return self.x <= x < self.x2 and self.y <= y < self.y2

    def overlaps(self, other: Segment) -> bool:
        """Check if two segments overlap."""
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def copy(self) -> Segment:
        """Create a copy of this segment."""
        return Segment(
            x=self.x,
            y=self.y,
            size=self.size,
            pred_type=self.pred_type,
            angle=self.angle,
            ref_type=self.ref_type,
            ref_x=self.ref_x,
            ref_y=self.ref_y,
            channel=self.channel,
        )

    def __repr__(self) -> str:
        return f"Segment(x={self.x}, y={self.y}, size={self.size})"


@dataclass
class SegmentList:
    """
    Collection of segments from quadtree decomposition.

    Provides iteration, lookup, and utility methods for working
    with segmented images.

    Attributes:
        segments: List of Segment objects
        width: Original image width
        height: Original image height
        channel: Channel index if channel-specific (optional)
    """

    segments: list[Segment] = field(default_factory=list)
    width: int = 0
    height: int = 0
    channel: int | None = None

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.segments)

    def __getitem__(self, index: int) -> Segment:
        return self.segments[index]

    def append(self, segment: Segment) -> None:
        """Add a segment to the list."""
        self.segments.append(segment)

    def extend(self, segments: Sequence[Segment]) -> None:
        """Add multiple segments."""
        self.segments.extend(segments)

    def find_at(self, x: int, y: int) -> Segment | None:
        """Find segment containing the given point."""
        for seg in self.segments:
            if seg.contains(x, y):
                return seg
        return None

    def get_coverage_mask(self) -> NDArray[np.int32]:
        """
        Create a mask showing segment indices at each pixel.

        Returns:
            2D array where each pixel contains the index of the
            segment covering it, or -1 if uncovered.
        """
        mask = np.full((self.height, self.width), -1, dtype=np.int32)
        for i, seg in enumerate(self.segments):
            mask[seg.y : seg.y2, seg.x : seg.x2] = i
        return mask

    def get_size_map(self) -> NDArray[np.int32]:
        """
        Create a map showing segment sizes at each pixel.

        Useful for visualization.
        """
        size_map = np.zeros((self.height, self.width), dtype=np.int32)
        for seg in self.segments:
            size_map[seg.y : seg.y2, seg.x : seg.x2] = seg.size
        return size_map

    def verify_coverage(self) -> tuple[bool, str]:
        """
        Verify that segments cover entire image without overlap.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.segments:
            return False, "No segments"

        coverage = np.zeros((self.height, self.width), dtype=np.int32)
        for seg in self.segments:
            coverage[seg.y : seg.y2, seg.x : seg.x2] += 1

        if coverage.min() == 0:
            uncovered = np.sum(coverage == 0)
            return False, f"{uncovered} pixels uncovered"

        if coverage.max() > 1:
            overlapping = np.sum(coverage > 1)
            return False, f"{overlapping} pixels covered multiple times"

        return True, "OK"

    def copy(self) -> SegmentList:
        """Create a deep copy."""
        return SegmentList(
            segments=[s.copy() for s in self.segments],
            width=self.width,
            height=self.height,
            channel=self.channel,
        )

    def __repr__(self) -> str:
        return f"SegmentList({len(self.segments)} segments, {self.width}x{self.height})"
