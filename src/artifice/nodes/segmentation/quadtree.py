"""
Quadtree segmentation node.

Implements content-adaptive quadtree decomposition as used in GLIC.
Regions are subdivided based on standard deviation threshold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer, Segment, SegmentList
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def _next_power_of_two(n: int) -> int:
    """Round up to next power of two."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _calc_stddev_sampled(
    data: NDArray[np.float32],
    x: int,
    y: int,
    size: int,
    num_samples: int = None,
) -> float:
    """
    Calculate standard deviation of a region using random sampling.

    This matches GLIC's approach - sample ~10% of pixels for efficiency.

    Args:
        data: 2D array (single channel)
        x: Left column
        y: Top row
        size: Region size
        num_samples: Number of samples (default: max(4, 0.1 * size^2))

    Returns:
        Estimated standard deviation
    """
    h, w = data.shape

    if num_samples is None:
        num_samples = max(4, int(0.1 * size * size))

    # Clamp region to image bounds
    x_end = min(x + size, w)
    y_end = min(y + size, h)

    actual_width = x_end - x
    actual_height = y_end - y

    if actual_width <= 0 or actual_height <= 0:
        return 0.0

    # Use Welford's online algorithm for numerical stability
    # (Matches GLIC's implementation)
    A = 0.0
    Q = 0.0

    rng = np.random.default_rng()

    for k in range(1, num_samples + 1):
        px = x + rng.integers(0, actual_width)
        py = y + rng.integers(0, actual_height)

        val = float(data[py, px])

        old_A = A
        A += (val - A) / k
        Q += (val - old_A) * (val - A)

    if num_samples <= 1:
        return 0.0

    return np.sqrt(Q / (num_samples - 1))


def _calc_stddev_full(
    data: NDArray[np.float32],
    x: int,
    y: int,
    size: int,
) -> float:
    """
    Calculate exact standard deviation of a region.

    Slower but more accurate than sampled version.
    """
    h, w = data.shape

    x_end = min(x + size, w)
    y_end = min(y + size, h)

    if x_end <= x or y_end <= y:
        return 0.0

    region = data[y:y_end, x:x_end]
    return float(np.std(region))


def quadtree_segment(
    data: NDArray[np.float32],
    min_size: int = 4,
    max_size: int = 64,
    threshold: float = 10.0,
    use_sampling: bool = True,
    channel: int | None = None,
    per_channel: bool = False,
) -> SegmentList:
    """
    Perform quadtree segmentation on image data.

    Args:
        data: 2D array (H, W) or 3D array (C, H, W)
        min_size: Minimum segment size
        max_size: Maximum segment size
        threshold: Stddev threshold for subdivision (in pixel value units)
        use_sampling: Use random sampling for stddev (faster, matches GLIC)
        channel: Optional channel index to store in segments
        per_channel: Ignored for 2D input. For 3D, if False uses luminance.

    Returns:
        SegmentList containing all segments
    """
    # Handle 3D (CHW) input by extracting single channel for segmentation
    if data.ndim == 3:
        c, h, w = data.shape
        # Use luminance for RGB or first channel otherwise
        if c >= 3:
            segment_data = 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]
        else:
            segment_data = data[0]
    else:
        segment_data = data
        h, w = data.shape

    # Determine initial size (power of 2 covering the image)
    initial_size = _next_power_of_two(max(w, h))

    segments = SegmentList(width=w, height=h, channel=channel)

    # Normalize threshold to [0, 1] range if data is in [0, 1]
    # GLIC uses 0-255 range, so we scale
    if segment_data.max() <= 1.0:
        threshold_normalized = threshold / 255.0
    else:
        threshold_normalized = threshold

    calc_stddev = _calc_stddev_sampled if use_sampling else _calc_stddev_full

    def _segment_recursive(x: int, y: int, size: int) -> None:
        """Recursive segmentation."""
        # Skip if completely outside image
        if x >= w or y >= h:
            return

        # Calculate stddev for this region
        stddev = calc_stddev(segment_data, x, y, size)

        # Decide whether to subdivide
        should_subdivide = (
            size > max_size or
            (size > min_size and stddev > threshold_normalized)
        )

        if should_subdivide and size > min_size:
            # Subdivide into 4 quadrants
            mid = size // 2
            _segment_recursive(x, y, mid)
            _segment_recursive(x + mid, y, mid)
            _segment_recursive(x, y + mid, mid)
            _segment_recursive(x + mid, y + mid, mid)
        else:
            # Create leaf segment
            # Clip size to actual image bounds
            actual_w = min(size, w - x)
            actual_h = min(size, h - y)
            if actual_w > 0 and actual_h > 0:
                seg = Segment(x=x, y=y, size=size, channel=channel)
                segments.append(seg)

    _segment_recursive(0, 0, initial_size)

    return segments


def quadtree_segment_multichannel(
    data: NDArray[np.float32],
    min_size: int = 4,
    max_size: int = 64,
    threshold: float = 10.0,
    per_channel: bool = False,
) -> list[SegmentList]:
    """
    Segment a multi-channel image.

    Args:
        data: 3D array (C, H, W)
        min_size: Minimum segment size
        max_size: Maximum segment size
        threshold: Stddev threshold
        per_channel: If True, segment each channel independently.
                     If False, use luminance for all channels.

    Returns:
        List of SegmentList, one per channel (or same for all if not per_channel)
    """
    c, h, w = data.shape

    if per_channel:
        return [
            quadtree_segment(data[i], min_size, max_size, threshold, channel=i)
            for i in range(c)
        ]
    else:
        # Use luminance (or first channel)
        if c >= 3:
            luma = 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]
        else:
            luma = data[0]

        base_segments = quadtree_segment(luma, min_size, max_size, threshold)

        # Copy for each channel
        result = []
        for i in range(c):
            seg_copy = base_segments.copy()
            seg_copy.channel = i
            for seg in seg_copy:
                seg.channel = i
            result.append(seg_copy)

        return result


@register_node
class QuadtreeSegmentNode(Node):
    """
    Perform quadtree segmentation on an image.

    Divides the image into variable-sized square regions based on
    content complexity (measured by standard deviation). Areas with
    more detail get smaller segments; uniform areas get larger ones.

    This is the foundation of GLIC-style glitch effects - the segment
    boundaries become visible when prediction and quantization are applied.

    Parameters:
        min_size: Minimum segment size (power of 2 recommended)
        max_size: Maximum segment size (power of 2 recommended)
        threshold: Stddev threshold for subdivision
        per_channel: Segment each channel independently
    """

    name = "Quadtree Segment"
    category = "Segmentation"
    description = "Content-adaptive quadtree segmentation"
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
            "segments",
            port_type=PortType.SEGMENTS,
            description="Segment list",
        )
        self.add_output(
            "visualization",
            port_type=PortType.IMAGE,
            description="Segment boundaries visualization",
        )

    def define_parameters(self) -> None:
        """Define parameters."""
        self.add_parameter(
            "min_size",
            param_type=ParameterType.INT,
            default=4,
            min_value=2,
            max_value=64,
            description="Minimum segment size",
        )
        self.add_parameter(
            "max_size",
            param_type=ParameterType.INT,
            default=64,
            min_value=4,
            max_value=256,
            description="Maximum segment size",
        )
        self.add_parameter(
            "threshold",
            param_type=ParameterType.FLOAT,
            default=15.0,
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            description="Subdivision threshold (stddev)",
        )
        self.add_parameter(
            "per_channel",
            param_type=ParameterType.BOOL,
            default=False,
            description="Segment each channel independently",
        )

    def process(self) -> None:
        """Perform segmentation."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        min_size = self.get_parameter("min_size")
        max_size = self.get_parameter("max_size")
        threshold = self.get_parameter("threshold")
        per_channel = self.get_parameter("per_channel")

        # Ensure min <= max
        if min_size > max_size:
            min_size, max_size = max_size, min_size

        # Perform segmentation
        segment_lists = quadtree_segment_multichannel(
            buffer.data,
            min_size=min_size,
            max_size=max_size,
            threshold=threshold,
            per_channel=per_channel,
        )

        # Output first channel's segments (or could output all)
        self.set_output_value("segments", segment_lists[0])

        # Create visualization
        viz_data = self._create_visualization(buffer, segment_lists[0])
        viz_buffer = ImageBuffer(
            data=viz_data,
            colorspace=buffer.colorspace,
            metadata={"source": "segmentation_viz"},
        )
        self.set_output_value("visualization", viz_buffer)

    def _create_visualization(
        self,
        buffer: ImageBuffer,
        segments: SegmentList,
    ) -> NDArray[np.float32]:
        """Create a visualization of segment boundaries."""
        # Start with copy of original
        viz = buffer.data.copy()

        # Draw segment boundaries
        for seg in segments:
            # Top edge
            if seg.y > 0:
                viz[:, seg.y, seg.x:min(seg.x + seg.size, buffer.width)] = 1.0

            # Left edge
            if seg.x > 0:
                viz[:, seg.y:min(seg.y + seg.size, buffer.height), seg.x] = 1.0

        return viz
