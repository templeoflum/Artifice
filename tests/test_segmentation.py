"""
Tests for quadtree segmentation.

Validates V2.2 - Quadtree Segmentation from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer, Segment, SegmentList
from artifice.nodes.segmentation.quadtree import (
    quadtree_segment,
    QuadtreeSegmentNode,
)


class TestQuadtreeSegmentation:
    """V2.2 - Test quadtree segmentation algorithm."""

    @pytest.fixture
    def uniform_image(self):
        """Create uniform image (should produce large segments)."""
        data = np.ones((3, 64, 64), dtype=np.float32) * 0.5
        return data

    @pytest.fixture
    def gradient_image(self):
        """Create gradient image (should produce smaller segments)."""
        data = np.zeros((3, 64, 64), dtype=np.float32)
        # Strong gradient
        data[0] = np.linspace(0, 1, 64)[np.newaxis, :]
        data[1] = np.linspace(0, 1, 64)[:, np.newaxis]
        data[2] = 0.5
        return data

    @pytest.fixture
    def mixed_image(self):
        """Create image with uniform and varied regions."""
        data = np.ones((3, 64, 64), dtype=np.float32) * 0.5
        # Add high-variance region in top-left quadrant
        noise = np.random.rand(32, 32).astype(np.float32)
        data[0, :32, :32] = noise
        data[1, :32, :32] = noise
        data[2, :32, :32] = noise
        return data

    def test_uniform_produces_large_segments(self, uniform_image):
        """V2.2 - Uniform areas produce larger segments."""
        segments = quadtree_segment(
            uniform_image,
            min_size=4,
            max_size=64,
            threshold=10.0,
        )

        # Should have few large segments for uniform image
        assert len(segments) <= 4  # At most 4 max-size segments

        # Segments should be at max size
        for seg in segments:
            assert seg.size >= 16  # Most should be large

    def test_varied_produces_small_segments(self, gradient_image):
        """V2.2 - Varied areas produce smaller segments."""
        segments = quadtree_segment(
            gradient_image,
            min_size=4,
            max_size=64,
            threshold=5.0,  # Lower threshold = more subdivision
        )

        # Should have more segments for gradient image
        assert len(segments) > 4

        # Should have some small segments
        small_segments = [s for s in segments if s.size <= 8]
        assert len(small_segments) > 0

    def test_coverage_complete(self, gradient_image):
        """V2.2 - Segments cover entire image without gaps."""
        segments = quadtree_segment(
            gradient_image,
            min_size=4,
            max_size=64,
            threshold=10.0,
        )

        # Verify coverage
        h, w = gradient_image.shape[1], gradient_image.shape[2]
        coverage = np.zeros((h, w), dtype=bool)

        for seg in segments:
            x_end = min(seg.x + seg.size, w)
            y_end = min(seg.y + seg.size, h)
            coverage[seg.y:y_end, seg.x:x_end] = True

        assert coverage.all(), "Segmentation has gaps"

    def test_no_overlap(self, gradient_image):
        """V2.2 - Segments don't overlap."""
        segments = quadtree_segment(
            gradient_image,
            min_size=4,
            max_size=64,
            threshold=10.0,
        )

        h, w = gradient_image.shape[1], gradient_image.shape[2]
        count = np.zeros((h, w), dtype=np.int32)

        for seg in segments:
            x_end = min(seg.x + seg.size, w)
            y_end = min(seg.y + seg.size, h)
            count[seg.y:y_end, seg.x:x_end] += 1

        assert count.max() == 1, "Segments overlap"

    def test_segment_sizes_power_of_two(self, gradient_image):
        """Segment sizes should be powers of 2."""
        segments = quadtree_segment(
            gradient_image,
            min_size=4,
            max_size=64,
            threshold=10.0,
        )

        for seg in segments:
            # Check if size is power of 2
            assert seg.size > 0
            assert (seg.size & (seg.size - 1)) == 0, f"Size {seg.size} not power of 2"

    def test_respects_min_size(self, gradient_image):
        """Segments respect minimum size."""
        min_size = 8
        segments = quadtree_segment(
            gradient_image,
            min_size=min_size,
            max_size=64,
            threshold=0.1,  # Very low threshold to force subdivision
        )

        for seg in segments:
            assert seg.size >= min_size

    def test_respects_max_size(self, uniform_image):
        """Segments respect maximum size."""
        max_size = 16
        segments = quadtree_segment(
            uniform_image,
            min_size=4,
            max_size=max_size,
            threshold=100.0,  # High threshold to minimize subdivision
        )

        for seg in segments:
            assert seg.size <= max_size

    def test_threshold_affects_subdivision(self):
        """Higher threshold produces fewer segments."""
        data = np.random.rand(3, 64, 64).astype(np.float32)

        segments_low = quadtree_segment(data, threshold=5.0)
        segments_high = quadtree_segment(data, threshold=50.0)

        assert len(segments_low) >= len(segments_high)

    def test_non_square_image(self):
        """Handle non-square images."""
        data = np.random.rand(3, 48, 80).astype(np.float32)

        segments = quadtree_segment(data, min_size=4, max_size=64, threshold=10.0)

        # Should cover entire image
        h, w = data.shape[1], data.shape[2]
        coverage = np.zeros((h, w), dtype=bool)

        for seg in segments:
            x_end = min(seg.x + seg.size, w)
            y_end = min(seg.y + seg.size, h)
            coverage[seg.y:y_end, seg.x:x_end] = True

        assert coverage.all()


class TestQuadtreeSegmentNode:
    """Tests for QuadtreeSegmentNode."""

    def test_node_execution(self):
        """Test node produces valid segments."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        buffer = ImageBuffer(data, colorspace="RGB")

        node = QuadtreeSegmentNode()
        node.inputs["image"].default = buffer
        node.set_parameter("min_size", 4)
        node.set_parameter("max_size", 64)
        node.set_parameter("threshold", 10.0)

        node.execute()

        segments = node.outputs["segments"].get_value()
        assert segments is not None
        assert isinstance(segments, SegmentList)
        assert len(segments) > 0

    def test_node_visualization_output(self):
        """Test visualization output."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        buffer = ImageBuffer(data, colorspace="RGB")

        node = QuadtreeSegmentNode()
        node.inputs["image"].default = buffer

        node.execute()

        viz = node.outputs["visualization"].get_value()
        assert viz is not None
        assert isinstance(viz, ImageBuffer)
        assert viz.shape == buffer.shape

    def test_node_parameters(self):
        """Test node parameter settings."""
        node = QuadtreeSegmentNode()

        assert "min_size" in node.parameters
        assert "max_size" in node.parameters
        assert "threshold" in node.parameters
        assert "per_channel" in node.parameters
