"""
Tests for core data types: ImageBuffer, Segment, SegmentList.

Validates V1.2 - ImageBuffer Operations from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ColorSpace, ImageBuffer, Segment, SegmentList


class TestImageBuffer:
    """Tests for ImageBuffer class."""

    def test_creation_from_array(self):
        """Test creating ImageBuffer from numpy array."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        buf = ImageBuffer(data, colorspace=ColorSpace.RGB)

        assert buf.shape == (3, 64, 64)
        assert buf.channels == 3
        assert buf.height == 64
        assert buf.width == 64
        assert buf.colorspace == ColorSpace.RGB

    def test_creation_from_2d_array(self):
        """Test creating ImageBuffer from 2D (grayscale) array."""
        data = np.random.rand(64, 64).astype(np.float32)
        buf = ImageBuffer(data)

        assert buf.shape == (1, 64, 64)
        assert buf.channels == 1

    def test_dtype_conversion(self):
        """Test automatic conversion to float32."""
        data = np.random.rand(3, 64, 64).astype(np.float64)
        buf = ImageBuffer(data)

        assert buf.data.dtype == np.float32

    def test_border_value_default(self):
        """Test default border value is zeros."""
        buf = ImageBuffer(np.zeros((3, 10, 10), dtype=np.float32))

        assert buf.border_value == (0.0, 0.0, 0.0)

    def test_border_value_custom(self):
        """Test custom border value."""
        buf = ImageBuffer(
            np.zeros((3, 10, 10), dtype=np.float32),
            border_value=(1.0, 0.5, 0.0),
        )

        assert buf.border_value == (1.0, 0.5, 0.0)

    def test_get_in_bounds(self):
        """Test get() for in-bounds coordinates."""
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        buf = ImageBuffer(data)

        assert buf.get(0, 0, 0) == 0.0
        assert buf.get(0, 1, 2) == 6.0
        assert buf.get(0, 2, 3) == 11.0

    def test_get_out_of_bounds(self):
        """Test get() returns border_value for out-of-bounds."""
        data = np.ones((3, 10, 10), dtype=np.float32)
        buf = ImageBuffer(data, border_value=(0.5, 0.5, 0.5))

        # Out of bounds
        assert buf.get(0, -1, 0) == 0.5
        assert buf.get(0, 0, -1) == 0.5
        assert buf.get(0, 10, 0) == 0.5
        assert buf.get(0, 0, 10) == 0.5

    def test_get_region(self):
        """Test get_region() with border handling."""
        data = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        buf = ImageBuffer(data, border_value=(99.0,))

        # Fully in bounds
        region = buf.get_region(0, 1, 1, 2, 2)
        expected = np.array([[5, 6], [9, 10]], dtype=np.float32)
        np.testing.assert_array_equal(region, expected)

        # Partially out of bounds (top-left)
        region = buf.get_region(0, -1, -1, 2, 2)
        assert region[0, 0] == 99.0  # Border
        assert region[1, 1] == 0.0  # Actual data

    def test_set_single_pixel(self):
        """Test set() for single pixel."""
        buf = ImageBuffer(np.zeros((1, 4, 4), dtype=np.float32))
        buf.set(0, 1, 2, 0.5)

        assert buf.get(0, 1, 2) == 0.5

    def test_set_region(self):
        """Test set_region() with clipping."""
        buf = ImageBuffer(np.zeros((1, 4, 4), dtype=np.float32))
        values = np.ones((2, 2), dtype=np.float32)

        buf.set_region(0, 1, 1, values)

        assert buf.get(0, 1, 1) == 1.0
        assert buf.get(0, 2, 2) == 1.0
        assert buf.get(0, 0, 0) == 0.0

    def test_copy(self):
        """Test deep copy."""
        original = ImageBuffer(np.ones((3, 10, 10), dtype=np.float32))
        copy = original.copy()

        # Modify copy
        copy.data[0, 0, 0] = 999.0

        # Original unchanged
        assert original.data[0, 0, 0] == 1.0

    def test_clone_empty(self):
        """Test clone_empty creates zeroed buffer."""
        original = ImageBuffer(
            np.ones((3, 10, 10), dtype=np.float32),
            colorspace=ColorSpace.LAB,
            border_value=(0.1, 0.2, 0.3),
        )
        clone = original.clone_empty()

        assert clone.shape == original.shape
        assert clone.colorspace == original.colorspace
        assert clone.border_value == original.border_value
        assert clone.data.sum() == 0.0

    def test_hwc_conversion_roundtrip(self):
        """Test HWC format conversion round-trip."""
        hwc_data = np.random.rand(64, 64, 3).astype(np.float32)
        buf = ImageBuffer.from_hwc(hwc_data)

        recovered = buf.to_hwc()

        np.testing.assert_array_almost_equal(hwc_data, recovered)

    def test_uint8_conversion_roundtrip(self):
        """Test uint8 conversion round-trip."""
        # Use values that survive quantization
        hwc_data = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        buf = ImageBuffer.from_hwc(hwc_data)

        uint8 = buf.to_uint8()
        assert uint8.dtype == np.uint8
        assert uint8[0, 0, 0] == 0
        assert uint8[0, 0, 1] == 127  # 0.5 * 255 = 127.5, truncates to 127
        assert uint8[0, 0, 2] == 255

    def test_arithmetic_operations(self):
        """Test arithmetic operations between buffers."""
        a = ImageBuffer(np.ones((3, 4, 4), dtype=np.float32) * 2)
        b = ImageBuffer(np.ones((3, 4, 4), dtype=np.float32) * 3)

        # Addition
        c = a + b
        assert c.data[0, 0, 0] == 5.0

        # Subtraction
        c = b - a
        assert c.data[0, 0, 0] == 1.0

        # Multiplication
        c = a * b
        assert c.data[0, 0, 0] == 6.0

        # Division
        c = b / a
        assert c.data[0, 0, 0] == 1.5

    def test_scalar_arithmetic(self):
        """Test arithmetic with scalars."""
        buf = ImageBuffer(np.ones((3, 4, 4), dtype=np.float32) * 2)

        assert (buf + 1).data[0, 0, 0] == 3.0
        assert (buf - 1).data[0, 0, 0] == 1.0
        assert (buf * 2).data[0, 0, 0] == 4.0
        assert (buf / 2).data[0, 0, 0] == 1.0


class TestSegment:
    """Tests for Segment class."""

    def test_creation(self):
        """Test segment creation."""
        seg = Segment(x=10, y=20, size=32)

        assert seg.x == 10
        assert seg.y == 20
        assert seg.size == 32

    def test_properties(self):
        """Test computed properties."""
        seg = Segment(x=10, y=20, size=32)

        assert seg.x2 == 42
        assert seg.y2 == 52
        assert seg.center == (26, 36)
        assert seg.area == 1024

    def test_contains(self):
        """Test point containment."""
        seg = Segment(x=10, y=10, size=10)

        assert seg.contains(10, 10)  # Top-left
        assert seg.contains(15, 15)  # Center
        assert seg.contains(19, 19)  # Near bottom-right
        assert not seg.contains(20, 20)  # Outside (exclusive)
        assert not seg.contains(9, 10)  # Left of segment

    def test_overlaps(self):
        """Test segment overlap detection."""
        a = Segment(x=0, y=0, size=10)
        b = Segment(x=5, y=5, size=10)
        c = Segment(x=10, y=10, size=10)

        assert a.overlaps(b)  # Overlapping
        assert b.overlaps(a)
        assert not a.overlaps(c)  # Adjacent but not overlapping
        assert not c.overlaps(a)

    def test_copy(self):
        """Test segment copy."""
        original = Segment(x=10, y=20, size=32, pred_type=5)
        copy = original.copy()

        copy.x = 100
        assert original.x == 10


class TestSegmentList:
    """Tests for SegmentList class."""

    def test_creation(self):
        """Test segment list creation."""
        sl = SegmentList(width=64, height=64)

        assert len(sl) == 0
        assert sl.width == 64
        assert sl.height == 64

    def test_append_and_iterate(self):
        """Test adding and iterating segments."""
        sl = SegmentList(width=64, height=64)
        sl.append(Segment(0, 0, 32))
        sl.append(Segment(32, 0, 32))

        assert len(sl) == 2

        segments = list(sl)
        assert segments[0].x == 0
        assert segments[1].x == 32

    def test_find_at(self):
        """Test finding segment at point."""
        sl = SegmentList(width=64, height=64)
        sl.append(Segment(0, 0, 32))
        sl.append(Segment(32, 0, 32))

        found = sl.find_at(10, 10)
        assert found is not None
        assert found.x == 0

        found = sl.find_at(40, 10)
        assert found is not None
        assert found.x == 32

        found = sl.find_at(0, 50)
        assert found is None

    def test_coverage_mask(self):
        """Test coverage mask generation."""
        sl = SegmentList(width=8, height=8)
        sl.append(Segment(0, 0, 4))
        sl.append(Segment(4, 0, 4))
        sl.append(Segment(0, 4, 4))
        sl.append(Segment(4, 4, 4))

        mask = sl.get_coverage_mask()

        assert mask.shape == (8, 8)
        assert mask[0, 0] == 0  # First segment
        assert mask[0, 4] == 1  # Second segment
        assert mask[4, 0] == 2  # Third segment
        assert mask[4, 4] == 3  # Fourth segment

    def test_verify_coverage_valid(self):
        """Test coverage verification for valid segmentation."""
        sl = SegmentList(width=8, height=8)
        sl.append(Segment(0, 0, 4))
        sl.append(Segment(4, 0, 4))
        sl.append(Segment(0, 4, 4))
        sl.append(Segment(4, 4, 4))

        is_valid, msg = sl.verify_coverage()
        assert is_valid
        assert msg == "OK"

    def test_verify_coverage_gap(self):
        """Test coverage verification detects gaps."""
        sl = SegmentList(width=8, height=8)
        sl.append(Segment(0, 0, 4))
        # Missing segments

        is_valid, msg = sl.verify_coverage()
        assert not is_valid
        assert "uncovered" in msg

    def test_verify_coverage_overlap(self):
        """Test coverage verification detects overlap."""
        sl = SegmentList(width=8, height=8)
        sl.append(Segment(0, 0, 8))  # Full coverage
        sl.append(Segment(0, 0, 4))  # Overlapping

        is_valid, msg = sl.verify_coverage()
        assert not is_valid
        assert "multiple times" in msg
