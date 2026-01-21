"""
Tests for color space conversion.

Validates V2.1 - Color Space Round-Trip from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.nodes.color.conversions import (
    convert_colorspace,
    list_colorspaces,
    rgb_to_hsb, hsb_to_rgb,
    rgb_to_ycbcr, ycbcr_to_rgb,
    rgb_to_lab, lab_to_rgb,
    rgb_to_yuv, yuv_to_rgb,
)


class TestColorSpaceRoundTrip:
    """V2.1 - Test that color spaces convert and invert correctly."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with gradients."""
        h, w = 32, 32
        data = np.zeros((3, h, w), dtype=np.float32)
        # Red gradient
        data[0] = np.linspace(0, 1, w)[np.newaxis, :]
        # Green gradient
        data[1] = np.linspace(0, 1, h)[:, np.newaxis]
        # Blue constant
        data[2] = 0.5
        return data

    @pytest.mark.parametrize("colorspace", [
        "HSB", "YUV", "YCbCr", "LAB", "XYZ", "HCL",
        "LUV", "CMY", "HWB", "YDbDr", "YXY",
    ])
    def test_roundtrip(self, test_image, colorspace):
        """V2.1 - Standard color spaces convert and invert correctly."""
        original = test_image.copy()

        # Convert to colorspace
        converted = convert_colorspace(original, "RGB", colorspace)

        # Convert back
        restored = convert_colorspace(converted, colorspace, "RGB")

        # Should match within tolerance
        np.testing.assert_allclose(
            original, restored, atol=0.02,
            err_msg=f"{colorspace} failed roundtrip"
        )

    @pytest.mark.parametrize("colorspace", [
        "YPbPr", "OHTA", "R-GGB-G", "GREY"
    ])
    def test_roundtrip_lossy(self, test_image, colorspace):
        """Wrap-around and lossy color spaces may not perfectly round-trip.

        These spaces use modular arithmetic for compression efficiency (matching GLIC).
        They should still produce valid output in the right shape.
        """
        original = test_image.copy()

        # Convert to colorspace
        converted = convert_colorspace(original, "RGB", colorspace)

        # Convert back
        restored = convert_colorspace(converted, colorspace, "RGB")

        # Shape should match
        assert restored.shape == original.shape
        # Values should be in valid range
        assert restored.min() >= -0.01
        assert restored.max() <= 1.01

    def test_identity_conversion(self, test_image):
        """RGB to RGB should be identity."""
        result = convert_colorspace(test_image, "RGB", "RGB")
        np.testing.assert_array_equal(test_image, result)

    def test_hsb_roundtrip_specific(self):
        """Test HSB with known values."""
        # Pure red
        rgb = np.array([[[1.0]], [[0.0]], [[0.0]]], dtype=np.float32)
        hsb = rgb_to_hsb(rgb)

        # Hue should be 0, Saturation 1, Brightness 1
        assert abs(hsb[0, 0, 0]) < 0.01 or abs(hsb[0, 0, 0] - 1.0) < 0.01  # H=0 or 1
        assert abs(hsb[1, 0, 0] - 1.0) < 0.01  # S=1
        assert abs(hsb[2, 0, 0] - 1.0) < 0.01  # B=1

        # Convert back
        rgb_back = hsb_to_rgb(hsb)
        np.testing.assert_allclose(rgb, rgb_back, atol=0.01)

    def test_ycbcr_roundtrip_specific(self):
        """Test YCbCr with known values."""
        # Mid grey
        rgb = np.array([[[0.5]], [[0.5]], [[0.5]]], dtype=np.float32)
        ycbcr = rgb_to_ycbcr(rgb)

        # Y should be ~0.5, Cb and Cr should be ~0.5 (neutral)
        assert abs(ycbcr[0, 0, 0] - 0.5) < 0.1
        assert abs(ycbcr[1, 0, 0] - 0.5) < 0.1
        assert abs(ycbcr[2, 0, 0] - 0.5) < 0.1

        # Convert back
        rgb_back = ycbcr_to_rgb(ycbcr)
        np.testing.assert_allclose(rgb, rgb_back, atol=0.01)

    def test_lab_roundtrip_specific(self):
        """Test LAB with known values."""
        # White
        rgb = np.array([[[1.0]], [[1.0]], [[1.0]]], dtype=np.float32)
        lab = rgb_to_lab(rgb)

        # L should be ~1.0 (100%), a and b should be ~0.5 (neutral)
        assert lab[0, 0, 0] > 0.9  # High L
        assert abs(lab[1, 0, 0] - 0.5) < 0.1  # a neutral
        assert abs(lab[2, 0, 0] - 0.5) < 0.1  # b neutral

        # Convert back
        rgb_back = lab_to_rgb(lab)
        np.testing.assert_allclose(rgb, rgb_back, atol=0.02)

    def test_values_in_range(self, test_image):
        """Converted values should stay in [0, 1] range."""
        for cs in list_colorspaces():
            converted = convert_colorspace(test_image, "RGB", cs)
            assert converted.min() >= -0.01, f"{cs} has negative values"
            assert converted.max() <= 1.01, f"{cs} exceeds 1.0"


class TestColorSpaceNode:
    """Tests for ColorSpaceNode."""

    def test_node_conversion(self):
        """Test conversion through node."""
        from artifice.nodes.color.colorspace import ColorSpaceNode

        # Create test image
        data = np.random.rand(3, 32, 32).astype(np.float32)
        buffer = ImageBuffer(data, colorspace="RGB")

        node = ColorSpaceNode()
        node.inputs["image"].default = buffer
        node.set_parameter("target_space", "YCbCr")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert result.colorspace == "YCbCr"
        assert result.shape == buffer.shape


class TestChannelOps:
    """Tests for channel operation nodes."""

    def test_split_merge_roundtrip(self):
        """Split then merge should preserve image."""
        from artifice.nodes.color.channel_ops import ChannelSplitNode, ChannelMergeNode

        data = np.random.rand(3, 32, 32).astype(np.float32)
        buffer = ImageBuffer(data, colorspace="RGB")

        # Split
        split_node = ChannelSplitNode()
        split_node.inputs["image"].default = buffer
        split_node.execute()

        ch0 = split_node.outputs["channel_0"].get_value()
        ch1 = split_node.outputs["channel_1"].get_value()
        ch2 = split_node.outputs["channel_2"].get_value()

        # Merge
        merge_node = ChannelMergeNode()
        merge_node.inputs["channel_0"].default = ch0
        merge_node.inputs["channel_1"].default = ch1
        merge_node.inputs["channel_2"].default = ch2
        merge_node.execute()

        result = merge_node.outputs["image"].get_value()

        np.testing.assert_array_equal(data, result.data)

    def test_channel_swap(self):
        """Test channel swapping."""
        from artifice.nodes.color.channel_ops import ChannelSwapNode

        # Create image with distinct channels
        data = np.zeros((3, 4, 4), dtype=np.float32)
        data[0] = 0.1  # R
        data[1] = 0.5  # G
        data[2] = 0.9  # B
        buffer = ImageBuffer(data, colorspace="RGB")

        node = ChannelSwapNode()
        node.inputs["image"].default = buffer
        # Swap R and B
        node.set_parameter("channel_0_source", "2")
        node.set_parameter("channel_2_source", "0")

        node.execute()

        result = node.outputs["image"].get_value()

        # R and B should be swapped
        assert result.data[0, 0, 0] == pytest.approx(0.9)
        assert result.data[2, 0, 0] == pytest.approx(0.1)
        assert result.data[1, 0, 0] == pytest.approx(0.5)  # G unchanged
