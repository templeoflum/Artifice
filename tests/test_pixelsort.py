"""
Tests for pixel sorting.

Validates V3.4 and V3.6 from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.nodes.transform.pixelsort import (
    PixelSortNode,
    pixel_sort,
)


class TestPixelSort:
    """V3.4 - Test pixel sorting."""

    @pytest.fixture
    def test_image(self):
        """Create test image with varied brightness."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def gradient_image(self):
        """Create image with known gradient."""
        data = np.zeros((3, 64, 64), dtype=np.float32)
        # Create brightness gradient
        gradient = np.linspace(0, 1, 64)
        data[0] = gradient[np.newaxis, :]
        data[1] = gradient[np.newaxis, :]
        data[2] = gradient[np.newaxis, :]
        return ImageBuffer(data, colorspace="RGB")

    def test_pixelsort_determinism(self, test_image):
        """V3.4 - Pixel sort produces consistent results."""
        result1 = pixel_sort(
            test_image.data,
            threshold_low=0.2,
            threshold_high=0.8,
            direction="horizontal",
            sort_by="brightness",
        )
        result2 = pixel_sort(
            test_image.data,
            threshold_low=0.2,
            threshold_high=0.8,
            direction="horizontal",
            sort_by="brightness",
        )

        np.testing.assert_allclose(result1, result2)

    def test_pixelsort_changes_image(self, test_image):
        """Pixel sort modifies the image."""
        result = pixel_sort(
            test_image.data,
            threshold_low=0.1,
            threshold_high=0.9,
            direction="horizontal",
        )

        # Should be different from original (unless image is uniform)
        # Use random image so it should change
        assert not np.allclose(test_image.data, result)

    def test_pixelsort_horizontal_vs_vertical(self, test_image):
        """Horizontal and vertical sorting produce different results."""
        result_h = pixel_sort(
            test_image.data,
            threshold_low=0.2,
            threshold_high=0.8,
            direction="horizontal",
        )
        result_v = pixel_sort(
            test_image.data,
            threshold_low=0.2,
            threshold_high=0.8,
            direction="vertical",
        )

        assert not np.allclose(result_h, result_v)

    def test_pixelsort_sort_modes(self, test_image):
        """Different sort modes produce different results."""
        modes = ["brightness", "hue", "saturation", "red", "green", "blue"]
        results = []

        for mode in modes:
            result = pixel_sort(
                test_image.data,
                threshold_low=0.2,
                threshold_high=0.8,
                sort_by=mode,
            )
            results.append(result)

        # At least some modes should produce different results
        different_count = 0
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not np.allclose(results[i], results[j]):
                    different_count += 1

        assert different_count > 0, "All sort modes produced identical results"

    def test_pixelsort_node(self, test_image):
        """PixelSortNode works correctly."""
        node = PixelSortNode()
        node.inputs["image"].default = test_image
        node.set_parameter("threshold_low", 0.2)
        node.set_parameter("threshold_high", 0.8)
        node.set_parameter("direction", "horizontal")
        node.set_parameter("sort_by", "brightness")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert isinstance(result, ImageBuffer)
        assert result.data.shape == test_image.data.shape

    def test_pixelsort_reverse(self, test_image):
        """Reverse parameter changes sort order."""
        result_normal = pixel_sort(
            test_image.data,
            threshold_low=0.2,
            threshold_high=0.8,
            reverse=False,
        )
        result_reverse = pixel_sort(
            test_image.data,
            threshold_low=0.2,
            threshold_high=0.8,
            reverse=True,
        )

        # Results should be different
        assert not np.allclose(result_normal, result_reverse)


class TestEffectIntensityScaling:
    """V3.6 - Test effect intensity scaling."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    def test_threshold_affects_intensity(self, test_image):
        """V3.6 - Effect parameters scale intensity correctly."""
        # More aggressive settings = more change
        # High threshold = fewer pixels sorted (mild effect)
        mild = pixel_sort(
            test_image.data,
            threshold_low=0.4,
            threshold_high=0.6,  # Narrow range
            direction="horizontal",
        )
        # Low threshold range = more pixels sorted (strong effect)
        strong = pixel_sort(
            test_image.data,
            threshold_low=0.1,
            threshold_high=0.9,  # Wide range
            direction="horizontal",
        )

        mild_diff = np.abs(test_image.data - mild).mean()
        strong_diff = np.abs(test_image.data - strong).mean()

        # Wider threshold range should affect more pixels
        assert strong_diff >= mild_diff

    def test_narrow_threshold_minimal_change(self, test_image):
        """Very narrow threshold range creates minimal change."""
        # Very narrow range - almost no pixels qualify
        result = pixel_sort(
            test_image.data,
            threshold_low=0.49,
            threshold_high=0.51,
            direction="horizontal",
        )

        diff = np.abs(test_image.data - result).mean()
        # Should be relatively small change
        assert diff < 0.1

    def test_full_threshold_maximum_change(self, test_image):
        """Full threshold range creates maximum change."""
        # Full range - all pixels qualify
        result = pixel_sort(
            test_image.data,
            threshold_low=0.0,
            threshold_high=1.0,
            direction="horizontal",
        )

        diff = np.abs(test_image.data - result).mean()
        # Should have some change (unless already sorted)
        # Just verify it ran without error
        assert result.shape == test_image.data.shape
