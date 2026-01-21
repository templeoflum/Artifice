"""
Tests for corruption operations.

Validates V3.5 from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.nodes.corruption.bit_ops import (
    BitShiftNode,
    BitFlipNode,
    ByteSwapNode,
    XORNoiseNode,
    bit_shift,
    bit_flip,
    byte_swap,
    xor_noise,
)
from artifice.nodes.corruption.data_ops import (
    DataRepeatNode,
    DataDropNode,
    DataWeaveNode,
    DataScrambleNode,
    data_repeat,
    data_drop,
    data_weave,
    data_scramble,
)


class TestBitOperations:
    """V3.5 - Test bit operations."""

    def test_bit_shift_left(self):
        """V3.5 - Bit shift left produces expected values."""
        data = np.array([0b10101010, 0b11110000], dtype=np.uint8)

        # Left shift by 2
        shifted = bit_shift(data, shift=2, direction="left", wrap=False)
        assert shifted[0] == 0b10101000
        assert shifted[1] == 0b11000000

    def test_bit_shift_right(self):
        """Bit shift right produces expected values."""
        data = np.array([0b10101010, 0b11110000], dtype=np.uint8)

        # Right shift by 2
        shifted = bit_shift(data, shift=2, direction="right", wrap=False)
        assert shifted[0] == 0b00101010
        assert shifted[1] == 0b00111100

    def test_bit_shift_wrap(self):
        """Bit shift with wrap rotates bits."""
        data = np.array([0b10000001], dtype=np.uint8)

        # Left shift by 1 with wrap
        shifted = bit_shift(data, shift=1, direction="left", wrap=True)
        assert shifted[0] == 0b00000011  # MSB wraps to LSB

        # Right shift by 1 with wrap
        shifted = bit_shift(data, shift=1, direction="right", wrap=True)
        assert shifted[0] == 0b11000000  # LSB wraps to MSB

    def test_bit_flip(self):
        """V3.5 - Bit flip toggles specific bit."""
        data = np.array([0b10101010, 0b11110000], dtype=np.uint8)

        # Flip LSB (bit 0)
        flipped = bit_flip(data, bit=0, probability=1.0)
        assert flipped[0] == 0b10101011  # LSB flipped
        assert flipped[1] == 0b11110001

        # Flip MSB (bit 7)
        flipped = bit_flip(data, bit=7, probability=1.0)
        assert flipped[0] == 0b00101010  # MSB flipped
        assert flipped[1] == 0b01110000

    def test_bit_flip_probability(self):
        """Bit flip probability affects number of flips."""
        data = np.ones((100,), dtype=np.uint8) * 0b10101010

        # With probability 0, nothing should change
        flipped = bit_flip(data.copy(), bit=0, probability=0.0)
        np.testing.assert_array_equal(data, flipped)

        # With probability 1, everything should change
        flipped = bit_flip(data.copy(), bit=0, probability=1.0)
        expected = np.ones((100,), dtype=np.uint8) * 0b10101011
        np.testing.assert_array_equal(expected, flipped)

    def test_byte_swap_adjacent(self):
        """Byte swap adjacent swaps pairs."""
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)

        swapped = byte_swap(data, swap_type="adjacent", stride=1)
        expected = np.array([2, 1, 4, 3, 6, 5], dtype=np.uint8)
        np.testing.assert_array_equal(swapped.ravel()[:6], expected)

    def test_byte_swap_reverse(self):
        """Byte swap reverse reverses groups."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)

        swapped = byte_swap(data, swap_type="reverse", stride=2)
        # Groups of 4 reversed
        expected = np.array([4, 3, 2, 1, 8, 7, 6, 5], dtype=np.uint8)
        np.testing.assert_array_equal(swapped.ravel()[:8], expected)

    def test_xor_noise(self):
        """XOR noise modifies data."""
        data = np.ones((64, 64), dtype=np.uint8) * 128

        result = xor_noise(data, noise_intensity=0.5, pattern="random")

        # Should be modified
        assert not np.allclose(data, result)

    def test_xor_noise_patterns(self):
        """XOR noise patterns produce different results."""
        data = np.ones((64, 64), dtype=np.uint8) * 128

        patterns = ["random", "stripes", "blocks", "gradient"]
        results = []

        for pattern in patterns:
            # Use fixed seed for reproducibility in random pattern
            if pattern == "random":
                np.random.seed(42)
            result = xor_noise(data.copy(), noise_intensity=0.5, pattern=pattern)
            results.append(result)

        # Different patterns should produce different results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if patterns[i] != "random" and patterns[j] != "random":
                    assert not np.allclose(results[i], results[j])


class TestBitOperationNodes:
    """Test bit operation nodes."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 32, 32).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    def test_bit_shift_node(self, test_image):
        """BitShiftNode works correctly."""
        node = BitShiftNode()
        node.inputs["image"].default = test_image
        node.set_parameter("shift", 2)
        node.set_parameter("direction", "left")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert isinstance(result, ImageBuffer)

    def test_bit_flip_node(self, test_image):
        """BitFlipNode works correctly."""
        node = BitFlipNode()
        node.inputs["image"].default = test_image
        node.set_parameter("bit", 4)
        node.set_parameter("probability", 0.5)

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None

    def test_byte_swap_node(self, test_image):
        """ByteSwapNode works correctly."""
        node = ByteSwapNode()
        node.inputs["image"].default = test_image
        node.set_parameter("swap_type", "adjacent")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None

    def test_xor_noise_node(self, test_image):
        """XORNoiseNode works correctly."""
        node = XORNoiseNode()
        node.inputs["image"].default = test_image
        node.set_parameter("intensity", 0.3)
        node.set_parameter("pattern", "stripes")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None


class TestDataOperations:
    """Test data corruption operations."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    def test_data_repeat(self):
        """Data repeat creates stuttering effect."""
        data = np.arange(100).reshape(10, 10).astype(np.float32)

        result = data_repeat(
            data,
            repeat_interval=5,
            repeat_length=2,
            direction="horizontal",
        )

        # Shape should be preserved
        assert result.shape == data.shape

        # Should be modified
        assert not np.allclose(data, result)

    def test_data_drop(self):
        """Data drop creates gaps."""
        data = np.arange(100).reshape(10, 10).astype(np.float32)

        result = data_drop(
            data,
            drop_interval=5,
            drop_length=2,
            direction="horizontal",
            fill_mode="black",
        )

        # Shape should be preserved
        assert result.shape == data.shape

        # Should have zeros (black fill)
        assert np.any(result == 0)

    def test_data_drop_fill_modes(self):
        """Data drop fill modes work correctly."""
        data = np.ones((10, 10), dtype=np.float32) * 128

        for fill_mode in ["shift", "black", "previous"]:
            result = data_drop(
                data.copy(),
                drop_interval=3,
                drop_length=1,
                fill_mode=fill_mode,
            )
            assert result.shape == data.shape

    def test_data_weave(self):
        """Data weave interleaves two images."""
        data1 = np.zeros((64, 64), dtype=np.float32)
        data2 = np.ones((64, 64), dtype=np.float32)

        result = data_weave(data1, data2, weave_width=4)

        # Should have both 0s and 1s
        assert np.any(result == 0)
        assert np.any(result == 1)

        # Shape should match
        assert result.shape == data1.shape

    def test_data_scramble(self):
        """Data scramble shuffles blocks."""
        data = np.arange(256).reshape(16, 16).astype(np.float32)

        np.random.seed(42)
        result = data_scramble(data, block_size=4, scramble_ratio=1.0)

        # Shape should be preserved
        assert result.shape == data.shape

        # Should be scrambled (different order)
        assert not np.allclose(data, result)


class TestDataOperationNodes:
    """Test data operation nodes."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def second_image(self):
        """Create second test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32) * 0.5
        return ImageBuffer(data, colorspace="RGB")

    def test_data_repeat_node(self, test_image):
        """DataRepeatNode works correctly."""
        node = DataRepeatNode()
        node.inputs["image"].default = test_image
        node.set_parameter("repeat_interval", 10)
        node.set_parameter("repeat_length", 3)
        node.set_parameter("direction", "horizontal")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert isinstance(result, ImageBuffer)

    def test_data_drop_node(self, test_image):
        """DataDropNode works correctly."""
        node = DataDropNode()
        node.inputs["image"].default = test_image
        node.set_parameter("drop_interval", 8)
        node.set_parameter("drop_length", 2)
        node.set_parameter("fill_mode", "black")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None

    def test_data_weave_node(self, test_image, second_image):
        """DataWeaveNode works correctly."""
        node = DataWeaveNode()
        node.inputs["image_a"].default = test_image
        node.inputs["image_b"].default = second_image
        node.set_parameter("weave_width", 8)
        node.set_parameter("direction", "horizontal")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None

    def test_data_scramble_node(self, test_image):
        """DataScrambleNode works correctly."""
        node = DataScrambleNode()
        node.inputs["image"].default = test_image
        node.set_parameter("block_size", 8)
        node.set_parameter("scramble_ratio", 0.5)

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
