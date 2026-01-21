"""
Tests for quantization system.

Validates V2.5 - Quantization from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.nodes.quantization.quantize_node import (
    quantize_value,
    dequantize_value,
    QuantizeNode,
    DequantizeNode,
)


class TestQuantization:
    """V2.5 - Test quantization functions."""

    def test_quantize_unsigned_8bit(self):
        """8-bit unsigned quantization."""
        values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        quantized = quantize_value(values, bits=8, signed=False)

        assert quantized[0] == 0
        assert quantized[1] == 128  # 0.5 * 255 rounded
        assert quantized[2] == 255

    def test_quantize_signed_8bit(self):
        """8-bit signed quantization."""
        values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        quantized = quantize_value(values, bits=8, signed=True)

        # For 8-bit signed: max_val = 127, so -1.0 -> -127, 1.0 -> 127
        assert quantized[0] == -127
        assert quantized[1] == 0
        assert quantized[2] == 127

    def test_dequantize_unsigned_8bit(self):
        """8-bit unsigned dequantization."""
        quantized = np.array([0, 128, 255], dtype=np.int32)
        values = dequantize_value(quantized, bits=8, signed=False)

        np.testing.assert_allclose(values[0], 0.0, atol=0.01)
        np.testing.assert_allclose(values[1], 0.5, atol=0.01)
        np.testing.assert_allclose(values[2], 1.0, atol=0.01)

    def test_dequantize_signed_8bit(self):
        """8-bit signed dequantization."""
        quantized = np.array([-127, 0, 127], dtype=np.int32)
        values = dequantize_value(quantized, bits=8, signed=True)

        np.testing.assert_allclose(values[0], -1.0, atol=0.01)
        np.testing.assert_allclose(values[1], 0.0, atol=0.01)
        np.testing.assert_allclose(values[2], 1.0, atol=0.01)

    def test_roundtrip_preserves_values(self):
        """V2.5 - Quantize/dequantize round-trip."""
        original = np.random.rand(64, 64).astype(np.float32)

        for bits in [4, 6, 8, 10, 12]:
            quantized = quantize_value(original, bits=bits, signed=False)
            restored = dequantize_value(quantized, bits=bits, signed=False)

            # Error should decrease with more bits
            max_error = 1.0 / (2 ** bits)
            np.testing.assert_allclose(
                original, restored, atol=max_error * 2,
                err_msg=f"Round-trip failed at {bits} bits"
            )

    def test_roundtrip_signed(self):
        """Signed quantization round-trip."""
        original = (np.random.rand(64, 64).astype(np.float32) - 0.5) * 2  # [-1, 1]

        for bits in [4, 6, 8, 10]:
            quantized = quantize_value(original, bits=bits, signed=True)
            restored = dequantize_value(quantized, bits=bits, signed=True)

            max_error = 2.0 / (2 ** (bits - 1))  # Signed has half the range per side
            np.testing.assert_allclose(
                original, restored, atol=max_error * 2,
                err_msg=f"Signed round-trip failed at {bits} bits"
            )

    def test_clipping(self):
        """Out-of-range values are clipped."""
        values = np.array([-2.0, 0.5, 2.0], dtype=np.float32)

        # Unsigned
        quantized = quantize_value(values, bits=8, signed=False)
        assert quantized[0] == 0  # -2 clipped to 0
        assert quantized[2] == 255  # 2 clipped to 1

        # Signed: -2 clipped to -1 -> -127, 2 clipped to 1 -> 127
        quantized = quantize_value(values, bits=8, signed=True)
        assert quantized[0] == -127  # -2 clipped to -1 then quantized
        assert quantized[2] == 127  # 2 clipped to 1

    def test_bit_depth_range(self):
        """Various bit depths produce correct ranges."""
        value = np.array([1.0], dtype=np.float32)

        for bits in [1, 2, 4, 8, 12, 16]:
            quantized = quantize_value(value, bits=bits, signed=False)
            expected_max = (1 << bits) - 1
            assert quantized[0] == expected_max


class TestQuantizeNode:
    """Tests for QuantizeNode."""

    def test_node_execution(self):
        """Test QuantizeNode produces output."""
        data = np.random.rand(3, 32, 32).astype(np.float32)
        buffer = ImageBuffer(data)

        node = QuantizeNode()
        node.inputs["image"].default = buffer
        node.set_parameter("bits", 8)
        node.set_parameter("signed", False)

        node.execute()

        result = node.outputs["quantized"].get_value()
        assert result is not None
        assert isinstance(result, ImageBuffer)

        int_result = node.outputs["quantized_int"].get_value()
        assert int_result is not None
        assert int_result.dtype == np.int32

    def test_node_bit_depth_affects_quality(self):
        """Lower bit depth produces more quantization error."""
        data = np.random.rand(3, 32, 32).astype(np.float32)
        buffer = ImageBuffer(data)

        errors = []
        for bits in [4, 6, 8, 10]:
            node = QuantizeNode()
            node.inputs["image"].default = buffer
            node.set_parameter("bits", bits)
            node.set_parameter("signed", False)
            node.execute()

            result = node.outputs["quantized"].get_value()
            error = np.abs(data - result.data).mean()
            errors.append(error)

        # Error should decrease with more bits
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] * 0.5  # Each step roughly halves error

    def test_node_metadata(self):
        """Quantization metadata is preserved."""
        data = np.zeros((3, 8, 8), dtype=np.float32)
        buffer = ImageBuffer(data)

        node = QuantizeNode()
        node.inputs["image"].default = buffer
        node.set_parameter("bits", 6)
        node.execute()

        result = node.outputs["quantized"].get_value()
        assert result.metadata.get("quantization_bits") == 6


class TestDequantizeNode:
    """Tests for DequantizeNode."""

    def test_node_execution(self):
        """Test DequantizeNode produces output."""
        quantized = np.random.randint(-128, 128, (3, 32, 32), dtype=np.int32)

        node = DequantizeNode()
        node.inputs["quantized_int"].default = quantized
        node.set_parameter("bits", 8)
        node.set_parameter("signed", True)

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert isinstance(result, ImageBuffer)
        assert result.data.dtype == np.float32

    def test_node_roundtrip(self):
        """Quantize -> Dequantize through nodes."""
        data = np.random.rand(3, 32, 32).astype(np.float32)
        buffer = ImageBuffer(data)

        # Quantize
        q_node = QuantizeNode()
        q_node.inputs["image"].default = buffer
        q_node.set_parameter("bits", 8)
        q_node.set_parameter("signed", False)
        q_node.execute()

        quantized_int = q_node.outputs["quantized_int"].get_value()

        # Dequantize
        dq_node = DequantizeNode()
        dq_node.inputs["quantized_int"].default = quantized_int
        dq_node.set_parameter("bits", 8)
        dq_node.set_parameter("signed", False)
        dq_node.execute()

        result = dq_node.outputs["image"].get_value()

        # Should be close to original
        np.testing.assert_allclose(data, result.data, atol=0.01)
