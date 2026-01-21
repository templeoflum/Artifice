"""
Tests for GLIC pipeline.

Validates V2.6 - GLIC Pipeline from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer, SegmentList
from artifice.nodes.pipeline.glic_pipeline import (
    GLICEncodeNode,
    GLICDecodeNode,
)


class TestGLICPipeline:
    """V2.6 - Test complete GLIC encode/decode pipeline."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        # Create a test pattern with gradients
        h, w = 64, 64
        data = np.zeros((3, h, w), dtype=np.float32)
        # Red gradient
        data[0] = np.linspace(0, 1, w)[np.newaxis, :]
        # Green gradient
        data[1] = np.linspace(0, 1, h)[:, np.newaxis]
        # Blue constant
        data[2] = 0.5
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def random_image(self):
        """Create random test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    def test_encode_produces_outputs(self, test_image):
        """Encode node produces all required outputs."""
        node = GLICEncodeNode()
        node.inputs["image"].default = test_image

        node.execute()

        # Check all outputs
        residuals = node.outputs["residuals"].get_value()
        segments = node.outputs["segments"].get_value()
        predicted = node.outputs["predicted"].get_value()
        quantized_int = node.outputs["quantized_int"].get_value()

        assert residuals is not None
        assert isinstance(residuals, ImageBuffer)

        assert segments is not None
        assert isinstance(segments, SegmentList)
        assert len(segments) > 0

        assert predicted is not None
        assert isinstance(predicted, ImageBuffer)

        assert quantized_int is not None

    def test_encode_decode_roundtrip(self, test_image):
        """V2.6 - Encode then decode recovers image (with quantization loss)."""
        # Encode
        encode_node = GLICEncodeNode()
        encode_node.inputs["image"].default = test_image
        encode_node.set_parameter("colorspace", "YCbCr")
        encode_node.set_parameter("predictor", "PAETH")
        encode_node.set_parameter("quantization_bits", 8)
        encode_node.execute()

        segments = encode_node.outputs["segments"].get_value()
        quantized_int = encode_node.outputs["quantized_int"].get_value()

        # Decode
        decode_node = GLICDecodeNode()
        decode_node.inputs["quantized_int"].default = quantized_int
        decode_node.inputs["segments"].default = segments
        decode_node.inputs["reference"].default = test_image
        decode_node.set_parameter("colorspace", "YCbCr")
        decode_node.set_parameter("predictor", "PAETH")
        decode_node.set_parameter("quantization_bits", 8)
        decode_node.set_parameter("output_colorspace", "RGB")
        decode_node.execute()

        result = decode_node.outputs["image"].get_value()

        # Should be close to original (with quantization error)
        # 8-bit quantization + colorspace conversion can introduce some error
        # Check mean error is low and most pixels are within tolerance
        error = np.abs(test_image.data - result.data)
        mean_error = error.mean()
        max_error = error.max()

        # Mean error should be very low
        assert mean_error < 0.02, f"Mean error too high: {mean_error}"
        # Max error should be reasonable (colorspace conversion can amplify quantization error)
        assert max_error < 0.4, f"Max error too high: {max_error}"
        # Most pixels should be within tight tolerance
        pixels_within_tolerance = (error < 0.1).mean()
        assert pixels_within_tolerance > 0.99, f"Too many outliers: {1 - pixels_within_tolerance:.2%}"

    def test_high_bit_depth_improves_quality(self, test_image):
        """Higher quantization bits produce better quality."""
        errors = []

        for bits in [4, 6, 8, 10, 12]:
            # Encode
            encode_node = GLICEncodeNode()
            encode_node.inputs["image"].default = test_image
            encode_node.set_parameter("quantization_bits", bits)
            encode_node.execute()

            segments = encode_node.outputs["segments"].get_value()
            quantized_int = encode_node.outputs["quantized_int"].get_value()

            # Decode
            decode_node = GLICDecodeNode()
            decode_node.inputs["quantized_int"].default = quantized_int
            decode_node.inputs["segments"].default = segments
            decode_node.set_parameter("quantization_bits", bits)
            decode_node.execute()

            result = decode_node.outputs["image"].get_value()
            error = np.abs(test_image.data - result.data).mean()
            errors.append(error)

        # Error should decrease with more bits
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], f"More bits didn't improve quality: {errors}"

    def test_different_colorspaces(self, test_image):
        """Pipeline works with different color spaces."""
        colorspaces = ["RGB", "YCbCr", "YUV", "LAB"]

        for cs in colorspaces:
            # Encode
            encode_node = GLICEncodeNode()
            encode_node.inputs["image"].default = test_image
            encode_node.set_parameter("colorspace", cs)
            encode_node.set_parameter("quantization_bits", 10)
            encode_node.execute()

            segments = encode_node.outputs["segments"].get_value()
            quantized_int = encode_node.outputs["quantized_int"].get_value()

            # Decode
            decode_node = GLICDecodeNode()
            decode_node.inputs["quantized_int"].default = quantized_int
            decode_node.inputs["segments"].default = segments
            decode_node.set_parameter("colorspace", cs)
            decode_node.set_parameter("quantization_bits", 10)
            decode_node.execute()

            result = decode_node.outputs["image"].get_value()
            assert result is not None, f"Colorspace {cs} failed"

            # Should have reasonable quality
            error = np.abs(test_image.data - result.data).mean()
            assert error < 0.1, f"Colorspace {cs} had high error: {error}"

    def test_different_predictors(self, test_image):
        """Pipeline works with different predictors."""
        predictors = ["PAETH", "H", "V", "DC", "CORNER", "MEDIAN"]

        for pred in predictors:
            # Encode
            encode_node = GLICEncodeNode()
            encode_node.inputs["image"].default = test_image
            encode_node.set_parameter("predictor", pred)
            encode_node.set_parameter("quantization_bits", 10)
            encode_node.execute()

            segments = encode_node.outputs["segments"].get_value()
            quantized_int = encode_node.outputs["quantized_int"].get_value()

            # Decode
            decode_node = GLICDecodeNode()
            decode_node.inputs["quantized_int"].default = quantized_int
            decode_node.inputs["segments"].default = segments
            decode_node.set_parameter("predictor", pred)
            decode_node.set_parameter("quantization_bits", 10)
            decode_node.execute()

            result = decode_node.outputs["image"].get_value()
            assert result is not None, f"Predictor {pred} failed"

    def test_segmentation_parameters(self, random_image):
        """Segmentation parameters affect output."""
        # Fine segmentation
        encode_fine = GLICEncodeNode()
        encode_fine.inputs["image"].default = random_image
        encode_fine.set_parameter("min_segment_size", 4)
        encode_fine.set_parameter("max_segment_size", 16)
        encode_fine.set_parameter("segment_threshold", 5.0)
        encode_fine.execute()
        segments_fine = encode_fine.outputs["segments"].get_value()

        # Coarse segmentation
        encode_coarse = GLICEncodeNode()
        encode_coarse.inputs["image"].default = random_image
        encode_coarse.set_parameter("min_segment_size", 16)
        encode_coarse.set_parameter("max_segment_size", 64)
        encode_coarse.set_parameter("segment_threshold", 50.0)
        encode_coarse.execute()
        segments_coarse = encode_coarse.outputs["segments"].get_value()

        # Fine should have more segments
        assert len(segments_fine) >= len(segments_coarse)

    def test_encode_metadata(self, test_image):
        """Encoded output contains metadata."""
        node = GLICEncodeNode()
        node.inputs["image"].default = test_image
        node.set_parameter("colorspace", "YCbCr")
        node.set_parameter("predictor", "PAETH")
        node.set_parameter("quantization_bits", 8)
        node.execute()

        residuals = node.outputs["residuals"].get_value()

        assert residuals.metadata.get("glic_encoded") == True
        assert residuals.metadata.get("colorspace") == "YCbCr"
        assert residuals.metadata.get("predictor") == "PAETH"
        assert residuals.metadata.get("quantization_bits") == 8


class TestGLICDecodeNode:
    """Additional tests for GLICDecodeNode."""

    def test_decode_without_reference(self):
        """Decode works without reference image."""
        # Create minimal test data
        quantized = np.zeros((3, 32, 32), dtype=np.int32)
        from artifice.core.data_types import Segment, SegmentList
        segments = SegmentList(width=32, height=32)
        segments.append(Segment(0, 0, 32))

        node = GLICDecodeNode()
        node.inputs["quantized_int"].default = quantized
        node.inputs["segments"].default = segments
        node.set_parameter("predictor", "CORNER")

        node.execute()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert result.colorspace == "RGB"  # Default output

    def test_output_colorspace_conversion(self):
        """Output can be converted to different colorspace."""
        # Create test data
        data = np.random.rand(3, 32, 32).astype(np.float32)
        buffer = ImageBuffer(data, colorspace="RGB")

        # Encode in YCbCr
        encode = GLICEncodeNode()
        encode.inputs["image"].default = buffer
        encode.set_parameter("colorspace", "YCbCr")
        encode.execute()

        segments = encode.outputs["segments"].get_value()
        quantized = encode.outputs["quantized_int"].get_value()

        # Decode to LAB
        decode = GLICDecodeNode()
        decode.inputs["quantized_int"].default = quantized
        decode.inputs["segments"].default = segments
        decode.set_parameter("colorspace", "YCbCr")
        decode.set_parameter("output_colorspace", "LAB")
        decode.execute()

        result = decode.outputs["image"].get_value()
        assert result.colorspace == "LAB"
