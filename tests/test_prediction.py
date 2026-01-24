"""
Tests for prediction system.

Validates V2.3 - Predictor Algorithms from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer, Segment, SegmentList
from artifice.nodes.prediction.predictors import (
    PredictorType,
    predict_segment,
    PREDICTOR_NAMES,
    CORE_PREDICTORS,
)
from artifice.nodes.prediction.predict_node import (
    PredictNode,
    ResidualNode,
    ReconstructNode,
)


class TestPredictors:
    """V2.3 - Test predictor algorithms."""

    @pytest.fixture
    def test_data(self):
        """Create test image data."""
        h, w = 32, 32
        data = np.zeros((h, w), dtype=np.float32)
        # Create a gradient pattern
        data[:] = np.linspace(0, 1, w)[np.newaxis, :]
        data += np.linspace(0, 0.5, h)[:, np.newaxis]
        data = np.clip(data, 0, 1)
        return data

    @pytest.fixture
    def test_segment(self):
        """Create a test segment."""
        return Segment(x=8, y=8, size=8)

    def test_all_predictors_exist(self):
        """All 16 GLIC predictors are implemented."""
        # Check that we have all 16 core predictor types
        assert len(CORE_PREDICTORS) == 16, f"Expected 16 core predictors, got {len(CORE_PREDICTORS)}"

        # Verify all expected predictor types exist
        expected_types = [
            PredictorType.NONE, PredictorType.CORNER, PredictorType.H, PredictorType.V,
            PredictorType.DC, PredictorType.DCMEDIAN, PredictorType.MEDIAN, PredictorType.AVG,
            PredictorType.TRUEMOTION, PredictorType.PAETH, PredictorType.LDIAG,
            PredictorType.HV, PredictorType.JPEGLS, PredictorType.DIFF,
            PredictorType.REF, PredictorType.ANGLE,
        ]
        for ptype in expected_types:
            assert ptype in CORE_PREDICTORS, f"Missing predictor type: {ptype}"
            assert ptype in PREDICTOR_NAMES, f"Missing predictor name for: {ptype}"

    def test_predictor_output_shape(self, test_data, test_segment):
        """V2.3 - Predictors output correct shape."""
        for pred_type in CORE_PREDICTORS:
            pred = predict_segment(test_data, test_segment, pred_type, border=0.0)
            assert pred.shape == (test_segment.size, test_segment.size)

    def test_predictor_output_range(self, test_data, test_segment):
        """Predictor outputs are in valid range."""
        for pred_type in CORE_PREDICTORS:
            pred = predict_segment(test_data, test_segment, pred_type, border=0.0)
            # Predictions should be bounded (though can exceed [0,1] slightly)
            assert not np.isnan(pred).any(), f"{pred_type} produced NaN"
            assert not np.isinf(pred).any(), f"{pred_type} produced Inf"

    def test_corner_predictor(self, test_data, test_segment):
        """CORNER predictor uses top-left value."""
        pred = predict_segment(test_data, test_segment, PredictorType.CORNER, 0.0)
        # Should be constant (top-left corner value)
        expected = test_data[test_segment.y - 1, test_segment.x - 1]
        np.testing.assert_allclose(pred, expected, atol=0.01)

    def test_h_predictor(self, test_data, test_segment):
        """H predictor uses left neighbor."""
        pred = predict_segment(test_data, test_segment, PredictorType.H, 0.0)
        # First column should use left neighbor column
        left_col = test_data[test_segment.y:test_segment.y+test_segment.size, test_segment.x-1]
        np.testing.assert_allclose(pred[:, 0], left_col, atol=0.01)

    def test_v_predictor(self, test_data, test_segment):
        """V predictor uses top neighbor."""
        pred = predict_segment(test_data, test_segment, PredictorType.V, 0.0)
        # First row should use top neighbor row
        top_row = test_data[test_segment.y-1, test_segment.x:test_segment.x+test_segment.size]
        np.testing.assert_allclose(pred[0, :], top_row, atol=0.01)

    def test_dc_predictor(self, test_data, test_segment):
        """DC predictor uses average of neighbors."""
        pred = predict_segment(test_data, test_segment, PredictorType.DC, 0.0)
        # Should be constant (average of border pixels)
        assert np.allclose(pred, pred[0, 0])

    def test_paeth_predictor(self, test_data, test_segment):
        """PAETH predictor produces valid output."""
        pred = predict_segment(test_data, test_segment, PredictorType.PAETH, 0.0)
        # Just verify it runs and produces reasonable output
        assert pred.shape == (test_segment.size, test_segment.size)
        assert not np.isnan(pred).any()

    def test_border_value_used(self):
        """Border value is used for out-of-bounds access."""
        data = np.ones((16, 16), dtype=np.float32) * 0.5
        seg = Segment(x=0, y=0, size=8)  # At edge
        border = 0.25

        pred = predict_segment(data, seg, PredictorType.CORNER, border)
        # Corner at (0,0) uses border value
        assert np.allclose(pred, border)

    def test_meta_predictor_sad(self, test_data, test_segment):
        """SAD meta-predictor selects best predictor."""
        pred = predict_segment(test_data, test_segment, PredictorType.SAD, 0.0)
        # Should produce valid output
        assert pred.shape == (test_segment.size, test_segment.size)
        assert not np.isnan(pred).any()

    def test_meta_predictor_bsad(self, test_data, test_segment):
        """BSAD meta-predictor selects worst predictor."""
        pred = predict_segment(test_data, test_segment, PredictorType.BSAD, 0.0)
        # Should produce valid output
        assert pred.shape == (test_segment.size, test_segment.size)
        assert not np.isnan(pred).any()

    def test_meta_predictor_random(self, test_data, test_segment):
        """RANDOM meta-predictor selects randomly."""
        pred = predict_segment(test_data, test_segment, PredictorType.RANDOM, 0.0)
        assert pred.shape == (test_segment.size, test_segment.size)


class TestPredictNode:
    """Tests for PredictNode."""

    @pytest.fixture
    def test_buffer(self):
        """Create test image buffer."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def test_segments(self):
        """Create test segments."""
        from artifice.nodes.segmentation.quadtree import quadtree_segment
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return quadtree_segment(data, min_size=8, max_size=32, threshold=20.0)

    def test_predict_node_execution(self, test_buffer, test_segments):
        """Test PredictNode produces output."""
        node = PredictNode()
        node.inputs["image"].default = test_buffer
        node.inputs["segments"].default = test_segments
        node.set_parameter("predictor", "PAETH")

        node.execute()

        result = node.outputs["predicted"].get_value()
        assert result is not None
        assert isinstance(result, ImageBuffer)
        assert result.shape == test_buffer.shape

    def test_predict_node_all_predictors(self, test_buffer, test_segments):
        """Test all predictors work through node."""
        predictors = ["PAETH", "H", "V", "DC", "CORNER", "MEDIAN"]

        for pred_name in predictors:
            node = PredictNode()
            node.inputs["image"].default = test_buffer
            node.inputs["segments"].default = test_segments
            node.set_parameter("predictor", pred_name)

            node.execute()
            result = node.outputs["predicted"].get_value()
            assert result is not None, f"Predictor {pred_name} failed"


class TestResidualNode:
    """Tests for ResidualNode."""

    def test_residual_calculation(self):
        """V2.4 - Residual = actual - predicted."""
        actual_data = np.array([[[0.5, 0.6], [0.4, 0.5]]], dtype=np.float32)
        predicted_data = np.array([[[0.4, 0.5], [0.4, 0.4]]], dtype=np.float32)

        actual = ImageBuffer(actual_data)
        predicted = ImageBuffer(predicted_data)

        node = ResidualNode()
        node.inputs["actual"].default = actual
        node.inputs["predicted"].default = predicted

        node.execute()

        result = node.outputs["residual"].get_value()
        expected = actual_data - predicted_data

        np.testing.assert_allclose(result.data, expected)

    def test_residual_clamp_none(self):
        """CLAMP_NONE keeps full residual range including negatives."""
        actual_data = np.array([[[0.2, 0.8]]], dtype=np.float32)
        predicted_data = np.array([[[0.5, 0.3]]], dtype=np.float32)

        actual = ImageBuffer(actual_data)
        predicted = ImageBuffer(predicted_data)

        node = ResidualNode()
        node.inputs["actual"].default = actual
        node.inputs["predicted"].default = predicted
        node.set_parameter("clamp_method", "NONE")

        node.execute()

        result = node.outputs["residual"].get_value()
        # Raw residuals: 0.2-0.5=-0.3, 0.8-0.3=0.5
        expected = np.array([[[-0.3, 0.5]]], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-6)

    def test_residual_clamp_mod256(self):
        """CLAMP_MOD256 wraps residuals to 0-1 range."""
        actual_data = np.array([[[0.2, 0.8]]], dtype=np.float32)
        predicted_data = np.array([[[0.5, 0.3]]], dtype=np.float32)

        actual = ImageBuffer(actual_data)
        predicted = ImageBuffer(predicted_data)

        node = ResidualNode()
        node.inputs["actual"].default = actual
        node.inputs["predicted"].default = predicted
        node.set_parameter("clamp_method", "MOD256")

        node.execute()

        result = node.outputs["residual"].get_value()
        # Raw residuals: -0.3, 0.5
        # MOD256: np.mod(-0.3, 1.0) = 0.7, np.mod(0.5, 1.0) = 0.5
        expected = np.array([[[0.7, 0.5]]], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-6)

    def test_residual_metadata(self):
        """Residual output has correct metadata."""
        actual = ImageBuffer(np.zeros((3, 8, 8), dtype=np.float32))
        predicted = ImageBuffer(np.zeros((3, 8, 8), dtype=np.float32))

        node = ResidualNode()
        node.inputs["actual"].default = actual
        node.inputs["predicted"].default = predicted

        node.execute()

        result = node.outputs["residual"].get_value()
        assert result.metadata.get("is_residual") == True


class TestReconstructNode:
    """Tests for ReconstructNode."""

    def test_reconstruction(self):
        """Reconstruction = residual + predicted."""
        residual_data = np.array([[[0.1, 0.1], [0.0, 0.1]]], dtype=np.float32)
        predicted_data = np.array([[[0.4, 0.5], [0.4, 0.4]]], dtype=np.float32)

        residual = ImageBuffer(residual_data)
        predicted = ImageBuffer(predicted_data)

        node = ReconstructNode()
        node.inputs["residual"].default = residual
        node.inputs["predicted"].default = predicted

        node.execute()

        result = node.outputs["reconstructed"].get_value()
        expected = np.clip(residual_data + predicted_data, 0.0, 1.0)

        np.testing.assert_allclose(result.data, expected)

    def test_reconstruction_clips_values(self):
        """Reconstruction clips to [0, 1]."""
        residual = ImageBuffer(np.ones((1, 4, 4), dtype=np.float32) * 0.5)
        predicted = ImageBuffer(np.ones((1, 4, 4), dtype=np.float32) * 0.8)

        node = ReconstructNode()
        node.inputs["residual"].default = residual
        node.inputs["predicted"].default = predicted

        node.execute()

        result = node.outputs["reconstructed"].get_value()
        assert result.data.max() <= 1.0

    def test_reconstruction_clamp_mod256(self):
        """Reconstruction with MOD256 unwraps residuals before adding."""
        # Residual was MOD256'd: original -0.3 became 0.7
        residual_data = np.array([[[0.7, 0.5]]], dtype=np.float32)
        predicted_data = np.array([[[0.5, 0.3]]], dtype=np.float32)

        residual = ImageBuffer(residual_data)
        predicted = ImageBuffer(predicted_data)

        node = ReconstructNode()
        node.inputs["residual"].default = residual
        node.inputs["predicted"].default = predicted
        node.set_parameter("clamp_method", "MOD256")

        node.execute()

        result = node.outputs["reconstructed"].get_value()
        # MOD256 unwrap: 0.7 > 0.5 means it was negative, so 0.7-1.0=-0.3
        # Then: -0.3 + 0.5 = 0.2, 0.5 + 0.3 = 0.8
        expected = np.array([[[0.2, 0.8]]], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-6)


class TestPredictionRoundTrip:
    """V2.4 - Test prediction/residual/reconstruction round-trip."""

    def test_roundtrip_exact(self):
        """Reconstruction recovers original (without quantization)."""
        # Create test image
        original_data = np.random.rand(3, 64, 64).astype(np.float32)
        original = ImageBuffer(original_data, colorspace="RGB")

        # Create segments
        from artifice.nodes.segmentation.quadtree import quadtree_segment
        segments = quadtree_segment(original_data, min_size=8, max_size=32)

        # Predict
        predict_node = PredictNode()
        predict_node.inputs["image"].default = original
        predict_node.inputs["segments"].default = segments
        predict_node.set_parameter("predictor", "PAETH")
        predict_node.execute()
        predicted = predict_node.outputs["predicted"].get_value()

        # Calculate residual
        residual_node = ResidualNode()
        residual_node.inputs["actual"].default = original
        residual_node.inputs["predicted"].default = predicted
        residual_node.execute()
        residual = residual_node.outputs["residual"].get_value()

        # Reconstruct
        reconstruct_node = ReconstructNode()
        reconstruct_node.inputs["residual"].default = residual
        reconstruct_node.inputs["predicted"].default = predicted
        reconstruct_node.execute()
        reconstructed = reconstruct_node.outputs["reconstructed"].get_value()

        # Should match original
        np.testing.assert_allclose(
            original.data, reconstructed.data, atol=1e-5,
            err_msg="Round-trip prediction failed"
        )

    def test_roundtrip_mod256_clamp(self):
        """Round-trip with MOD256 clamp method preserves original."""
        # Create test image
        original_data = np.random.rand(3, 64, 64).astype(np.float32)
        original = ImageBuffer(original_data, colorspace="RGB")

        # Create segments
        from artifice.nodes.segmentation.quadtree import quadtree_segment
        segments = quadtree_segment(original_data, min_size=8, max_size=32)

        # Predict
        predict_node = PredictNode()
        predict_node.inputs["image"].default = original
        predict_node.inputs["segments"].default = segments
        predict_node.set_parameter("predictor", "PAETH")
        predict_node.execute()
        predicted = predict_node.outputs["predicted"].get_value()

        # Calculate residual with MOD256
        residual_node = ResidualNode()
        residual_node.inputs["actual"].default = original
        residual_node.inputs["predicted"].default = predicted
        residual_node.set_parameter("clamp_method", "MOD256")
        residual_node.execute()
        residual = residual_node.outputs["residual"].get_value()

        # Verify residual is in [0, 1] range (MOD256 property)
        assert residual.data.min() >= 0.0, "MOD256 residual should be >= 0"
        assert residual.data.max() <= 1.0, "MOD256 residual should be <= 1"

        # Reconstruct with MOD256
        reconstruct_node = ReconstructNode()
        reconstruct_node.inputs["residual"].default = residual
        reconstruct_node.inputs["predicted"].default = predicted
        reconstruct_node.set_parameter("clamp_method", "MOD256")
        reconstruct_node.execute()
        reconstructed = reconstruct_node.outputs["reconstructed"].get_value()

        # Should match original
        np.testing.assert_allclose(
            original.data, reconstructed.data, atol=1e-5,
            err_msg="Round-trip MOD256 prediction failed"
        )
