"""
Prediction nodes for GLIC-style processing.

PredictNode: Generate predictions for segments
ResidualNode: Calculate residuals (actual - predicted)
ReconstructNode: Reconstruct from residuals + predictions
"""

import numpy as np

from artifice.core.data_types import ImageBuffer, SegmentList
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node
from artifice.nodes.prediction.predictors import (
    PREDICTOR_NAMES,
    PredictorType,
    predict_segment,
)


@register_node
class PredictNode(Node):
    """
    Generate predictions for image segments.

    Uses GLIC prediction algorithms to generate predicted values
    for each segment based on neighboring pixels. The predictions
    can then be subtracted to get residuals.

    Outputs the predicted image - each segment contains predicted values
    based on the selected predictor algorithm.
    """

    name = "Predict"
    category = "Prediction"
    description = "Generate predictions for segments"
    icon = "trending-up"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image",
            required=True,
        )
        self.add_input(
            "segments",
            port_type=PortType.SEGMENTS,
            description="Segment list",
            required=True,
        )
        self.add_output(
            "predicted",
            port_type=PortType.IMAGE,
            description="Predicted image",
        )

    def define_parameters(self) -> None:
        """Define predictor selection parameter."""
        predictor_choices = [
            "PAETH", "H", "V", "DC", "AVG", "MEDIAN",
            "DCMEDIAN", "CORNER", "TRUEMOTION", "LDIAG",
            "HV", "JPEGLS", "DIFF", "REF", "ANGLE",
            "SAD", "BSAD", "RANDOM", "NONE"
        ]
        self.add_parameter(
            "predictor",
            param_type=ParameterType.ENUM,
            default="PAETH",
            choices=predictor_choices,
            description="Prediction algorithm",
        )

    def process(self) -> None:
        """Generate predictions."""
        buffer: ImageBuffer = self.get_input_value("image")
        segments: SegmentList = self.get_input_value("segments")

        if buffer is None:
            raise ValueError("No input image")
        if segments is None:
            raise ValueError("No segments")

        predictor_name = self.get_parameter("predictor")

        # Map name to type
        name_to_type = {
            "NONE": PredictorType.NONE,
            "CORNER": PredictorType.CORNER,
            "H": PredictorType.H,
            "V": PredictorType.V,
            "DC": PredictorType.DC,
            "DCMEDIAN": PredictorType.DCMEDIAN,
            "MEDIAN": PredictorType.MEDIAN,
            "AVG": PredictorType.AVG,
            "TRUEMOTION": PredictorType.TRUEMOTION,
            "PAETH": PredictorType.PAETH,
            "LDIAG": PredictorType.LDIAG,
            "HV": PredictorType.HV,
            "JPEGLS": PredictorType.JPEGLS,
            "DIFF": PredictorType.DIFF,
            "REF": PredictorType.REF,
            "ANGLE": PredictorType.ANGLE,
            "SAD": PredictorType.SAD,
            "BSAD": PredictorType.BSAD,
            "RANDOM": PredictorType.RANDOM,
        }

        predictor = name_to_type.get(predictor_name, PredictorType.PAETH)

        # Process each channel
        result_data = np.zeros_like(buffer.data)

        for c in range(buffer.channels):
            data = buffer.data[c]
            h, w = data.shape
            border = buffer.border_value[c] if buffer.border_value else 0.0

            for seg in segments:
                pred = predict_segment(data, seg, predictor, border)

                # Place prediction
                x_end = min(seg.x + seg.size, w)
                y_end = min(seg.y + seg.size, h)
                pred_h = y_end - seg.y
                pred_w = x_end - seg.x

                result_data[c, seg.y:y_end, seg.x:x_end] = pred[:pred_h, :pred_w]

        result = ImageBuffer(
            data=result_data,
            colorspace=buffer.colorspace,
            border_value=buffer.border_value,
            metadata={
                **buffer.metadata,
                "predictor": predictor_name,
            },
        )

        self.set_output_value("predicted", result)


@register_node
class ResidualNode(Node):
    """
    Calculate residuals (actual - predicted).

    The residual is the difference between the actual image and
    the predicted image. This is what gets quantized and stored
    in GLIC-style compression.
    """

    name = "Residual"
    category = "Prediction"
    description = "Calculate prediction residuals"
    icon = "minus"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "actual",
            port_type=PortType.IMAGE,
            description="Actual/original image",
            required=True,
        )
        self.add_input(
            "predicted",
            port_type=PortType.IMAGE,
            description="Predicted image",
            required=True,
        )
        self.add_output(
            "residual",
            port_type=PortType.IMAGE,
            description="Residual image (actual - predicted)",
        )

    def process(self) -> None:
        """Calculate residuals."""
        actual: ImageBuffer = self.get_input_value("actual")
        predicted: ImageBuffer = self.get_input_value("predicted")

        if actual is None:
            raise ValueError("No actual image")
        if predicted is None:
            raise ValueError("No predicted image")

        if actual.shape != predicted.shape:
            raise ValueError("Image shapes must match")

        # Calculate residual
        residual_data = actual.data - predicted.data

        result = ImageBuffer(
            data=residual_data,
            colorspace=actual.colorspace,
            border_value=actual.border_value,
            metadata={
                **actual.metadata,
                "is_residual": True,
            },
        )

        self.set_output_value("residual", result)


@register_node
class ReconstructNode(Node):
    """
    Reconstruct image from residuals and predictions.

    Adds predictions back to residuals to recover the image.
    This is the decoding step in GLIC-style compression.
    """

    name = "Reconstruct"
    category = "Prediction"
    description = "Reconstruct from residuals + predictions"
    icon = "plus"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "residual",
            port_type=PortType.IMAGE,
            description="Residual image",
            required=True,
        )
        self.add_input(
            "predicted",
            port_type=PortType.IMAGE,
            description="Predicted image",
            required=True,
        )
        self.add_output(
            "reconstructed",
            port_type=PortType.IMAGE,
            description="Reconstructed image",
        )

    def process(self) -> None:
        """Reconstruct image."""
        residual: ImageBuffer = self.get_input_value("residual")
        predicted: ImageBuffer = self.get_input_value("predicted")

        if residual is None:
            raise ValueError("No residual image")
        if predicted is None:
            raise ValueError("No predicted image")

        if residual.shape != predicted.shape:
            raise ValueError("Image shapes must match")

        # Reconstruct
        reconstructed_data = residual.data + predicted.data

        # Optionally clip to valid range
        reconstructed_data = np.clip(reconstructed_data, 0.0, 1.0)

        result = ImageBuffer(
            data=reconstructed_data,
            colorspace=residual.colorspace,
            border_value=residual.border_value,
            metadata={
                **residual.metadata,
                "is_residual": False,
            },
        )

        self.set_output_value("reconstructed", result)
