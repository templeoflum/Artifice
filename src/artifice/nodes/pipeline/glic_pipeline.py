"""
GLIC Pipeline nodes.

These macro nodes combine the GLIC processing stages:
1. Color space conversion
2. Quadtree segmentation
3. Prediction
4. Residual calculation
5. Quantization

GLICEncodeNode: Full encoding pipeline (image -> quantized residuals)
GLICDecodeNode: Full decoding pipeline (quantized residuals -> image)
"""

import numpy as np

from artifice.core.data_types import ImageBuffer, SegmentList
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node
from artifice.nodes.color.conversions import convert_colorspace
from artifice.nodes.segmentation.quadtree import quadtree_segment
from artifice.nodes.prediction.predictors import (
    PredictorType,
    predict_segment,
    PREDICTOR_NAMES,
)
from artifice.nodes.quantization.quantize_node import (
    quantize_value,
    dequantize_value,
)


@register_node
class GLICEncodeNode(Node):
    """
    GLIC encoding pipeline.

    Combines all GLIC processing steps into a single node:
    1. Convert to target color space
    2. Segment image with quadtree
    3. Generate predictions
    4. Calculate residuals
    5. Quantize residuals

    This is the complete encoding side of GLIC-style compression.
    """

    name = "GLIC Encode"
    category = "Pipeline"
    description = "Complete GLIC encoding pipeline"
    icon = "package"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Input image",
            required=True,
        )
        # Outputs
        self.add_output(
            "residuals",
            port_type=PortType.IMAGE,
            description="Quantized residual image",
        )
        self.add_output(
            "segments",
            port_type=PortType.SEGMENTS,
            description="Segment list for decoding",
        )
        self.add_output(
            "predicted",
            port_type=PortType.IMAGE,
            description="Predicted image (for visualization)",
        )
        self.add_output(
            "quantized_int",
            port_type=PortType.ARRAY,
            description="Quantized integer values",
        )

    def define_parameters(self) -> None:
        """Define pipeline parameters."""
        # Color space
        colorspace_choices = [
            "RGB", "YCbCr", "YUV", "HSB", "LAB", "XYZ",
            "LUV", "HCL", "CMY", "HWB", "YDbDr", "YXY",
            "YPbPr", "OHTA", "R-GGB-G", "GREY",
        ]
        self.add_parameter(
            "colorspace",
            param_type=ParameterType.ENUM,
            default="YCbCr",
            choices=colorspace_choices,
            description="Working color space",
        )

        # Segmentation
        self.add_parameter(
            "min_segment_size",
            param_type=ParameterType.INT,
            default=4,
            min_value=2,
            max_value=64,
            description="Minimum segment size",
        )
        self.add_parameter(
            "max_segment_size",
            param_type=ParameterType.INT,
            default=64,
            min_value=4,
            max_value=512,
            description="Maximum segment size",
        )
        self.add_parameter(
            "segment_threshold",
            param_type=ParameterType.FLOAT,
            default=10.0,
            min_value=0.1,
            max_value=100.0,
            description="Segmentation variance threshold",
        )

        # Prediction
        predictor_choices = list(PREDICTOR_NAMES.values())
        self.add_parameter(
            "predictor",
            param_type=ParameterType.ENUM,
            default="PAETH",
            choices=predictor_choices,
            description="Prediction algorithm",
        )

        # Quantization
        self.add_parameter(
            "quantization_bits",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=16,
            description="Quantization bit depth",
        )

    def process(self) -> None:
        """Execute GLIC encoding pipeline."""
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        # Get parameters
        colorspace = self.get_parameter("colorspace")
        min_size = self.get_parameter("min_segment_size")
        max_size = self.get_parameter("max_segment_size")
        threshold = self.get_parameter("segment_threshold")
        predictor_name = self.get_parameter("predictor")
        bits = self.get_parameter("quantization_bits")

        # Map predictor name to type
        name_to_type = {v: k for k, v in PREDICTOR_NAMES.items()}
        predictor = name_to_type.get(predictor_name, PredictorType.PAETH)

        # Step 1: Color space conversion
        # Get source colorspace as string
        src_colorspace = buffer.colorspace
        if hasattr(src_colorspace, 'name'):
            src_colorspace = src_colorspace.name
        elif hasattr(src_colorspace, 'value'):
            src_colorspace = src_colorspace.value
        else:
            src_colorspace = str(src_colorspace)

        if src_colorspace != colorspace:
            converted_data = convert_colorspace(
                buffer.data, src_colorspace, colorspace
            )
        else:
            converted_data = buffer.data.copy()

        working_buffer = ImageBuffer(
            data=converted_data,
            colorspace=colorspace,
            border_value=buffer.border_value,
            metadata=buffer.metadata,
        )

        # Step 2: Quadtree segmentation (use first channel for segmentation)
        segments = quadtree_segment(
            working_buffer.data,
            min_size=min_size,
            max_size=max_size,
            threshold=threshold,
            per_channel=False,
        )

        # Step 3 & 4: Prediction and residual calculation
        h, w = working_buffer.height, working_buffer.width
        predicted_data = np.zeros_like(working_buffer.data)
        residual_data = np.zeros_like(working_buffer.data)

        for c in range(working_buffer.channels):
            channel_data = working_buffer.data[c]
            border = working_buffer.border_value[c] if working_buffer.border_value else 0.0

            for seg in segments:
                pred = predict_segment(channel_data, seg, predictor, border)

                # Place prediction
                x_end = min(seg.x + seg.size, w)
                y_end = min(seg.y + seg.size, h)
                pred_h = y_end - seg.y
                pred_w = x_end - seg.x

                predicted_data[c, seg.y:y_end, seg.x:x_end] = pred[:pred_h, :pred_w]

            # Calculate residual
            residual_data[c] = channel_data - predicted_data[c]

        # Step 5: Quantization
        quantized_int = quantize_value(residual_data, bits, signed=True)
        quantized_float = dequantize_value(quantized_int, bits, signed=True)

        # Create output buffers
        residuals = ImageBuffer(
            data=quantized_float,
            colorspace=colorspace,
            border_value=working_buffer.border_value,
            metadata={
                **buffer.metadata,
                "glic_encoded": True,
                "colorspace": colorspace,
                "predictor": predictor_name,
                "quantization_bits": bits,
            },
        )

        predicted = ImageBuffer(
            data=predicted_data,
            colorspace=colorspace,
            border_value=working_buffer.border_value,
            metadata=buffer.metadata,
        )

        self.set_output_value("residuals", residuals)
        self.set_output_value("segments", segments)
        self.set_output_value("predicted", predicted)
        self.set_output_value("quantized_int", quantized_int)


@register_node
class GLICDecodeNode(Node):
    """
    GLIC decoding pipeline.

    Reconstructs an image from GLIC-encoded data:
    1. Dequantize residuals
    2. Regenerate predictions
    3. Add residuals to predictions
    4. Convert back to RGB

    This is the complete decoding side of GLIC-style compression.
    """

    name = "GLIC Decode"
    category = "Pipeline"
    description = "Complete GLIC decoding pipeline"
    icon = "package"
    _abstract = False

    def define_ports(self) -> None:
        """Define ports."""
        self.add_input(
            "quantized_int",
            port_type=PortType.ARRAY,
            description="Quantized integer residuals",
            required=True,
        )
        self.add_input(
            "segments",
            port_type=PortType.SEGMENTS,
            description="Segment list",
            required=True,
        )
        self.add_input(
            "reference",
            port_type=PortType.IMAGE,
            description="Reference for metadata (optional)",
            required=False,
        )
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Reconstructed image",
        )

    def define_parameters(self) -> None:
        """Define pipeline parameters."""
        # These must match encoding parameters
        colorspace_choices = [
            "RGB", "YCbCr", "YUV", "HSB", "LAB", "XYZ",
            "LUV", "HCL", "CMY", "HWB", "YDbDr", "YXY",
            "YPbPr", "OHTA", "R-GGB-G", "GREY",
        ]
        self.add_parameter(
            "colorspace",
            param_type=ParameterType.ENUM,
            default="YCbCr",
            choices=colorspace_choices,
            description="Color space used in encoding",
        )

        predictor_choices = list(PREDICTOR_NAMES.values())
        self.add_parameter(
            "predictor",
            param_type=ParameterType.ENUM,
            default="PAETH",
            choices=predictor_choices,
            description="Predictor used in encoding",
        )

        self.add_parameter(
            "quantization_bits",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=16,
            description="Quantization bits used in encoding",
        )

        self.add_parameter(
            "output_colorspace",
            param_type=ParameterType.ENUM,
            default="RGB",
            choices=colorspace_choices,
            description="Output color space",
        )

    def process(self) -> None:
        """Execute GLIC decoding pipeline."""
        quantized_int = self.get_input_value("quantized_int")
        segments: SegmentList = self.get_input_value("segments")
        reference: ImageBuffer | None = self.get_input_value("reference")

        if quantized_int is None:
            raise ValueError("No quantized input")
        if segments is None:
            raise ValueError("No segments")

        # Get parameters
        colorspace = self.get_parameter("colorspace")
        predictor_name = self.get_parameter("predictor")
        bits = self.get_parameter("quantization_bits")
        output_colorspace = self.get_parameter("output_colorspace")

        # Map predictor name to type
        name_to_type = {v: k for k, v in PREDICTOR_NAMES.items()}
        predictor = name_to_type.get(predictor_name, PredictorType.PAETH)

        # Step 1: Dequantize residuals
        residuals = dequantize_value(quantized_int, bits, signed=True)

        # Get dimensions
        if len(residuals.shape) == 3:
            channels, h, w = residuals.shape
        else:
            h, w = residuals.shape
            channels = 1
            residuals = residuals[np.newaxis, :, :]

        # Step 2 & 3: Regenerate predictions and reconstruct
        # This is tricky because we need the reconstructed values to make predictions
        # GLIC decodes segment by segment in order
        reconstructed = np.zeros_like(residuals)

        # Get border value
        if reference is not None and reference.border_value:
            border_values = reference.border_value
        else:
            border_values = tuple([0.0] * channels)

        for c in range(channels):
            border = border_values[c] if c < len(border_values) else 0.0

            for seg in segments:
                # Generate prediction from already-reconstructed data
                pred = predict_segment(reconstructed[c], seg, predictor, border)

                # Get residual for this segment
                x_end = min(seg.x + seg.size, w)
                y_end = min(seg.y + seg.size, h)
                pred_h = y_end - seg.y
                pred_w = x_end - seg.x

                seg_residual = residuals[c, seg.y:y_end, seg.x:x_end]

                # Reconstruct: prediction + residual
                reconstructed[c, seg.y:y_end, seg.x:x_end] = (
                    pred[:pred_h, :pred_w] + seg_residual
                )

        # Clip to valid range
        reconstructed = np.clip(reconstructed, 0.0, 1.0)

        # Step 4: Convert to output color space
        if colorspace != output_colorspace:
            output_data = convert_colorspace(
                reconstructed, colorspace, output_colorspace
            )
        else:
            output_data = reconstructed

        output_data = np.clip(output_data, 0.0, 1.0)

        # Create output buffer
        result = ImageBuffer(
            data=output_data,
            colorspace=output_colorspace,
            border_value=border_values if reference else None,
            metadata={
                "glic_decoded": True,
            },
        )

        self.set_output_value("image", result)
