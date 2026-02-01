"""GPU-accelerated GLIC codec nodes."""

from __future__ import annotations

from typing import ClassVar

from artifice.core.gpu_node import GPUNode, ShaderUniform
from artifice.core.node import ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class GLICPredictGPUNode(GPUNode):
    """GPU-accelerated GLIC prediction.

    Implements all 14 base GLIC predictors plus special selection modes:
    - 0-13: Specific predictors (None, Corner, H, V, DC, etc.)
    - 14: SAD (Best) - automatically selects best predictor per block
    - 15: BSAD (Worst) - automatically selects WORST predictor per block (GLITCH ART!)
    - 16: Random - random predictor per block

    BSAD is the key to GLIC-style glitch effects - it deliberately picks
    the predictor that produces the MAXIMUM error, which when quantized
    creates dramatic color shifts and artifacts.
    """

    name = "GLIC Predict (GPU)"
    category = "GLIC"
    description = "Generate predictions using GLIC predictors (GPU accelerated)"
    shader_file = "glic/predict.glsl"
    _abstract = False

    PREDICTORS = [
        "None",          # 0
        "Corner",        # 1
        "Horizontal",    # 2
        "Vertical",      # 3
        "DC Mean",       # 4
        "DC Median",     # 5
        "Median",        # 6
        "Average",       # 7
        "TrueMotion",    # 8
        "Paeth",         # 9
        "Linear Diag",   # 10
        "H/V Position",  # 11
        "JPEG-LS",       # 12
        "Difference",    # 13
        "SAD (Best)",    # 14 - auto-select best
        "BSAD (Worst)",  # 15 - auto-select WORST (glitch!)
        "Random",        # 16 - random per block
    ]

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("block_size", "block_size", "int", 16),
        ShaderUniform("predictor_mode", "predictor_mode", "int", 9),
        ShaderUniform("border_value", "border_value", "float", 0.5),
        ShaderUniform("seed", "seed", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Prediction image")

    def define_parameters(self) -> None:
        self.add_parameter("block_size", param_type=ParameterType.ENUM, default="16",
                          choices=["4", "8", "16", "32", "64"], description="Prediction block size")
        self.add_parameter("predictor_mode", param_type=ParameterType.ENUM, default="Paeth",
                          choices=self.PREDICTORS, description="Predictor algorithm")
        self.add_parameter("border_value", param_type=ParameterType.FLOAT, default=0.5,
                          min_value=0.0, max_value=1.0, step=0.01, description="Border pixel value")
        self.add_parameter("seed", param_type=ParameterType.INT, default=0,
                          min_value=0, max_value=999999, description="Random seed (for Random mode)")

    def _upload_uniforms(self) -> None:
        if "block_size" in self._compiled_shader:
            self._compiled_shader["block_size"].value = int(self.get_parameter("block_size"))
        if "predictor_mode" in self._compiled_shader:
            mode = self.get_parameter("predictor_mode")
            self._compiled_shader["predictor_mode"].value = self.PREDICTORS.index(mode) if mode in self.PREDICTORS else 9
        if "border_value" in self._compiled_shader:
            self._compiled_shader["border_value"].value = self.get_parameter("border_value")
        if "seed" in self._compiled_shader:
            self._compiled_shader["seed"].value = self.get_parameter("seed")


@register_node
class GLICResidualGPUNode(GPUNode):
    """GPU-accelerated GLIC residual calculation.

    Calculates the difference between the original image and the prediction.
    Different methods produce different glitch characteristics:

    - Subtract: Simple difference (can go negative)
    - Clamp: Clamped to [0,1] range
    - Wrap: Wrapped around (modulo)
    - CLAMP_MOD256: GLIC-style - creates distinctive color shifts
    """

    name = "GLIC Residual (GPU)"
    category = "GLIC"
    description = "Calculate residuals between image and prediction (GPU accelerated)"
    shader_file = "glic/residual.glsl"
    _abstract = False

    METHODS = ["Subtract", "Clamp", "Wrap", "CLAMP_MOD256"]

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("method", "method", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Original image")
        self.add_input("prediction", PortType.IMAGE, "Prediction image")
        self.add_output("image", PortType.IMAGE, "Residual image")

    def define_parameters(self) -> None:
        self.add_parameter("method", param_type=ParameterType.ENUM, default="Subtract",
                          choices=self.METHODS, description="Residual calculation method")

    def _upload_uniforms(self) -> None:
        if "method" in self._compiled_shader:
            method = self.get_parameter("method")
            self._compiled_shader["method"].value = self.METHODS.index(method) if method in self.METHODS else 0


@register_node
class GLICReconstructGPUNode(GPUNode):
    """GPU-accelerated GLIC reconstruction.

    Reconstructs the image from prediction and residuals.
    The method should match the residual calculation method used.
    """

    name = "GLIC Reconstruct (GPU)"
    category = "GLIC"
    description = "Reconstruct image from prediction and residuals (GPU accelerated)"
    shader_file = "glic/reconstruct.glsl"
    _abstract = False

    METHODS = ["Add", "Clamp", "Wrap", "CLAMP_MOD256"]

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("method", "method", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("prediction", PortType.IMAGE, "Prediction image")
        self.add_input("residual", PortType.IMAGE, "Residual image")
        self.add_output("image", PortType.IMAGE, "Reconstructed image")

    def define_parameters(self) -> None:
        self.add_parameter("method", param_type=ParameterType.ENUM, default="Add",
                          choices=self.METHODS, description="Reconstruction method")

    def _upload_uniforms(self) -> None:
        if "method" in self._compiled_shader:
            method = self.get_parameter("method")
            self._compiled_shader["method"].value = self.METHODS.index(method) if method in self.METHODS else 0
