"""GPU-accelerated transform nodes."""

from __future__ import annotations

from typing import ClassVar

from artifice.core.gpu_node import GPUNode, ShaderUniform
from artifice.core.node import ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class PixelSortGPUNode(GPUNode):
    """GPU-accelerated pixel sorting."""

    name = "Pixel Sort (GPU)"
    category = "Transform"
    description = "Sort pixels within rows or columns (GPU accelerated)"
    shader_file = "transform/pixelsort.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("threshold_mode", "threshold_mode", "int", 0),
        ShaderUniform("threshold_low", "threshold_low", "float", 0.25),
        ShaderUniform("threshold_high", "threshold_high", "float", 0.8),
        ShaderUniform("sort_by", "sort_by", "int", 0),
        ShaderUniform("direction", "direction", "int", 0),
        ShaderUniform("reverse_sort", "reverse_sort", "int", 0),
        ShaderUniform("seed", "seed", "int", 0),
    ]

    THRESHOLD_MODES = ["Brightness", "Random", "None"]
    SORT_BY = ["Brightness", "Hue", "Saturation", "Red", "Green", "Blue"]
    DIRECTIONS = ["Horizontal", "Vertical"]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Sorted image")

    def define_parameters(self) -> None:
        self.add_parameter("threshold_mode", param_type=ParameterType.ENUM, default="Brightness",
                          choices=self.THRESHOLD_MODES, description="Threshold mode")
        self.add_parameter("threshold_low", param_type=ParameterType.FLOAT, default=0.25,
                          min_value=0.0, max_value=1.0, step=0.01, description="Lower threshold")
        self.add_parameter("threshold_high", param_type=ParameterType.FLOAT, default=0.8,
                          min_value=0.0, max_value=1.0, step=0.01, description="Upper threshold")
        self.add_parameter("sort_by", param_type=ParameterType.ENUM, default="Brightness",
                          choices=self.SORT_BY, description="Sort criterion")
        self.add_parameter("direction", param_type=ParameterType.ENUM, default="Horizontal",
                          choices=self.DIRECTIONS, description="Sort direction")
        self.add_parameter("reverse_sort", param_type=ParameterType.ENUM, default="Ascending",
                          choices=["Ascending", "Descending"], description="Sort order")
        self.add_parameter("seed", param_type=ParameterType.INT, default=0,
                          min_value=0, max_value=999999, description="Random seed")

    def _upload_uniforms(self) -> None:
        for name, choices in [("threshold_mode", self.THRESHOLD_MODES),
                               ("sort_by", self.SORT_BY),
                               ("direction", self.DIRECTIONS)]:
            val = self.get_parameter(name)
            if name in self._compiled_shader:
                self._compiled_shader[name].value = choices.index(val) if val in choices else 0
        if "threshold_low" in self._compiled_shader:
            self._compiled_shader["threshold_low"].value = self.get_parameter("threshold_low")
        if "threshold_high" in self._compiled_shader:
            self._compiled_shader["threshold_high"].value = self.get_parameter("threshold_high")
        if "reverse_sort" in self._compiled_shader:
            self._compiled_shader["reverse_sort"].value = 1 if self.get_parameter("reverse_sort") == "Descending" else 0
        if "seed" in self._compiled_shader:
            self._compiled_shader["seed"].value = self.get_parameter("seed")


@register_node
class MirrorGPUNode(GPUNode):
    """GPU-accelerated image mirroring."""

    name = "Mirror (GPU)"
    category = "Transform"
    description = "Mirror/flip image (GPU accelerated)"
    shader_file = "transform/mirror.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("axis", "axis", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Mirrored image")

    def define_parameters(self) -> None:
        self.add_parameter("axis", param_type=ParameterType.ENUM, default="Horizontal",
                          choices=["Horizontal", "Vertical", "Both"], description="Mirror axis")

    def _upload_uniforms(self) -> None:
        axis_map = {"Horizontal": 0, "Vertical": 1, "Both": 2}
        if "axis" in self._compiled_shader:
            self._compiled_shader["axis"].value = axis_map.get(self.get_parameter("axis"), 0)


@register_node
class RotateGPUNode(GPUNode):
    """GPU-accelerated image rotation."""

    name = "Rotate (GPU)"
    category = "Transform"
    description = "Rotate image (GPU accelerated)"
    shader_file = "transform/rotate.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("angle", "angle", "float", 0.0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Rotated image")

    def define_parameters(self) -> None:
        self.add_parameter("angle", param_type=ParameterType.FLOAT, default=0.0,
                          min_value=0.0, max_value=360.0, step=1.0, description="Rotation angle (degrees)")


@register_node
class BlurGPUNode(GPUNode):
    """GPU-accelerated blur."""

    name = "Blur (GPU)"
    category = "Transform"
    description = "Apply blur effect (GPU accelerated)"
    shader_file = "transform/blur.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("radius", "radius", "float", 2.0),
        ShaderUniform("blur_type", "blur_type", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Blurred image")

    def define_parameters(self) -> None:
        self.add_parameter("radius", param_type=ParameterType.FLOAT, default=2.0,
                          min_value=0.0, max_value=50.0, step=0.5, description="Blur radius")
        self.add_parameter("blur_type", param_type=ParameterType.ENUM, default="Gaussian",
                          choices=["Gaussian", "Box"], description="Blur type")

    def _upload_uniforms(self) -> None:
        if "radius" in self._compiled_shader:
            self._compiled_shader["radius"].value = self.get_parameter("radius")
        if "blur_type" in self._compiled_shader:
            self._compiled_shader["blur_type"].value = 0 if self.get_parameter("blur_type") == "Gaussian" else 1


@register_node
class SharpenGPUNode(GPUNode):
    """GPU-accelerated sharpening."""

    name = "Sharpen (GPU)"
    category = "Transform"
    description = "Apply sharpening effect (GPU accelerated)"
    shader_file = "transform/sharpen.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("amount", "amount", "float", 1.0),
        ShaderUniform("radius", "radius", "float", 1.0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Sharpened image")

    def define_parameters(self) -> None:
        self.add_parameter("amount", param_type=ParameterType.FLOAT, default=1.0,
                          min_value=0.0, max_value=5.0, step=0.1, description="Sharpening strength")
        self.add_parameter("radius", param_type=ParameterType.FLOAT, default=1.0,
                          min_value=0.5, max_value=10.0, step=0.5, description="Effect radius")


@register_node
class EdgeDetectGPUNode(GPUNode):
    """GPU-accelerated edge detection."""

    name = "Edge Detect (GPU)"
    category = "Transform"
    description = "Detect edges (GPU accelerated)"
    shader_file = "transform/edge_detect.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("method", "method", "int", 0),
        ShaderUniform("threshold", "threshold", "float", 0.1),
    ]

    METHODS = ["Sobel", "Prewitt", "Laplacian"]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Edge image")

    def define_parameters(self) -> None:
        self.add_parameter("method", param_type=ParameterType.ENUM, default="Sobel",
                          choices=self.METHODS, description="Edge detection method")
        self.add_parameter("threshold", param_type=ParameterType.FLOAT, default=0.1,
                          min_value=0.0, max_value=1.0, step=0.01, description="Edge threshold")

    def _upload_uniforms(self) -> None:
        if "method" in self._compiled_shader:
            method = self.get_parameter("method")
            self._compiled_shader["method"].value = self.METHODS.index(method) if method in self.METHODS else 0
        if "threshold" in self._compiled_shader:
            self._compiled_shader["threshold"].value = self.get_parameter("threshold")


@register_node
class DCTGPUNode(GPUNode):
    """GPU-accelerated Discrete Cosine Transform."""

    name = "DCT (GPU)"
    category = "Transform"
    description = "Discrete Cosine Transform (GPU accelerated)"
    shader_file = "transform/dct.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("block_size", "block_size", "int", 8),
        ShaderUniform("quality", "quality", "float", 1.0),
        ShaderUniform("inverse", "inverse", "bool", False),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Transformed image")

    def define_parameters(self) -> None:
        self.add_parameter("block_size", param_type=ParameterType.ENUM, default="8",
                          choices=["4", "8", "16", "32"], description="DCT block size")
        self.add_parameter("quality", param_type=ParameterType.FLOAT, default=1.0,
                          min_value=0.0, max_value=2.0, step=0.01, description="Quality factor")
        self.add_parameter("inverse", param_type=ParameterType.BOOL, default=False,
                          description="Apply inverse DCT")

    def _upload_uniforms(self) -> None:
        if "block_size" in self._compiled_shader:
            self._compiled_shader["block_size"].value = int(self.get_parameter("block_size"))
        if "quality" in self._compiled_shader:
            self._compiled_shader["quality"].value = self.get_parameter("quality")
        if "inverse" in self._compiled_shader:
            self._compiled_shader["inverse"].value = 1 if self.get_parameter("inverse") else 0


@register_node
class FFTGPUNode(GPUNode):
    """GPU-accelerated Fast Fourier Transform."""

    name = "FFT (GPU)"
    category = "Transform"
    description = "Fast Fourier Transform (GPU accelerated)"
    shader_file = "transform/fft.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("inverse", "inverse", "bool", False),
        ShaderUniform("shift", "shift", "bool", True),
        ShaderUniform("log_scale", "log_scale", "bool", True),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Transformed image")

    def define_parameters(self) -> None:
        self.add_parameter("inverse", param_type=ParameterType.BOOL, default=False,
                          description="Apply inverse FFT")
        self.add_parameter("shift", param_type=ParameterType.BOOL, default=True,
                          description="Center zero frequency")
        self.add_parameter("log_scale", param_type=ParameterType.BOOL, default=True,
                          description="Logarithmic magnitude display")


@register_node
class WaveletGPUNode(GPUNode):
    """GPU-accelerated wavelet transform."""

    name = "Wavelet (GPU)"
    category = "Transform"
    description = "Wavelet transform decomposition (GPU accelerated)"
    shader_file = "transform/wavelet.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("wavelet_type", "wavelet_type", "int", 0),
        ShaderUniform("levels", "levels", "int", 3),
        ShaderUniform("inverse", "inverse", "bool", False),
        ShaderUniform("threshold", "threshold", "float", 0.0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Transformed image")

    def define_parameters(self) -> None:
        self.add_parameter("wavelet_type", param_type=ParameterType.ENUM, default="Haar",
                          choices=["Haar", "Daubechies"], description="Wavelet type")
        self.add_parameter("levels", param_type=ParameterType.INT, default=3,
                          min_value=1, max_value=6, description="Decomposition levels")
        self.add_parameter("inverse", param_type=ParameterType.BOOL, default=False,
                          description="Apply inverse transform")
        self.add_parameter("threshold", param_type=ParameterType.FLOAT, default=0.0,
                          min_value=0.0, max_value=1.0, step=0.01, description="Coefficient threshold")

    def _upload_uniforms(self) -> None:
        if "wavelet_type" in self._compiled_shader:
            self._compiled_shader["wavelet_type"].value = 0 if self.get_parameter("wavelet_type") == "Haar" else 1
        if "levels" in self._compiled_shader:
            self._compiled_shader["levels"].value = self.get_parameter("levels")
        if "inverse" in self._compiled_shader:
            self._compiled_shader["inverse"].value = 1 if self.get_parameter("inverse") else 0
        if "threshold" in self._compiled_shader:
            self._compiled_shader["threshold"].value = self.get_parameter("threshold")
