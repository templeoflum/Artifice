"""GPU-accelerated color processing nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from artifice.core.gpu_node import GPUNode, ShaderUniform
from artifice.core.node import ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node

if TYPE_CHECKING:
    pass


@register_node
class ColorSpaceGPUNode(GPUNode):
    """GPU-accelerated color space conversion.

    Converts between all 16 GLIC color spaces:

    Perceptual (good for gradients):
    - LAB: Perceptually uniform, excellent for color manipulation
    - LUV: Similar to LAB, better for additive color
    - HCL: Cylindrical LAB, intuitive hue control

    Video/Compression:
    - YCbCr: JPEG/MPEG standard, separates luma from chroma
    - YUV: Analog video standard
    - YPbPr: Component video
    - YDbDr: SECAM video

    Artist-Friendly:
    - HSV/HSB: Intuitive hue/saturation/brightness
    - HSL: Similar but different lightness model
    - HWB: Hue/whiteness/blackness (CSS standard)

    Scientific:
    - XYZ: CIE 1931, device-independent
    - YXY: Chromaticity diagram coordinates

    Special:
    - CMY: Subtractive color (print)
    - OHTA: Optimal color features
    - GREY: Grayscale (luma only)

    Glitch effects work best by processing in a luma-chroma space
    (YCbCr, LAB) then corrupting the chroma channels while
    preserving luma for structure.
    """

    name = "Color Space (GPU)"
    category = "Color"
    description = "Convert between all 16 GLIC color spaces (GPU accelerated)"
    shader_file = "color/colorspace.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("from_space", "from_space", "int", 0),
        ShaderUniform("to_space", "to_space", "int", 1),
    ]

    # All 16 GLIC color spaces - order must match shader defines
    SPACES = [
        "RGB",      # 0: Identity
        "HSV",      # 1: Hue-Saturation-Value
        "HSL",      # 2: Hue-Saturation-Lightness
        "YCbCr",    # 3: JPEG/MPEG luma-chroma
        "YUV",      # 4: Analog video
        "LAB",      # 5: CIE L*a*b* (perceptual)
        "XYZ",      # 6: CIE 1931
        "LUV",      # 7: CIE L*u*v*
        "HCL",      # 8: Hue-Chroma-Luma
        "CMY",      # 9: Cyan-Magenta-Yellow
        "HWB",      # 10: Hue-Whiteness-Blackness
        "YPbPr",    # 11: Component video
        "YDbDr",    # 12: SECAM video
        "OHTA",     # 13: Optimal features
        "YXY",      # 14: CIE chromaticity
        "GREY",     # 15: Grayscale
    ]

    # Map from string name to shader ID
    SPACE_MAP = {name: idx for idx, name in enumerate(SPACES)}

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "from_space",
            param_type=ParameterType.ENUM,
            default="RGB",
            choices=self.SPACES,
            description="Source color space",
        )
        self.add_parameter(
            "to_space",
            param_type=ParameterType.ENUM,
            default="YCbCr",  # Default to YCbCr - best for glitch effects
            choices=self.SPACES,
            description="Target color space",
        )

    def _upload_uniforms(self) -> None:
        """Upload uniforms with enum conversion."""
        from_str = self.get_parameter("from_space")
        to_str = self.get_parameter("to_space")

        from_int = self.SPACE_MAP.get(from_str, 0)
        to_int = self.SPACE_MAP.get(to_str, 0)

        if "from_space" in self._compiled_shader:
            self._compiled_shader["from_space"].value = from_int
        if "to_space" in self._compiled_shader:
            self._compiled_shader["to_space"].value = to_int


@register_node
class BlendGPUNode(GPUNode):
    """GPU-accelerated image blending."""

    name = "Blend (GPU)"
    category = "Color"
    description = "Blend two images together (GPU accelerated)"
    shader_file = "color/blend.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("blend_mode", "blend_mode", "int", 0),
        ShaderUniform("mix_factor", "mix_factor", "float", 0.5),
    ]

    BLEND_MODES = ["Mix", "Add", "Multiply", "Screen", "Overlay", "Difference"]

    def define_ports(self) -> None:
        self.add_input("A", PortType.IMAGE, "First image")
        self.add_input("B", PortType.IMAGE, "Second image")
        self.add_output("image", PortType.IMAGE, "Blended image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "blend_mode",
            param_type=ParameterType.ENUM,
            default="Mix",
            choices=self.BLEND_MODES,
            description="Blend mode",
        )
        self.add_parameter(
            "mix_factor",
            param_type=ParameterType.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Mix factor (0=A, 1=B)",
        )

    def _upload_uniforms(self) -> None:
        mode_str = self.get_parameter("blend_mode")
        mode_int = self.BLEND_MODES.index(mode_str) if mode_str in self.BLEND_MODES else 0
        if "blend_mode" in self._compiled_shader:
            self._compiled_shader["blend_mode"].value = mode_int
        if "mix_factor" in self._compiled_shader:
            self._compiled_shader["mix_factor"].value = self.get_parameter("mix_factor")


@register_node
class InvertGPUNode(GPUNode):
    """GPU-accelerated color inversion."""

    name = "Invert (GPU)"
    category = "Color"
    description = "Invert image colors (GPU accelerated)"
    shader_file = "color/invert.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("invert_r", "invert_r", "bool", True),
        ShaderUniform("invert_g", "invert_g", "bool", True),
        ShaderUniform("invert_b", "invert_b", "bool", True),
        ShaderUniform("invert_a", "invert_a", "bool", False),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Inverted image")

    def define_parameters(self) -> None:
        self.add_parameter("invert_r", param_type=ParameterType.BOOL, default=True, description="Invert red")
        self.add_parameter("invert_g", param_type=ParameterType.BOOL, default=True, description="Invert green")
        self.add_parameter("invert_b", param_type=ParameterType.BOOL, default=True, description="Invert blue")
        self.add_parameter("invert_a", param_type=ParameterType.BOOL, default=False, description="Invert alpha")


@register_node
class BrightnessContrastGPUNode(GPUNode):
    """GPU-accelerated brightness/contrast adjustment."""

    name = "Brightness/Contrast (GPU)"
    category = "Color"
    description = "Adjust brightness and contrast (GPU accelerated)"
    shader_file = "color/brightness_contrast.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("brightness", "brightness", "float", 0.0),
        ShaderUniform("contrast", "contrast", "float", 1.0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Adjusted image")

    def define_parameters(self) -> None:
        self.add_parameter("brightness", param_type=ParameterType.FLOAT, default=0.0,
                          min_value=-1.0, max_value=1.0, step=0.01, description="Brightness adjustment")
        self.add_parameter("contrast", param_type=ParameterType.FLOAT, default=1.0,
                          min_value=0.0, max_value=3.0, step=0.01, description="Contrast multiplier")


@register_node
class ThresholdGPUNode(GPUNode):
    """GPU-accelerated threshold."""

    name = "Threshold (GPU)"
    category = "Color"
    description = "Convert to binary or threshold levels (GPU accelerated)"
    shader_file = "color/threshold.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("threshold", "threshold", "float", 0.5),
        ShaderUniform("smoothness", "smoothness", "float", 0.0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Thresholded image")

    def define_parameters(self) -> None:
        self.add_parameter("threshold", param_type=ParameterType.FLOAT, default=0.5,
                          min_value=0.0, max_value=1.0, step=0.01, description="Threshold level")
        self.add_parameter("smoothness", param_type=ParameterType.FLOAT, default=0.0,
                          min_value=0.0, max_value=0.5, step=0.01, description="Edge smoothness")


@register_node
class PosterizeGPUNode(GPUNode):
    """GPU-accelerated posterization."""

    name = "Posterize (GPU)"
    category = "Color"
    description = "Reduce color levels (GPU accelerated)"
    shader_file = "color/posterize.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("levels", "levels", "int", 4),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Posterized image")

    def define_parameters(self) -> None:
        self.add_parameter("levels", param_type=ParameterType.INT, default=4,
                          min_value=2, max_value=256, description="Number of color levels")


@register_node
class ChannelSplitGPUNode(GPUNode):
    """GPU-accelerated channel split."""

    name = "Channel Split (GPU)"
    category = "Color"
    description = "Split image into individual channels (GPU accelerated)"
    shader_file = "color/channel_split.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("channel", "channel", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Channel output")

    def define_parameters(self) -> None:
        self.add_parameter("channel", param_type=ParameterType.ENUM, default="Red",
                          choices=["Red", "Green", "Blue", "Alpha"], description="Channel to extract")

    def _upload_uniforms(self) -> None:
        channel_map = {"Red": 0, "Green": 1, "Blue": 2, "Alpha": 3}
        channel = self.get_parameter("channel")
        if "channel" in self._compiled_shader:
            self._compiled_shader["channel"].value = channel_map.get(channel, 0)


@register_node
class ChannelMergeGPUNode(GPUNode):
    """GPU-accelerated channel merge."""

    name = "Channel Merge (GPU)"
    category = "Color"
    description = "Merge channels into single image (GPU accelerated)"
    shader_file = "color/channel_merge.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = []

    def define_ports(self) -> None:
        self.add_input("R", PortType.IMAGE, "Red channel")
        self.add_input("G", PortType.IMAGE, "Green channel")
        self.add_input("B", PortType.IMAGE, "Blue channel")
        self.add_output("image", PortType.IMAGE, "Merged image")

    def define_parameters(self) -> None:
        pass


@register_node
class ChannelSwapGPUNode(GPUNode):
    """GPU-accelerated channel swap."""

    name = "Channel Swap (GPU)"
    category = "Color"
    description = "Swap/remap color channels (GPU accelerated)"
    shader_file = "color/channel_swap.glsl"
    _abstract = False

    CHANNELS = ["Red", "Green", "Blue", "Alpha", "Zero", "One"]

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("r_source", "r_source", "int", 0),
        ShaderUniform("g_source", "g_source", "int", 1),
        ShaderUniform("b_source", "b_source", "int", 2),
        ShaderUniform("a_source", "a_source", "int", 3),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Channel-swapped image")

    def define_parameters(self) -> None:
        self.add_parameter("r_source", param_type=ParameterType.ENUM, default="Red",
                          choices=self.CHANNELS, description="Red channel source")
        self.add_parameter("g_source", param_type=ParameterType.ENUM, default="Green",
                          choices=self.CHANNELS, description="Green channel source")
        self.add_parameter("b_source", param_type=ParameterType.ENUM, default="Blue",
                          choices=self.CHANNELS, description="Blue channel source")
        self.add_parameter("a_source", param_type=ParameterType.ENUM, default="Alpha",
                          choices=self.CHANNELS, description="Alpha channel source")

    def _upload_uniforms(self) -> None:
        for param in ["r_source", "g_source", "b_source", "a_source"]:
            val = self.get_parameter(param)
            idx = self.CHANNELS.index(val) if val in self.CHANNELS else 0
            if param in self._compiled_shader:
                self._compiled_shader[param].value = idx
