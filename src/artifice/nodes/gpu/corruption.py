"""GPU-accelerated corruption nodes."""

from __future__ import annotations

from typing import ClassVar

from artifice.core.gpu_node import GPUNode, ShaderUniform
from artifice.core.node import ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class BitFlipGPUNode(GPUNode):
    """GPU-accelerated bit flip corruption."""

    name = "Bit Flip (GPU)"
    category = "Corruption"
    description = "Randomly flip bits in image data (GPU accelerated)"
    shader_file = "corruption/bitflip.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("probability", "probability", "float", 0.01),
        ShaderUniform("seed", "seed", "int", 0),
        ShaderUniform("bits_per_channel", "bits", "int", 8),
        ShaderUniform("affect_alpha", "affect_alpha", "bool", False),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "probability",
            param_type=ParameterType.FLOAT,
            default=0.01,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            description="Probability of flipping each bit",
        )
        self.add_parameter(
            "seed",
            param_type=ParameterType.INT,
            default=0,
            min_value=0,
            max_value=999999,
            description="Random seed for reproducibility",
        )
        self.add_parameter(
            "bits",
            param_type=ParameterType.INT,
            default=8,
            min_value=1,
            max_value=8,
            description="Number of bits per channel to consider",
        )
        self.add_parameter(
            "affect_alpha",
            param_type=ParameterType.BOOL,
            default=False,
            description="Whether to affect alpha channel",
        )


@register_node
class BitShiftGPUNode(GPUNode):
    """GPU-accelerated bit shift corruption."""

    name = "Bit Shift (GPU)"
    category = "Corruption"
    description = "Shift bits in image data (GPU accelerated)"
    shader_file = "corruption/bitshift.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("shift_amount", "shift", "int", 1),
        ShaderUniform("wrap", "wrap", "bool", True),
        ShaderUniform("affect_alpha", "affect_alpha", "bool", False),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "shift",
            param_type=ParameterType.INT,
            default=1,
            min_value=-7,
            max_value=7,
            description="Bit shift amount (negative = right shift)",
        )
        self.add_parameter(
            "wrap",
            param_type=ParameterType.BOOL,
            default=True,
            description="Wrap bits around (rotate) instead of shifting in zeros",
        )
        self.add_parameter(
            "affect_alpha",
            param_type=ParameterType.BOOL,
            default=False,
            description="Whether to affect alpha channel",
        )


@register_node
class XORNoiseGPUNode(GPUNode):
    """GPU-accelerated XOR noise corruption."""

    name = "XOR Noise (GPU)"
    category = "Corruption"
    description = "Apply XOR noise to image data (GPU accelerated)"
    shader_file = "corruption/xor_noise.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("noise_seed", "seed", "int", 0),
        ShaderUniform("intensity", "intensity", "float", 0.5),
        ShaderUniform("affect_alpha", "affect_alpha", "bool", False),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "seed",
            param_type=ParameterType.INT,
            default=0,
            min_value=0,
            max_value=999999,
            description="Random seed for noise generation",
        )
        self.add_parameter(
            "intensity",
            param_type=ParameterType.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Noise intensity (0-1)",
        )
        self.add_parameter(
            "affect_alpha",
            param_type=ParameterType.BOOL,
            default=False,
            description="Whether to affect alpha channel",
        )


@register_node
class DataRepeatGPUNode(GPUNode):
    """GPU-accelerated data repeat."""

    name = "Data Repeat (GPU)"
    category = "Corruption"
    description = "Repeat rows or columns of data (GPU accelerated)"
    shader_file = "corruption/data_repeat.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("repeat_size", "repeat_size", "int", 8),
        ShaderUniform("direction", "direction", "int", 0),
        ShaderUniform("offset", "offset", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter("repeat_size", param_type=ParameterType.INT, default=8,
                          min_value=1, max_value=128, description="Size of repeated segment")
        self.add_parameter("direction", param_type=ParameterType.ENUM, default="Horizontal",
                          choices=["Horizontal", "Vertical"], description="Repeat direction")
        self.add_parameter("offset", param_type=ParameterType.INT, default=0,
                          min_value=0, max_value=256, description="Starting offset")

    def _upload_uniforms(self) -> None:
        if "repeat_size" in self._compiled_shader:
            self._compiled_shader["repeat_size"].value = self.get_parameter("repeat_size")
        if "direction" in self._compiled_shader:
            self._compiled_shader["direction"].value = 0 if self.get_parameter("direction") == "Horizontal" else 1
        if "offset" in self._compiled_shader:
            self._compiled_shader["offset"].value = self.get_parameter("offset")


@register_node
class DataDropGPUNode(GPUNode):
    """GPU-accelerated data drop."""

    name = "Data Drop (GPU)"
    category = "Corruption"
    description = "Skip/drop rows or columns of data (GPU accelerated)"
    shader_file = "corruption/data_drop.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("drop_size", "drop_size", "int", 4),
        ShaderUniform("keep_size", "keep_size", "int", 12),
        ShaderUniform("direction", "direction", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter("drop_size", param_type=ParameterType.INT, default=4,
                          min_value=1, max_value=64, description="Size of dropped segment")
        self.add_parameter("keep_size", param_type=ParameterType.INT, default=12,
                          min_value=1, max_value=64, description="Size of kept segment")
        self.add_parameter("direction", param_type=ParameterType.ENUM, default="Horizontal",
                          choices=["Horizontal", "Vertical"], description="Drop direction")

    def _upload_uniforms(self) -> None:
        if "drop_size" in self._compiled_shader:
            self._compiled_shader["drop_size"].value = self.get_parameter("drop_size")
        if "keep_size" in self._compiled_shader:
            self._compiled_shader["keep_size"].value = self.get_parameter("keep_size")
        if "direction" in self._compiled_shader:
            self._compiled_shader["direction"].value = 0 if self.get_parameter("direction") == "Horizontal" else 1


@register_node
class DataScrambleGPUNode(GPUNode):
    """GPU-accelerated data scramble."""

    name = "Data Scramble (GPU)"
    category = "Corruption"
    description = "Shuffle data segments (GPU accelerated)"
    shader_file = "corruption/data_scramble.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("block_size", "block_size", "int", 16),
        ShaderUniform("direction", "direction", "int", 0),
        ShaderUniform("seed", "seed", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter("block_size", param_type=ParameterType.INT, default=16,
                          min_value=4, max_value=128, description="Scramble block size")
        self.add_parameter("direction", param_type=ParameterType.ENUM, default="Horizontal",
                          choices=["Horizontal", "Vertical", "Both"], description="Scramble direction")
        self.add_parameter("seed", param_type=ParameterType.INT, default=0,
                          min_value=0, max_value=999999, description="Random seed")

    def _upload_uniforms(self) -> None:
        if "block_size" in self._compiled_shader:
            self._compiled_shader["block_size"].value = self.get_parameter("block_size")
        if "direction" in self._compiled_shader:
            dir_map = {"Horizontal": 0, "Vertical": 1, "Both": 2}
            self._compiled_shader["direction"].value = dir_map.get(self.get_parameter("direction"), 0)
        if "seed" in self._compiled_shader:
            self._compiled_shader["seed"].value = self.get_parameter("seed")


@register_node
class DataWeaveGPUNode(GPUNode):
    """GPU-accelerated data weave."""

    name = "Data Weave (GPU)"
    category = "Corruption"
    description = "Interleave two images (GPU accelerated)"
    shader_file = "corruption/data_weave.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("weave_size", "weave_size", "int", 4),
        ShaderUniform("direction", "direction", "int", 0),
        ShaderUniform("blend_width", "blend_width", "float", 0.0),
        ShaderUniform("mix_amount", "mix_amount", "float", 0.5),
        ShaderUniform("offset", "offset", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "First image")
        self.add_input("image_b", PortType.IMAGE, "Second image")
        self.add_output("image", PortType.IMAGE, "Woven image")

    def define_parameters(self) -> None:
        self.add_parameter("weave_size", param_type=ParameterType.INT, default=4,
                          min_value=1, max_value=128, description="Width of weave stripes")
        self.add_parameter("direction", param_type=ParameterType.ENUM, default="Horizontal",
                          choices=["Horizontal", "Vertical", "Checker"], description="Weave pattern")
        self.add_parameter("blend_width", param_type=ParameterType.FLOAT, default=0.0,
                          min_value=0.0, max_value=1.0, step=0.01, description="Edge blend smoothness")
        self.add_parameter("mix_amount", param_type=ParameterType.FLOAT, default=0.5,
                          min_value=0.0, max_value=1.0, step=0.01, description="Balance between images")
        self.add_parameter("offset", param_type=ParameterType.INT, default=0,
                          min_value=0, max_value=256, description="Pattern offset")

    def _upload_uniforms(self) -> None:
        if "weave_size" in self._compiled_shader:
            self._compiled_shader["weave_size"].value = self.get_parameter("weave_size")
        if "direction" in self._compiled_shader:
            dir_map = {"Horizontal": 0, "Vertical": 1, "Checker": 2}
            self._compiled_shader["direction"].value = dir_map.get(self.get_parameter("direction"), 0)
        if "blend_width" in self._compiled_shader:
            self._compiled_shader["blend_width"].value = self.get_parameter("blend_width")
        if "mix_amount" in self._compiled_shader:
            self._compiled_shader["mix_amount"].value = self.get_parameter("mix_amount")
        if "offset" in self._compiled_shader:
            self._compiled_shader["offset"].value = self.get_parameter("offset")
