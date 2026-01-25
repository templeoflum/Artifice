"""GPU-accelerated node base class.

Provides the foundation for nodes that execute on the GPU via compute shaders.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType

if TYPE_CHECKING:
    from artifice.gpu.backend import GPUBackend
    from artifice.gpu.texture import Texture


@dataclass
class ShaderUniform:
    """Describes a shader uniform parameter.

    Maps a node parameter to a shader uniform variable.
    """

    name: str                    # Uniform name in shader
    param_name: str              # Node parameter name
    uniform_type: str = "float"  # "float", "int", "bool", "vec2", etc.
    default: Any = 0.0


class GPUNode(Node):
    """Base class for GPU-accelerated nodes.

    GPU nodes execute their processing on the GPU using compute shaders.
    They can still have CPU fallback implementations for compatibility.

    Subclasses must define:
        - shader_file: Path to the GLSL compute shader
        - uniforms: List of ShaderUniform mappings
        - define_ports(): Port definitions
        - define_parameters(): Parameter definitions

    Example:
        class BitFlipGPU(GPUNode):
            name = "Bit Flip (GPU)"
            category = "Corruption"
            shader_file = "corruption/bitflip.glsl"
            uniforms = [
                ShaderUniform("probability", "probability", "float", 0.01),
                ShaderUniform("seed", "seed", "int", 0),
            ]
    """

    # Class attributes - override in subclasses
    shader_file: ClassVar[str] = ""  # Path relative to gpu/shaders/
    uniforms: ClassVar[list[ShaderUniform]] = []
    local_size: ClassVar[tuple[int, int, int]] = (16, 16, 1)  # Workgroup size

    # GPU state
    _compiled_shader: Any = None
    _backend: GPUBackend | None = None
    _input_textures: dict[str, Texture] = field(default_factory=dict)
    _output_textures: dict[str, Texture] = field(default_factory=dict)

    def __init__(self):
        super().__init__()
        self._compiled_shader = None
        self._backend = None
        self._input_textures = {}
        self._output_textures = {}

    def process(self) -> None:
        """CPU fallback - not used for GPU nodes.

        GPU nodes use execute_gpu() instead. This method exists to satisfy
        the Node base class abstract method requirement.
        """
        # GPU nodes don't use the CPU process path
        # They override execute_gpu() instead
        pass

    def compile(self, backend: GPUBackend) -> None:
        """Compile the compute shader.

        Called once when the node is added to a GPU graph.

        Args:
            backend: GPU backend to compile with
        """
        if not self.shader_file:
            raise ValueError(f"{self.__class__.__name__} has no shader_file defined")

        self._backend = backend
        shader_source = backend.load_shader(self.shader_file)
        self._compiled_shader = backend.compile_compute_shader(shader_source)

    @property
    def is_compiled(self) -> bool:
        """Return True if shader is compiled and ready."""
        return self._compiled_shader is not None

    def allocate_textures(self, backend: GPUBackend, width: int, height: int) -> None:
        """Allocate GPU textures for inputs/outputs.

        Args:
            backend: GPU backend
            width: Image width
            height: Image height
        """
        from artifice.gpu.backend import TextureFormat

        # Allocate output textures
        for port_name, port in self.outputs.items():
            if port.port_type == PortType.IMAGE:
                texture = backend.create_texture(width, height, TextureFormat.RGBA32F)
                self._output_textures[port_name] = texture

    def set_input_texture(self, port_name: str, texture: Texture) -> None:
        """Set the input texture for a port.

        Args:
            port_name: Name of the input port
            texture: GPU texture to use
        """
        self._input_textures[port_name] = texture

    def get_output_texture(self, port_name: str = "image") -> Texture | None:
        """Get the output texture for a port.

        Args:
            port_name: Name of the output port

        Returns:
            GPU texture or None if not allocated
        """
        return self._output_textures.get(port_name)

    def execute_gpu(self, backend: GPUBackend) -> None:
        """Execute the node on the GPU.

        This is the main GPU execution path. It:
        1. Binds input textures
        2. Binds output textures
        3. Uploads uniform parameters
        4. Dispatches the compute shader

        Args:
            backend: GPU backend
        """
        if not self.is_compiled:
            self.compile(backend)

        # Get output size from first input texture
        width, height = self._get_output_size()

        # Bind input textures
        binding = 0
        for port_name in self.inputs:
            if port_name in self._input_textures:
                texture = self._input_textures[port_name]
                texture.bind_as_image(binding, "read")
                binding += 1

        # Bind output textures
        for port_name in self.outputs:
            if port_name in self._output_textures:
                texture = self._output_textures[port_name]
                texture.bind_as_image(binding, "write")
                binding += 1

        # Upload uniforms
        self._upload_uniforms()

        # Calculate dispatch size
        local_x, local_y, local_z = self.local_size
        groups_x = (width + local_x - 1) // local_x
        groups_y = (height + local_y - 1) // local_y
        groups_z = 1

        # Dispatch
        backend.dispatch_compute(self._compiled_shader, groups_x, groups_y, groups_z)

        # Memory barrier to ensure writes are visible
        backend.memory_barrier()

    def _get_output_size(self) -> tuple[int, int]:
        """Get the output texture size.

        Returns:
            (width, height) tuple
        """
        # Use first input texture size
        for texture in self._input_textures.values():
            return (texture.width, texture.height)

        # Use first output texture size
        for texture in self._output_textures.values():
            return (texture.width, texture.height)

        # Default
        return (512, 512)

    def _upload_uniforms(self) -> None:
        """Upload parameter values to shader uniforms."""
        for uniform in self.uniforms:
            value = self.get_parameter(uniform.param_name)

            # Convert to shader value
            if uniform.uniform_type == "bool":
                value = 1 if value else 0

            # Set uniform on shader
            if uniform.name in self._compiled_shader:
                self._compiled_shader[uniform.name].value = value

    def release(self) -> None:
        """Release GPU resources."""
        # Release output textures
        if self._backend:
            for texture in self._output_textures.values():
                self._backend.destroy_texture(texture)

        self._output_textures.clear()
        self._input_textures.clear()
        self._compiled_shader = None
        self._backend = None


class GPUPassthroughNode(GPUNode):
    """A GPU node that passes input directly to output.

    Useful for testing the GPU pipeline without any processing.
    """

    name = "GPU Passthrough"
    category = "Utility"
    description = "Pass image through GPU pipeline unchanged"
    shader_file = ""  # No shader needed

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        pass

    def execute_gpu(self, backend: GPUBackend) -> None:
        """Just copy input to output."""
        if "image" in self._input_textures and "image" in self._output_textures:
            self._output_textures["image"].copy_from(self._input_textures["image"])


# ============================================================================
# GPU Node Implementations
# ============================================================================


class BitFlipGPUNode(GPUNode):
    """GPU-accelerated bit flip corruption."""

    name = "Bit Flip (GPU)"
    category = "Corruption"
    description = "Randomly flip bits in image data (GPU accelerated)"
    shader_file = "corruption/bitflip.glsl"
    uniforms = [
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


class BitShiftGPUNode(GPUNode):
    """GPU-accelerated bit shift corruption."""

    name = "Bit Shift (GPU)"
    category = "Corruption"
    description = "Shift bits in image data (GPU accelerated)"
    shader_file = "corruption/bitshift.glsl"
    uniforms = [
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


class QuantizeGPUNode(GPUNode):
    """GPU-accelerated quantization."""

    name = "Quantize (GPU)"
    category = "Quantization"
    description = "Reduce color precision (GPU accelerated)"
    shader_file = "quantization/quantize.glsl"
    uniforms = [
        ShaderUniform("levels", "levels", "int", 8),
        ShaderUniform("mode", "mode", "int", 0),
        ShaderUniform("dither", "dither", "bool", False),
        ShaderUniform("dither_strength", "dither_strength", "float", 1.0),
    ]

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "levels",
            param_type=ParameterType.INT,
            default=8,
            min_value=2,
            max_value=256,
            description="Number of quantization levels",
        )
        self.add_parameter(
            "mode",
            param_type=ParameterType.ENUM,
            default="uniform",
            choices=["uniform", "adaptive", "per_channel"],
            description="Quantization mode",
        )
        self.add_parameter(
            "dither",
            param_type=ParameterType.BOOL,
            default=False,
            description="Apply ordered dithering",
        )
        self.add_parameter(
            "dither_strength",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=2.0,
            description="Dithering strength",
        )

    def _upload_uniforms(self) -> None:
        """Upload uniforms with enum conversion."""
        # Convert enum to int
        mode_str = self.get_parameter("mode")
        mode_map = {"uniform": 0, "adaptive": 1, "per_channel": 2}
        mode_int = mode_map.get(mode_str, 0)

        if "mode" in self._compiled_shader:
            self._compiled_shader["mode"].value = mode_int

        # Upload other uniforms normally
        for uniform in self.uniforms:
            if uniform.param_name == "mode":
                continue  # Already handled

            value = self.get_parameter(uniform.param_name)
            if uniform.uniform_type == "bool":
                value = 1 if value else 0

            if uniform.name in self._compiled_shader:
                self._compiled_shader[uniform.name].value = value


class ColorSpaceGPUNode(GPUNode):
    """GPU-accelerated color space conversion."""

    name = "Color Space (GPU)"
    category = "Color"
    description = "Convert between color spaces (GPU accelerated)"
    shader_file = "color/colorspace.glsl"
    uniforms = [
        ShaderUniform("from_space", "from_space", "int", 0),
        ShaderUniform("to_space", "to_space", "int", 1),
    ]

    # Color space name to shader ID mapping
    SPACE_MAP = {
        "RGB": 0,
        "HSV": 1,
        "HSL": 2,
        "LAB": 3,
        "XYZ": 4,
        "YCbCr": 5,
        "YUV": 6,
        "LUV": 7,
    }

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        spaces = list(self.SPACE_MAP.keys())
        self.add_parameter(
            "from_space",
            param_type=ParameterType.ENUM,
            default="RGB",
            choices=spaces,
            description="Source color space",
        )
        self.add_parameter(
            "to_space",
            param_type=ParameterType.ENUM,
            default="HSV",
            choices=spaces,
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
