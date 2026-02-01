"""GPU-accelerated generator nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from artifice.core.gpu_node import GPUNode, ShaderUniform
from artifice.core.node import ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node

if TYPE_CHECKING:
    from artifice.gpu.backend import GPUBackend


@register_node
class TestCardGPUNode(GPUNode):
    """GPU-accelerated test card generator.

    Generates a procedural test card entirely on the GPU for real-time
    preview. The test card contains multiple visual elements:

    - Color bars (R, G, B, C, M, Y, W, K)
    - Checkerboard patterns (coarse and fine)
    - Diagonal lines
    - Zone plate (Fresnel pattern)
    - Step wedge (discrete gray levels)
    - Radial gradient
    - Perlin-like noise
    - Rainbow hue sweep
    - Grayscale gradient
    """

    name = "Test Card (GPU)"
    category = "Generator"
    description = "Generate a procedural test card (GPU accelerated)"
    shader_file = "generator/testcard.glsl"
    _abstract = False

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("size", "size", "int", 512),
        ShaderUniform("seed", "seed", "int", 42),
        ShaderUniform("time", "time", "float", 0.0),
    ]

    def define_ports(self) -> None:
        """Define output port for generated image."""
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Generated test card image",
        )

    def define_parameters(self) -> None:
        """Define test card parameters."""
        self.add_parameter(
            "size",
            param_type=ParameterType.INT,
            default=512,
            min_value=128,
            max_value=2048,
            step=128,
            description="Output image size (square)",
        )
        self.add_parameter(
            "seed",
            param_type=ParameterType.INT,
            default=42,
            min_value=0,
            max_value=999999,
            description="Random seed for noise patterns",
        )
        self.add_parameter(
            "time",
            param_type=ParameterType.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=1000.0,
            description="Time value for animated effects",
        )

    def execute_gpu(self, backend: GPUBackend) -> None:
        """Execute test card generation on GPU.

        This overrides the base execute_gpu because this node has no inputs -
        it generates an image from scratch.
        """
        if not self.is_compiled:
            self.compile(backend)

        # Get size from parameters
        size = self.get_parameter("size")

        # Ensure output texture is allocated at correct size
        from artifice.gpu.backend import TextureFormat

        if "image" not in self._output_textures:
            self._output_textures["image"] = backend.create_texture(
                size, size, TextureFormat.RGBA32F
            )
        else:
            # Check if size changed
            tex = self._output_textures["image"]
            if tex.width != size or tex.height != size:
                backend.destroy_texture(tex)
                self._output_textures["image"] = backend.create_texture(
                    size, size, TextureFormat.RGBA32F
                )

        # Bind output texture
        self._output_textures["image"].bind_as_image(0, "write")

        # Upload uniforms
        self._upload_uniforms()

        # Calculate dispatch size
        local_x, local_y = 16, 16
        groups_x = (size + local_x - 1) // local_x
        groups_y = (size + local_y - 1) // local_y

        # Dispatch compute shader
        backend.dispatch_compute(self._compiled_shader, groups_x, groups_y, 1)

        # Memory barrier
        backend.memory_barrier()


@register_node
class NoiseGPUNode(GPUNode):
    """GPU-accelerated procedural noise generator."""

    name = "Noise (GPU)"
    category = "Generator"
    description = "Generate procedural noise patterns (GPU accelerated)"
    shader_file = "generator/noise.glsl"
    _abstract = False

    NOISE_TYPES = ["Perlin", "Simplex", "White", "Value"]

    uniforms: ClassVar[list[ShaderUniform]] = [
        ShaderUniform("size", "size", "int", 512),
        ShaderUniform("noise_type", "noise_type", "int", 0),
        ShaderUniform("scale", "scale", "float", 1.0),
        ShaderUniform("octaves", "octaves", "int", 4),
        ShaderUniform("persistence", "persistence", "float", 0.5),
        ShaderUniform("seed", "seed", "int", 0),
    ]

    def define_ports(self) -> None:
        self.add_output("image", PortType.IMAGE, "Generated noise image")

    def define_parameters(self) -> None:
        self.add_parameter("size", param_type=ParameterType.INT, default=512,
                          min_value=64, max_value=4096, step=64, description="Output size")
        self.add_parameter("noise_type", param_type=ParameterType.ENUM, default="Perlin",
                          choices=self.NOISE_TYPES, description="Noise algorithm")
        self.add_parameter("scale", param_type=ParameterType.FLOAT, default=1.0,
                          min_value=0.01, max_value=100.0, step=0.1, description="Noise scale")
        self.add_parameter("octaves", param_type=ParameterType.INT, default=4,
                          min_value=1, max_value=8, description="Fractal octaves")
        self.add_parameter("persistence", param_type=ParameterType.FLOAT, default=0.5,
                          min_value=0.0, max_value=1.0, step=0.01, description="Amplitude decay")
        self.add_parameter("seed", param_type=ParameterType.INT, default=0,
                          min_value=0, max_value=999999, description="Random seed")

    def _upload_uniforms(self) -> None:
        if "size" in self._compiled_shader:
            self._compiled_shader["size"].value = self.get_parameter("size")
        if "noise_type" in self._compiled_shader:
            noise = self.get_parameter("noise_type")
            self._compiled_shader["noise_type"].value = self.NOISE_TYPES.index(noise) if noise in self.NOISE_TYPES else 0
        if "scale" in self._compiled_shader:
            self._compiled_shader["scale"].value = self.get_parameter("scale")
        if "octaves" in self._compiled_shader:
            self._compiled_shader["octaves"].value = self.get_parameter("octaves")
        if "persistence" in self._compiled_shader:
            self._compiled_shader["persistence"].value = self.get_parameter("persistence")
        if "seed" in self._compiled_shader:
            self._compiled_shader["seed"].value = self.get_parameter("seed")

    def execute_gpu(self, backend: GPUBackend) -> None:
        """Execute noise generation on GPU."""
        if not self.is_compiled:
            self.compile(backend)

        size = self.get_parameter("size")

        from artifice.gpu.backend import TextureFormat

        if "image" not in self._output_textures:
            self._output_textures["image"] = backend.create_texture(size, size, TextureFormat.RGBA32F)
        else:
            tex = self._output_textures["image"]
            if tex.width != size or tex.height != size:
                backend.destroy_texture(tex)
                self._output_textures["image"] = backend.create_texture(size, size, TextureFormat.RGBA32F)

        self._output_textures["image"].bind_as_image(0, "write")
        self._upload_uniforms()

        local_x, local_y = 16, 16
        groups_x = (size + local_x - 1) // local_x
        groups_y = (size + local_y - 1) // local_y

        backend.dispatch_compute(self._compiled_shader, groups_x, groups_y, 1)
        backend.memory_barrier()
