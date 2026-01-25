"""ModernGL (OpenGL 4.3+) backend implementation.

This is the primary GPU backend using OpenGL compute shaders via ModernGL.
Requires OpenGL 4.3+ for compute shader support.
"""

from __future__ import annotations

from typing import Any

import moderngl
import numpy as np

from artifice.gpu.backend import (
    GPUBackend,
    TextureFormat,
    BufferUsage,
)
from artifice.gpu.texture import Texture
from artifice.gpu.buffer import Buffer


class ModernGLBackend(GPUBackend):
    """OpenGL 4.3+ backend via ModernGL.

    This backend uses OpenGL compute shaders for GPU processing.
    It can run standalone (headless) or integrate with Qt's OpenGL context.
    """

    # Map TextureFormat to ModernGL format strings
    FORMAT_MAP = {
        TextureFormat.R32F: (1, "r32f"),
        TextureFormat.RG32F: (2, "rg32f"),
        TextureFormat.RGBA32F: (4, "rgba32f"),
        TextureFormat.R16F: (1, "r16f"),
        TextureFormat.RGBA16F: (4, "rgba16f"),
    }

    def __init__(self, standalone: bool = True):
        """Initialize ModernGL backend.

        Args:
            standalone: If True, create a standalone (headless) context.
                       If False, use the current OpenGL context (e.g., from Qt).
        """
        self._standalone = standalone
        self._ctx: moderngl.Context | None = None
        self._initialized = False

        # Shader cache
        self._compute_shaders: dict[str, moderngl.ComputeShader] = {}
        self._render_programs: dict[str, moderngl.Program] = {}

        # Full-screen quad for blitting
        self._quad_vao: moderngl.VertexArray | None = None
        self._quad_vbo: moderngl.Buffer | None = None

    def initialize(self) -> None:
        """Initialize the OpenGL context."""
        if self._initialized:
            return

        if self._standalone:
            # Create headless context
            self._ctx = moderngl.create_standalone_context(require=430)
        else:
            # Use existing context (must be current)
            self._ctx = moderngl.create_context(require=430)

        # Check for compute shader support
        if not self._ctx.version_code >= 430:
            raise RuntimeError(
                f"OpenGL 4.3+ required for compute shaders, "
                f"got {self._ctx.version_code}"
            )

        # Mark as initialized BEFORE creating quad (which uses ctx property)
        self._initialized = True

        # Create full-screen quad for blitting
        self._create_quad()

    def shutdown(self) -> None:
        """Release all OpenGL resources."""
        if not self._initialized:
            return

        # Clean up cached shaders
        self._compute_shaders.clear()
        self._render_programs.clear()

        # Clean up quad
        if self._quad_vao:
            self._quad_vao.release()
        if self._quad_vbo:
            self._quad_vbo.release()

        # Release context (if standalone)
        if self._standalone and self._ctx:
            self._ctx.release()

        self._ctx = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return True if context is initialized."""
        return self._initialized

    @property
    def ctx(self) -> moderngl.Context:
        """Return the ModernGL context."""
        if not self._initialized:
            raise RuntimeError("GPU backend not initialized")
        return self._ctx

    # Texture operations

    def create_texture(
        self,
        width: int,
        height: int,
        format: TextureFormat = TextureFormat.RGBA32F,
    ) -> Texture:
        """Create a GPU texture."""
        if format not in self.FORMAT_MAP:
            raise ValueError(f"Unsupported texture format: {format}")

        channels, _ = self.FORMAT_MAP[format]

        # Create ModernGL texture
        mgl_texture = self.ctx.texture(
            (width, height),
            channels,
            dtype="f4" if "32" in format.name else "f2",
        )

        # Set texture parameters for compute shader access
        mgl_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        mgl_texture.repeat_x = False
        mgl_texture.repeat_y = False

        return Texture(
            width=width,
            height=height,
            channels=channels,
            format=format,
            handle=mgl_texture,
            _backend=self,
        )

    def destroy_texture(self, texture: Texture) -> None:
        """Destroy a GPU texture."""
        if texture.handle:
            texture.handle.release()

    # Buffer operations

    def create_buffer(
        self,
        size: int,
        usage: BufferUsage = BufferUsage.UNIFORM,
    ) -> Buffer:
        """Create a GPU buffer."""
        # ModernGL doesn't differentiate usage at creation time
        mgl_buffer = self.ctx.buffer(reserve=size, dynamic=True)

        return Buffer(
            size=size,
            usage=usage,
            handle=mgl_buffer,
            _backend=self,
        )

    def destroy_buffer(self, buffer: Buffer) -> None:
        """Destroy a GPU buffer."""
        if buffer.handle:
            buffer.handle.release()

    # Compute shader operations

    def compile_compute_shader(self, source: str) -> moderngl.ComputeShader:
        """Compile a compute shader from GLSL source."""
        return self.ctx.compute_shader(source)

    def dispatch_compute(
        self,
        shader: moderngl.ComputeShader,
        groups_x: int,
        groups_y: int,
        groups_z: int = 1,
    ) -> None:
        """Dispatch a compute shader."""
        shader.run(groups_x, groups_y, groups_z)

    # Render shader operations

    def compile_render_shader(
        self,
        vertex_source: str,
        fragment_source: str,
    ) -> moderngl.Program:
        """Compile a vertex/fragment shader pair."""
        return self.ctx.program(
            vertex_shader=vertex_source,
            fragment_shader=fragment_source,
        )

    def blit_to_screen(
        self,
        texture: Texture,
        shader: moderngl.Program,
        viewport: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Render a texture to the screen using a full-screen quad."""
        if viewport:
            self.ctx.viewport = viewport

        # Bind texture to sampler
        texture.handle.use(location=0)

        # Set uniform
        if "tex" in shader:
            shader["tex"].value = 0

        # Render quad
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    # Synchronization

    def sync(self) -> None:
        """Wait for all GPU operations to complete."""
        self.ctx.finish()

    def create_fence(self) -> Any:
        """Create a GPU fence."""
        # ModernGL doesn't have direct fence support
        # We use memory barrier + finish as alternative
        self.ctx.memory_barrier()
        return None

    def wait_fence(self, fence: Any, timeout_ns: int = 1_000_000_000) -> bool:
        """Wait for fence (no-op with current ModernGL)."""
        self.ctx.finish()
        return True

    # Memory barriers

    def memory_barrier(self) -> None:
        """Insert a memory barrier for compute shader synchronization."""
        self.ctx.memory_barrier()

    # Private methods

    def _create_quad(self) -> None:
        """Create a full-screen quad for texture blitting."""
        # Simple vertex shader
        vertex_src = """
        #version 430

        in vec2 in_position;
        out vec2 uv;

        void main() {
            uv = in_position * 0.5 + 0.5;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """

        # Simple fragment shader
        fragment_src = """
        #version 430

        uniform sampler2D tex;
        in vec2 uv;
        out vec4 fragColor;

        void main() {
            fragColor = texture(tex, uv);
        }
        """

        # Compile blit shader
        self._blit_program = self.ctx.program(
            vertex_shader=vertex_src,
            fragment_shader=fragment_src,
        )

        # Create quad vertices (triangle strip)
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)

        self._quad_vbo = self.ctx.buffer(vertices.tobytes())
        self._quad_vao = self.ctx.vertex_array(
            self._blit_program,
            [(self._quad_vbo, "2f", "in_position")],
        )

    def get_blit_program(self) -> moderngl.Program:
        """Return the default blit shader program."""
        return self._blit_program


# Pre-built shader snippets for common operations

FULLSCREEN_QUAD_VERT = """
#version 430

in vec2 in_position;
out vec2 uv;

void main() {
    uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

TEXTURE_DISPLAY_FRAG = """
#version 430

uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;

void main() {
    fragColor = texture(tex, uv);
}
"""
