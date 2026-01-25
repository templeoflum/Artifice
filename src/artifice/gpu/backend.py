"""Abstract GPU backend interface.

This module defines the abstract interface for GPU backends, allowing
the engine to work with different GPU APIs (OpenGL, Vulkan, etc.)
through a unified interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artifice.gpu.texture import Texture
    from artifice.gpu.buffer import Buffer


class TextureFormat(Enum):
    """GPU texture formats."""
    R32F = auto()      # Single channel float32
    RG32F = auto()     # Two channel float32
    RGBA32F = auto()   # Four channel float32 (standard)
    R16F = auto()      # Single channel float16
    RGBA16F = auto()   # Four channel float16


class BufferUsage(Enum):
    """GPU buffer usage hints."""
    UNIFORM = auto()    # Uniform buffer for shader parameters
    STORAGE = auto()    # Storage buffer for compute read/write
    VERTEX = auto()     # Vertex buffer
    INDEX = auto()      # Index buffer


class GPUBackend(ABC):
    """Abstract GPU backend interface.

    All GPU operations go through this interface, allowing swapping
    between OpenGL, Vulkan, or other backends.
    """

    # Shader directory relative to this file
    SHADER_DIR = Path(__file__).parent / "shaders"

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the GPU context.

        Must be called before any other GPU operations.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown and release all GPU resources."""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Return True if GPU context is initialized."""
        pass

    # Texture operations
    @abstractmethod
    def create_texture(
        self,
        width: int,
        height: int,
        format: TextureFormat = TextureFormat.RGBA32F,
    ) -> Texture:
        """Create a GPU texture.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            format: Pixel format

        Returns:
            GPU texture handle
        """
        pass

    @abstractmethod
    def destroy_texture(self, texture: Texture) -> None:
        """Destroy a GPU texture and free its memory."""
        pass

    # Buffer operations
    @abstractmethod
    def create_buffer(
        self,
        size: int,
        usage: BufferUsage = BufferUsage.UNIFORM,
    ) -> Buffer:
        """Create a GPU buffer.

        Args:
            size: Buffer size in bytes
            usage: Buffer usage hint

        Returns:
            GPU buffer handle
        """
        pass

    @abstractmethod
    def destroy_buffer(self, buffer: Buffer) -> None:
        """Destroy a GPU buffer and free its memory."""
        pass

    # Compute shader operations
    @abstractmethod
    def compile_compute_shader(self, source: str) -> Any:
        """Compile a compute shader from source.

        Args:
            source: GLSL compute shader source code

        Returns:
            Compiled shader handle (backend-specific)
        """
        pass

    @abstractmethod
    def dispatch_compute(
        self,
        shader: Any,
        groups_x: int,
        groups_y: int,
        groups_z: int = 1,
    ) -> None:
        """Dispatch a compute shader.

        Args:
            shader: Compiled compute shader
            groups_x: Number of workgroups in X dimension
            groups_y: Number of workgroups in Y dimension
            groups_z: Number of workgroups in Z dimension
        """
        pass

    # Render operations (for preview display)
    @abstractmethod
    def compile_render_shader(
        self,
        vertex_source: str,
        fragment_source: str,
    ) -> Any:
        """Compile a vertex/fragment shader pair.

        Args:
            vertex_source: GLSL vertex shader source
            fragment_source: GLSL fragment shader source

        Returns:
            Compiled shader program handle
        """
        pass

    @abstractmethod
    def blit_to_screen(
        self,
        texture: Texture,
        shader: Any,
        viewport: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Render a texture to the screen.

        Args:
            texture: Source texture to display
            shader: Compiled render shader
            viewport: Optional (x, y, width, height) viewport
        """
        pass

    # Synchronization
    @abstractmethod
    def sync(self) -> None:
        """Wait for all GPU operations to complete.

        This is a full GPU sync and should be used sparingly.
        """
        pass

    @abstractmethod
    def create_fence(self) -> Any:
        """Create a GPU fence for async sync.

        Returns:
            Fence handle (backend-specific)
        """
        pass

    @abstractmethod
    def wait_fence(self, fence: Any, timeout_ns: int = 1_000_000_000) -> bool:
        """Wait for a fence to complete.

        Args:
            fence: Fence handle
            timeout_ns: Timeout in nanoseconds

        Returns:
            True if fence completed, False if timeout
        """
        pass

    # Utility
    def load_shader(self, path: str | Path) -> str:
        """Load shader source from file.

        Args:
            path: Path relative to shaders/ directory, or absolute path

        Returns:
            Shader source code
        """
        shader_path = Path(path)
        if not shader_path.is_absolute():
            shader_path = self.SHADER_DIR / shader_path

        return shader_path.read_text()

    def load_compute_shader(self, path: str | Path) -> Any:
        """Load and compile a compute shader from file.

        Args:
            path: Path relative to shaders/ directory

        Returns:
            Compiled compute shader
        """
        source = self.load_shader(path)
        return self.compile_compute_shader(source)
