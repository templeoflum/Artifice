"""GPU subsystem for real-time rendering.

This module provides GPU-accelerated processing via compute shaders.
The architecture is backend-agnostic, supporting ModernGL (OpenGL 4.3+)
with future support for wgpu-py (Vulkan/Metal/DX12).
"""

from artifice.gpu.backend import GPUBackend, TextureFormat, BufferUsage
from artifice.gpu.texture import Texture, TexturePool
from artifice.gpu.buffer import Buffer, UniformBuffer

__all__ = [
    # Backend
    "GPUBackend",
    "TextureFormat",
    "BufferUsage",
    # Texture
    "Texture",
    "TexturePool",
    # Buffer
    "Buffer",
    "UniformBuffer",
]


def create_backend(backend_type: str = "moderngl") -> GPUBackend:
    """Create a GPU backend.

    Args:
        backend_type: Backend type ("moderngl" or "wgpu")

    Returns:
        Initialized GPU backend
    """
    if backend_type == "moderngl":
        from artifice.gpu.moderngl_backend import ModernGLBackend
        return ModernGLBackend(standalone=True)
    elif backend_type == "wgpu":
        raise NotImplementedError("wgpu backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
