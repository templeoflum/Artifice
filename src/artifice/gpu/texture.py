"""GPU texture wrapper.

Provides a unified interface for GPU textures across different backends,
with zero-copy operations where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from artifice.gpu.backend import TextureFormat


@dataclass
class Texture:
    """GPU texture wrapper with zero-copy operations.

    Textures store image data on the GPU and are the primary data type
    passed between nodes in the GPU pipeline.

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        channels: Number of color channels (1, 2, or 4)
        format: Texture format enum
        handle: Backend-specific texture handle
    """

    width: int
    height: int
    channels: int
    format: TextureFormat
    handle: Any = field(repr=False)

    # Metadata for tracking (optional)
    name: str = ""
    _backend: Any = field(default=None, repr=False)

    def upload(self, data: NDArray[np.float32]) -> None:
        """Upload CPU data to GPU texture.

        This should be minimized in real-time paths - prefer keeping
        data on GPU between nodes.

        Args:
            data: NumPy array with shape (H, W, C) or (H, W) for single channel.
                  Must be float32 in range [0, 1].
        """
        # Ensure correct shape
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        if data.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"Data shape {data.shape} doesn't match texture "
                f"({self.height}, {self.width}, {self.channels})"
            )

        # Ensure contiguous float32
        data = np.ascontiguousarray(data, dtype=np.float32)

        # Backend-specific upload
        self.handle.write(data.tobytes())

    def download(self) -> NDArray[np.float32]:
        """Download GPU texture to CPU.

        This is expensive and should only be used for export,
        not in the real-time rendering path.

        Returns:
            NumPy array with shape (H, W, C), float32
        """
        raw = self.handle.read()
        data = np.frombuffer(raw, dtype=np.float32)
        return data.reshape(self.height, self.width, self.channels)

    def bind_as_image(
        self,
        unit: int,
        access: str = "read",
    ) -> None:
        """Bind texture for compute shader image access.

        Args:
            unit: Image unit to bind to (0-7 typically)
            access: Access mode - "read", "write", or "readwrite"
        """
        read = access in ("read", "readwrite")
        write = access in ("write", "readwrite")
        self.handle.bind_to_image(unit, read=read, write=write)

    def bind_as_sampler(self, unit: int) -> None:
        """Bind texture for sampler access in fragment shaders.

        Args:
            unit: Texture unit to bind to
        """
        self.handle.use(location=unit)

    def clear(self, value: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Clear texture to a solid color.

        Args:
            value: Clear color (R, G, B, A) in [0, 1]
        """
        # Create clear data
        channels = min(len(value), self.channels)
        clear_data = np.full(
            (self.height, self.width, self.channels),
            value[:channels],
            dtype=np.float32,
        )
        self.upload(clear_data)

    def copy_from(self, source: Texture) -> None:
        """Copy data from another texture.

        Both textures must have the same dimensions.

        Args:
            source: Source texture to copy from
        """
        if (source.width, source.height) != (self.width, self.height):
            raise ValueError("Texture dimensions must match for copy")

        # For now, go through CPU - optimize later with GPU copy
        data = source.download()
        self.upload(data)

    @property
    def size(self) -> tuple[int, int]:
        """Return (width, height) tuple."""
        return (self.width, self.height)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (height, width, channels) tuple (NumPy convention)."""
        return (self.height, self.width, self.channels)

    def __del__(self):
        """Clean up GPU resources."""
        # The backend handles actual cleanup
        pass


class TexturePool:
    """Pool of reusable GPU textures to avoid allocation overhead.

    In real-time rendering, allocating new textures every frame is
    expensive. The pool maintains a cache of textures by size/format
    for reuse.
    """

    def __init__(self, backend: Any):
        """Initialize texture pool.

        Args:
            backend: GPU backend for texture creation
        """
        self._backend = backend
        self._pool: dict[tuple[int, int, int], list[Texture]] = {}
        self._in_use: set[int] = set()

    def acquire(
        self,
        width: int,
        height: int,
        channels: int = 4,
    ) -> Texture:
        """Acquire a texture from the pool.

        Returns an existing texture if available, otherwise creates new.

        Args:
            width: Texture width
            height: Texture height
            channels: Number of channels (default 4 for RGBA)

        Returns:
            GPU texture
        """
        from artifice.gpu.backend import TextureFormat

        key = (width, height, channels)

        # Check pool for available texture
        if key in self._pool and self._pool[key]:
            texture = self._pool[key].pop()
            self._in_use.add(id(texture))
            return texture

        # Create new texture
        format_map = {
            1: TextureFormat.R32F,
            2: TextureFormat.RG32F,
            4: TextureFormat.RGBA32F,
        }
        texture = self._backend.create_texture(
            width, height, format_map.get(channels, TextureFormat.RGBA32F)
        )
        self._in_use.add(id(texture))
        return texture

    def release(self, texture: Texture) -> None:
        """Return a texture to the pool for reuse.

        Args:
            texture: Texture to release
        """
        tex_id = id(texture)
        if tex_id not in self._in_use:
            return  # Already released or not from this pool

        self._in_use.discard(tex_id)

        key = (texture.width, texture.height, texture.channels)
        if key not in self._pool:
            self._pool[key] = []

        self._pool[key].append(texture)

    def clear(self) -> None:
        """Release all pooled textures."""
        for textures in self._pool.values():
            for texture in textures:
                self._backend.destroy_texture(texture)

        self._pool.clear()
        self._in_use.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Return pool statistics.

        Returns:
            Dict with 'pooled', 'in_use', 'total' counts
        """
        pooled = sum(len(t) for t in self._pool.values())
        return {
            "pooled": pooled,
            "in_use": len(self._in_use),
            "total": pooled + len(self._in_use),
        }
