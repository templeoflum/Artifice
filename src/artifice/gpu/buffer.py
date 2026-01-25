"""GPU buffer wrapper.

Provides a unified interface for GPU buffers (uniform, storage, etc.)
across different backends.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from artifice.gpu.backend import BufferUsage


@dataclass
class Buffer:
    """GPU buffer wrapper for uniform and storage data.

    Buffers are used to pass parameters to shaders (uniform buffers)
    or for general-purpose GPU storage (storage buffers).

    Attributes:
        size: Buffer size in bytes
        usage: Buffer usage hint
        handle: Backend-specific buffer handle
    """

    size: int
    usage: BufferUsage
    handle: Any = field(repr=False)

    # Optional name for debugging
    name: str = ""
    _backend: Any = field(default=None, repr=False)

    def write(self, data: bytes, offset: int = 0) -> None:
        """Write raw bytes to buffer.

        Args:
            data: Bytes to write
            offset: Byte offset in buffer
        """
        if offset + len(data) > self.size:
            raise ValueError(
                f"Write of {len(data)} bytes at offset {offset} "
                f"exceeds buffer size {self.size}"
            )
        self.handle.write(data, offset=offset)

    def write_floats(self, values: list[float] | NDArray, offset: int = 0) -> None:
        """Write float values to buffer.

        Args:
            values: Float values to write
            offset: Byte offset in buffer
        """
        if isinstance(values, np.ndarray):
            data = values.astype(np.float32).tobytes()
        else:
            data = struct.pack(f"{len(values)}f", *values)
        self.write(data, offset)

    def write_ints(self, values: list[int] | NDArray, offset: int = 0) -> None:
        """Write integer values to buffer.

        Args:
            values: Integer values to write
            offset: Byte offset in buffer
        """
        if isinstance(values, np.ndarray):
            data = values.astype(np.int32).tobytes()
        else:
            data = struct.pack(f"{len(values)}i", *values)
        self.write(data, offset)

    def write_struct(self, **kwargs: float | int | bool) -> None:
        """Write a struct of named values to buffer.

        Values are packed in the order provided. Each value is padded
        to 4 bytes (GLSL std140 layout basics).

        Args:
            **kwargs: Named values to write
        """
        data = b""
        for value in kwargs.values():
            if isinstance(value, bool):
                data += struct.pack("i", int(value))
            elif isinstance(value, int):
                data += struct.pack("i", value)
            elif isinstance(value, float):
                data += struct.pack("f", value)
            else:
                raise TypeError(f"Unsupported type: {type(value)}")

        self.write(data)

    def read(self, size: int = -1, offset: int = 0) -> bytes:
        """Read raw bytes from buffer.

        Args:
            size: Number of bytes to read (-1 for all)
            offset: Byte offset to start reading

        Returns:
            Buffer contents as bytes
        """
        if size < 0:
            size = self.size - offset
        return self.handle.read(size=size, offset=offset)

    def bind(self, binding: int) -> None:
        """Bind buffer to a binding point.

        Args:
            binding: Binding point index
        """
        self.handle.bind_to_uniform_block(binding)

    def bind_storage(self, binding: int) -> None:
        """Bind buffer as shader storage buffer.

        Args:
            binding: Binding point index
        """
        self.handle.bind_to_storage_buffer(binding)

    def clear(self) -> None:
        """Clear buffer to zeros."""
        self.write(b"\x00" * self.size)


class UniformBuffer:
    """Higher-level uniform buffer with named fields.

    Provides a structured way to define and update shader uniforms.
    """

    def __init__(self, backend: Any, layout: dict[str, type]):
        """Initialize uniform buffer.

        Args:
            backend: GPU backend
            layout: Dict mapping field names to types (float, int, bool)
        """
        self._backend = backend
        self._layout = layout
        self._values: dict[str, float | int | bool] = {}
        self._dirty = True

        # Calculate size (4 bytes per field for std140)
        size = len(layout) * 4

        from artifice.gpu.backend import BufferUsage
        self._buffer = backend.create_buffer(size, BufferUsage.UNIFORM)

        # Initialize defaults
        for name, field_type in layout.items():
            if field_type == bool:
                self._values[name] = False
            elif field_type == int:
                self._values[name] = 0
            else:
                self._values[name] = 0.0

    def __setitem__(self, name: str, value: float | int | bool) -> None:
        """Set a uniform value.

        Args:
            name: Field name
            value: New value
        """
        if name not in self._layout:
            raise KeyError(f"Unknown uniform field: {name}")

        if self._values.get(name) != value:
            self._values[name] = value
            self._dirty = True

    def __getitem__(self, name: str) -> float | int | bool:
        """Get a uniform value.

        Args:
            name: Field name

        Returns:
            Current value
        """
        return self._values[name]

    def update(self, **kwargs: float | int | bool) -> None:
        """Update multiple uniform values.

        Args:
            **kwargs: Field name/value pairs
        """
        for name, value in kwargs.items():
            self[name] = value

    def sync(self) -> None:
        """Upload dirty values to GPU."""
        if not self._dirty:
            return

        # Pack values in layout order
        data = b""
        for name, field_type in self._layout.items():
            value = self._values[name]
            if field_type == bool:
                data += struct.pack("i", int(value))
            elif field_type == int:
                data += struct.pack("i", value)
            else:
                data += struct.pack("f", value)

        self._buffer.write(data)
        self._dirty = False

    def bind(self, binding: int) -> None:
        """Sync and bind to a uniform block binding point.

        Args:
            binding: Binding point index
        """
        self.sync()
        self._buffer.bind(binding)

    @property
    def buffer(self) -> Buffer:
        """Return the underlying buffer."""
        return self._buffer
