"""Push-based streaming executor for real-time GPU rendering.

This module provides the StreamExecutor which drives the GPU node graph
at a target frame rate for real-time rendering.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from artifice.core.graph import NodeGraph
    from artifice.core.gpu_node import GPUNode
    from artifice.gpu.backend import GPUBackend
    from artifice.gpu.texture import Texture


class ExecutorState(Enum):
    """Executor state machine."""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


@dataclass
class FrameStats:
    """Statistics for a single frame."""
    frame_number: int = 0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0
    dropped: bool = False


@dataclass
class ExecutorStats:
    """Aggregate executor statistics."""
    frames_rendered: int = 0
    frames_dropped: int = 0
    avg_frame_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    current_fps: float = 0.0
    target_fps: float = 60.0

    # Rolling window for FPS calculation
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=60))

    def update(self, frame: FrameStats) -> None:
        """Update stats with a new frame."""
        self.frames_rendered += 1
        if frame.dropped:
            self.frames_dropped += 1

        self._frame_times.append(frame.total_time_ms)

        # Calculate averages
        if self._frame_times:
            self.avg_frame_time_ms = sum(self._frame_times) / len(self._frame_times)
            self.current_fps = 1000.0 / self.avg_frame_time_ms if self.avg_frame_time_ms > 0 else 0


class TripleBuffer:
    """Triple buffer for smooth frame delivery.

    Uses three buffers to allow:
    - One being displayed
    - One ready to display next
    - One being written by GPU
    """

    def __init__(self, backend: GPUBackend, width: int, height: int):
        """Initialize triple buffer.

        Args:
            backend: GPU backend
            width: Buffer width
            height: Buffer height
        """
        from artifice.gpu.backend import TextureFormat

        self._backend = backend
        self._width = width
        self._height = height

        # Create three textures
        self._textures = [
            backend.create_texture(width, height, TextureFormat.RGBA32F)
            for _ in range(3)
        ]

        self._write_idx = 0    # GPU writing here
        self._ready_idx = 1    # Ready for display
        self._display_idx = 2  # Currently displayed

    def get_write_texture(self) -> Texture:
        """Get texture for GPU to write to."""
        return self._textures[self._write_idx]

    def get_display_texture(self) -> Texture:
        """Get texture currently being displayed."""
        return self._textures[self._display_idx]

    def get_ready_texture(self) -> Texture:
        """Get texture ready for next display."""
        return self._textures[self._ready_idx]

    def swap(self) -> None:
        """Rotate buffers after frame complete."""
        # Rotate indices
        self._write_idx = (self._write_idx + 1) % 3
        self._ready_idx = (self._ready_idx + 1) % 3
        self._display_idx = (self._display_idx + 1) % 3

    def resize(self, width: int, height: int) -> None:
        """Resize all buffers."""
        if width == self._width and height == self._height:
            return

        from artifice.gpu.backend import TextureFormat

        # Release old textures
        for tex in self._textures:
            self._backend.destroy_texture(tex)

        # Create new ones
        self._textures = [
            self._backend.create_texture(width, height, TextureFormat.RGBA32F)
            for _ in range(3)
        ]

        self._width = width
        self._height = height

    def release(self) -> None:
        """Release all GPU resources."""
        for tex in self._textures:
            self._backend.destroy_texture(tex)
        self._textures.clear()

    @property
    def size(self) -> tuple[int, int]:
        """Return (width, height)."""
        return (self._width, self._height)


class StreamExecutor(QObject):
    """Push-based GPU graph executor for real-time rendering.

    Drives the node graph at a target FPS, pushing frames through
    the GPU pipeline and signaling when they're ready for display.

    Signals:
        frame_ready: Emitted when a new frame is ready for display
        stats_updated: Emitted with updated statistics
        state_changed: Emitted when executor state changes
    """

    frame_ready = Signal()
    stats_updated = Signal(ExecutorStats)
    state_changed = Signal(ExecutorState)

    def __init__(
        self,
        backend: GPUBackend,
        parent: QObject | None = None,
    ):
        """Initialize stream executor.

        Args:
            backend: GPU backend
            parent: Qt parent object
        """
        super().__init__(parent)

        self._backend = backend
        self._graph: NodeGraph | None = None
        self._execution_order: list[GPUNode] = []

        # State
        self._state = ExecutorState.STOPPED
        self._target_fps = 60.0
        self._frame_number = 0

        # Triple buffer
        self._triple_buffer: TripleBuffer | None = None
        self._output_width = 1920
        self._output_height = 1080

        # Stats
        self._stats = ExecutorStats(target_fps=self._target_fps)

        # Async task
        self._task: asyncio.Task | None = None

        # Callbacks
        self._on_frame_callbacks: list[Callable[[Texture], None]] = []

    def set_graph(self, graph: NodeGraph) -> None:
        """Set the node graph to execute.

        Args:
            graph: Node graph
        """
        self._graph = graph
        self._rebuild_execution_order()

    def set_output_size(self, width: int, height: int) -> None:
        """Set output resolution.

        Args:
            width: Output width
            height: Output height
        """
        self._output_width = width
        self._output_height = height

        if self._triple_buffer:
            self._triple_buffer.resize(width, height)

    def set_target_fps(self, fps: float) -> None:
        """Set target frame rate.

        Args:
            fps: Target FPS (1-240)
        """
        self._target_fps = max(1.0, min(240.0, fps))
        self._stats.target_fps = self._target_fps

    @property
    def state(self) -> ExecutorState:
        """Return current executor state."""
        return self._state

    @property
    def stats(self) -> ExecutorStats:
        """Return current statistics."""
        return self._stats

    @property
    def triple_buffer(self) -> TripleBuffer | None:
        """Return the triple buffer."""
        return self._triple_buffer

    def add_frame_callback(self, callback: Callable[[Texture], None]) -> None:
        """Add a callback to be called when a frame is ready.

        Args:
            callback: Function taking the output texture
        """
        self._on_frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[Texture], None]) -> None:
        """Remove a frame callback."""
        if callback in self._on_frame_callbacks:
            self._on_frame_callbacks.remove(callback)

    async def start(self) -> None:
        """Start streaming frames.

        This is an async method that runs until stop() is called.
        """
        if self._state == ExecutorState.RUNNING:
            return

        # Initialize triple buffer
        if not self._triple_buffer:
            self._triple_buffer = TripleBuffer(
                self._backend,
                self._output_width,
                self._output_height,
            )

        # Compile nodes
        self._compile_nodes()

        # Allocate node textures
        self._allocate_textures()

        self._state = ExecutorState.RUNNING
        self.state_changed.emit(self._state)

        frame_budget = 1.0 / self._target_fps

        while self._state == ExecutorState.RUNNING:
            frame_start = time.perf_counter()

            # Execute graph
            exec_start = time.perf_counter()
            output_texture = self._execute_frame()
            exec_time = (time.perf_counter() - exec_start) * 1000.0

            # Copy to triple buffer
            if output_texture and self._triple_buffer:
                write_tex = self._triple_buffer.get_write_texture()
                write_tex.copy_from(output_texture)
                self._triple_buffer.swap()

            # Notify callbacks
            display_tex = self._triple_buffer.get_display_texture() if self._triple_buffer else None
            for callback in self._on_frame_callbacks:
                callback(display_tex)

            # Emit signal
            self.frame_ready.emit()

            # Calculate frame time
            total_time = (time.perf_counter() - frame_start) * 1000.0

            # Update stats
            frame_stats = FrameStats(
                frame_number=self._frame_number,
                execution_time_ms=exec_time,
                total_time_ms=total_time,
                dropped=total_time > (frame_budget * 1000.0),
            )
            self._stats.update(frame_stats)
            self._frame_number += 1

            # Emit stats periodically
            if self._frame_number % 10 == 0:
                self.stats_updated.emit(self._stats)

            # Maintain frame rate
            elapsed = time.perf_counter() - frame_start
            sleep_time = frame_budget - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Let other tasks run even if we're behind
                await asyncio.sleep(0)

    def stop(self) -> None:
        """Stop streaming."""
        self._state = ExecutorState.STOPPED
        self.state_changed.emit(self._state)

    def pause(self) -> None:
        """Pause streaming."""
        if self._state == ExecutorState.RUNNING:
            self._state = ExecutorState.PAUSED
            self.state_changed.emit(self._state)

    def resume(self) -> None:
        """Resume from pause."""
        if self._state == ExecutorState.PAUSED:
            self._state = ExecutorState.RUNNING
            self.state_changed.emit(self._state)

    def execute_single_frame(self) -> Texture | None:
        """Execute a single frame synchronously.

        Useful for non-realtime rendering or testing.

        Returns:
            Output texture or None
        """
        if not self._triple_buffer:
            self._triple_buffer = TripleBuffer(
                self._backend,
                self._output_width,
                self._output_height,
            )

        self._compile_nodes()
        self._allocate_textures()

        output = self._execute_frame()

        if output and self._triple_buffer:
            write_tex = self._triple_buffer.get_write_texture()
            write_tex.copy_from(output)
            self._triple_buffer.swap()
            return self._triple_buffer.get_display_texture()

        return output

    def _execute_frame(self) -> Texture | None:
        """Execute one frame of the graph.

        Returns:
            Output texture from the final node
        """
        if not self._execution_order:
            return None

        # Execute nodes in order
        prev_output: Texture | None = None

        for node in self._execution_order:
            # Connect input from previous node
            if prev_output and "image" in node._input_textures:
                node.set_input_texture("image", prev_output)

            # Execute
            node.execute_gpu(self._backend)

            # Get output for next node
            prev_output = node.get_output_texture("image")

        return prev_output

    def _compile_nodes(self) -> None:
        """Compile all GPU nodes in the graph."""
        for node in self._execution_order:
            if not node.is_compiled:
                node.compile(self._backend)

    def _allocate_textures(self) -> None:
        """Allocate textures for all nodes."""
        for node in self._execution_order:
            node.allocate_textures(
                self._backend,
                self._output_width,
                self._output_height,
            )

    def _rebuild_execution_order(self) -> None:
        """Rebuild the execution order from the graph."""
        from artifice.core.gpu_node import GPUNode

        self._execution_order.clear()

        if not self._graph:
            return

        # Get topological order
        order = self._graph.get_execution_order()

        # Filter to GPU nodes only
        for node_id in order:
            node = self._graph.nodes.get(node_id)
            if isinstance(node, GPUNode):
                self._execution_order.append(node)

    def release(self) -> None:
        """Release all GPU resources."""
        # Stop first
        self.stop()

        # Release triple buffer
        if self._triple_buffer:
            self._triple_buffer.release()
            self._triple_buffer = None

        # Release node resources
        for node in self._execution_order:
            node.release()

        self._execution_order.clear()
