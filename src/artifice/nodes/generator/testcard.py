"""
Test card generator node.

Generates a procedural test card image for visualizing node effects.
"""

from __future__ import annotations

import numpy as np

from artifice.core.data_types import ColorSpace, ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


def _generate_perlin_noise(height: int, width: int, scale: float = 32.0) -> np.ndarray:
    """
    Generate simple Perlin-like noise using interpolated random values.

    Args:
        height: Output height.
        width: Output width.
        scale: Frequency scale for the noise.

    Returns:
        2D float32 array with values in [0, 1].
    """
    # Create low-res random grid
    grid_h = max(2, int(height / scale) + 1)
    grid_w = max(2, int(width / scale) + 1)
    grid = np.random.rand(grid_h, grid_w).astype(np.float32)

    # Create coordinate arrays for interpolation
    y = np.linspace(0, grid_h - 1, height)
    x = np.linspace(0, grid_w - 1, width)

    # Get integer and fractional parts
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, grid_h - 1)
    x1 = np.minimum(x0 + 1, grid_w - 1)

    # Fractional parts with smoothstep
    fy = y - y0
    fx = x - x0
    fy = fy * fy * (3 - 2 * fy)  # Smoothstep
    fx = fx * fx * (3 - 2 * fx)

    # Create meshgrid
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')
    y0_grid, x0_grid = np.meshgrid(y0, x0, indexing='ij')
    y1_grid, x1_grid = np.meshgrid(y1, x1, indexing='ij')

    # Bilinear interpolation
    n00 = grid[y0_grid, x0_grid]
    n01 = grid[y0_grid, x1_grid]
    n10 = grid[y1_grid, x0_grid]
    n11 = grid[y1_grid, x1_grid]

    n0 = n00 * (1 - fx_grid) + n01 * fx_grid
    n1 = n10 * (1 - fx_grid) + n11 * fx_grid

    return (n0 * (1 - fy_grid) + n1 * fy_grid).astype(np.float32)


def _generate_zone_plate(height: int, width: int, frequency: float = 50.0) -> np.ndarray:
    """
    Generate a zone plate (Fresnel pattern / concentric sine rings).

    This creates a frequency sweep pattern that's excellent for testing
    aliasing and frequency-domain operations.

    Args:
        height: Output height.
        width: Output width.
        frequency: Maximum frequency at the edges.

    Returns:
        2D float32 array with values in [0, 1].
    """
    center_y = height / 2
    center_x = width / 2
    y, x = np.ogrid[:height, :width]

    # Distance squared from center, normalized
    max_r = min(center_y, center_x)
    r2 = ((x - center_x) ** 2 + (y - center_y) ** 2) / (max_r ** 2)

    # Zone plate formula: sin(frequency * r^2)
    plate = np.sin(frequency * r2 * np.pi)

    # Normalize to [0, 1]
    return ((plate + 1) / 2).astype(np.float32)


def _hue_to_rgb(hue: float) -> tuple[float, float, float]:
    """Convert hue (0-1) to RGB."""
    c = 1.0
    x = c * (1 - abs((hue * 6) % 2 - 1))

    if hue < 1/6:
        return (c, x, 0.0)
    elif hue < 2/6:
        return (x, c, 0.0)
    elif hue < 3/6:
        return (0.0, c, x)
    elif hue < 4/6:
        return (0.0, x, c)
    elif hue < 5/6:
        return (x, 0.0, c)
    else:
        return (c, 0.0, x)


def generate_test_card(size: int = 512, seed: int | None = None) -> ImageBuffer:
    """
    Generate a procedural test card image.

    The test card contains multiple elements designed to help visualize
    the effects of different node types:

    Layout (512x512 default, 4 rows x 4 cols grid):
    +-------------+-------------+-------------+-------------+
    |             |             |             |             |
    |  Color Bars | Checkerboard|   Diagonal  |   Zone      |
    |  (RGBCMYWK) |   (8x8)     |    Lines    |   Plate     |
    |             |             |             |             |
    +-------------+-------------+-------------+-------------+
    |             |             |             |             |
    |    Step     |   Radial    |    Fine     |   Perlin    |
    |   Wedge     |  Gradient   | Checkerboard|   Noise     |
    |             |             |             |             |
    +-------------+-------------+-------------+-------------+
    |                                                       |
    |              Rainbow Hue Sweep (full width)           |
    |                                                       |
    +-------------------------------------------------------+
    |                                                       |
    |           Grayscale Gradient (full width)             |
    |                                                       |
    +-------------------------------------------------------+

    Args:
        size: Output image size (default 512, will be square).
        seed: Random seed for reproducible noise patterns.

    Returns:
        ImageBuffer containing the test card in RGB colorspace.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create RGB image (C, H, W format)
    data = np.zeros((3, size, size), dtype=np.float32)

    # Grid dimensions: 4 columns, 4 rows
    cell_w = size // 4
    cell_h = size // 4

    # === ROW 0 ===

    # [0,0] Color Bars (RGBCMYWK + Gray)
    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.0, 1.0),  # Magenta
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 1.0, 1.0),  # White
        (0.0, 0.0, 0.0),  # Black
    ]
    bar_width = cell_w // len(colors)
    for i, (r, g, b) in enumerate(colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < len(colors) - 1 else cell_w
        data[0, :cell_h, x_start:x_end] = r
        data[1, :cell_h, x_start:x_end] = g
        data[2, :cell_h, x_start:x_end] = b

    # [0,1] Checkerboard (8x8)
    checker_size = cell_w // 8
    for cy in range(8):
        for cx in range(8):
            val = 1.0 if (cy + cx) % 2 == 0 else 0.0
            y_start = cy * checker_size
            y_end = (cy + 1) * checker_size
            x_start = cell_w + cx * checker_size
            x_end = cell_w + (cx + 1) * checker_size
            data[:, y_start:y_end, x_start:x_end] = val

    # [0,2] Diagonal Lines
    y_coords, x_coords = np.ogrid[:cell_h, cell_w*2:cell_w*3]
    stripe_width = cell_w // 8
    diagonal = ((y_coords + (x_coords - cell_w*2)) // stripe_width) % 2
    data[:, :cell_h, cell_w*2:cell_w*3] = diagonal.astype(np.float32)

    # [0,3] Zone Plate
    zone = _generate_zone_plate(cell_h, cell_w, frequency=25.0)
    data[:, :cell_h, cell_w*3:cell_w*4] = zone

    # === ROW 1 ===

    # [1,0] Step Wedge (discrete gray levels for quantization testing)
    num_steps = 8
    step_width = cell_w // num_steps
    for i in range(num_steps):
        val = i / (num_steps - 1)  # 0.0, 0.143, 0.286, ... 1.0
        x_start = i * step_width
        x_end = (i + 1) * step_width if i < num_steps - 1 else cell_w
        data[:, cell_h:cell_h*2, x_start:x_end] = val

    # [1,1] Radial Gradient
    center_y = cell_h + cell_h // 2
    center_x = cell_w + cell_w // 2
    y_coords, x_coords = np.ogrid[cell_h:cell_h*2, cell_w:cell_w*2]
    dist = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_dist = np.sqrt(2) * cell_w / 2
    radial = 1.0 - np.clip(dist / max_dist, 0, 1)
    data[:, cell_h:cell_h*2, cell_w:cell_w*2] = radial.astype(np.float32)

    # [1,2] Fine Checkerboard (16x16)
    fine_checker_size = cell_w // 16
    for cy in range(16):
        for cx in range(16):
            val = 1.0 if (cy + cx) % 2 == 0 else 0.0
            y_start = cell_h + cy * fine_checker_size
            y_end = cell_h + (cy + 1) * fine_checker_size
            x_start = cell_w*2 + cx * fine_checker_size
            x_end = cell_w*2 + (cx + 1) * fine_checker_size
            data[:, y_start:y_end, x_start:x_end] = val

    # [1,3] Perlin Noise
    noise = _generate_perlin_noise(cell_h, cell_w, scale=cell_w / 8)
    data[:, cell_h:cell_h*2, cell_w*3:cell_w*4] = noise

    # === ROW 2: Rainbow Hue Sweep (full width) ===
    for x in range(size):
        hue = x / size
        r, g, b = _hue_to_rgb(hue)
        data[0, cell_h*2:cell_h*3, x] = r
        data[1, cell_h*2:cell_h*3, x] = g
        data[2, cell_h*2:cell_h*3, x] = b

    # === ROW 3: Grayscale Gradient (full width) ===
    gray_ramp = np.linspace(0, 1, size, dtype=np.float32)
    gray_ramp = np.tile(gray_ramp, (cell_h, 1))
    data[:, cell_h*3:size, :] = gray_ramp

    # Create the ImageBuffer
    buffer = ImageBuffer(data=data, colorspace=ColorSpace.RGB)
    buffer.metadata["generator"] = "TestCard"
    buffer.metadata["size"] = size
    buffer.metadata["seed"] = seed

    return buffer


@register_node
class TestCardNode(Node):
    """
    Generate a procedural test card image.

    Creates a calibration image containing multiple visual elements designed
    to help visualize and test the effects of different processing nodes:

    - Color bars (R, G, B, C, M, Y, W, K) - test color/channel operations
    - Checkerboard patterns (coarse and fine) - test frequency/DCT/FFT effects
    - Diagonal lines - test directional operations (pixel sort, wavelets)
    - Zone plate (concentric sine rings) - test frequency response/aliasing
    - Step wedge (discrete gray levels) - test quantization level merging
    - Radial gradient - test smooth gradients and circular operations
    - Perlin noise - test segmentation and texture effects
    - Rainbow hue sweep - test color space conversions
    - Grayscale gradient - test tonal response across full width

    This node has no inputs and generates a fresh test card each execution.
    Use the seed parameter for reproducible random elements.
    """

    name = "Test Card"
    category = "Generator"
    description = "Generate a procedural test card for calibration"
    icon = "grid"
    _abstract = False

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
            description="Random seed for reproducible patterns",
        )
        self.add_parameter(
            "use_seed",
            param_type=ParameterType.BOOL,
            default=True,
            description="Use fixed seed (disable for random each time)",
        )

    def process(self) -> None:
        """Generate the test card image."""
        size = self.get_parameter("size")
        use_seed = self.get_parameter("use_seed")
        seed = self.get_parameter("seed") if use_seed else None

        buffer = generate_test_card(size=size, seed=seed)
        self.set_output_value("image", buffer)
