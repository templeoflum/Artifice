"""
Image saver node.

Saves ImageBuffer to disk in various formats.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from artifice.core.data_types import ImageBuffer


def _get_default_output_dir() -> str:
    """Get the default output directory (project output folder)."""
    # Get the project root (artifice package is in src/artifice)
    output_dir = Path(__file__).parent.parent.parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class ImageSaverNode(Node):
    """
    Save an image to disk.

    Supports common image formats: PNG, JPG, JPEG, TIFF, TIF, WebP, BMP.
    Format is determined by file extension.
    """

    name = "Image Saver"
    category = "I/O"
    description = "Save an image to disk"
    icon = "save"
    _abstract = False

    def define_ports(self) -> None:
        """Define input port for image to save."""
        self.add_input(
            "image",
            port_type=PortType.IMAGE,
            description="Image to save",
            required=True,
        )

    def define_parameters(self) -> None:
        """Define file path and quality parameters."""
        self.add_parameter(
            "path",
            param_type=ParameterType.FILEPATH,
            default="",
            description="Output file path",
            file_filter="PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tiff *.tif);;BMP (*.bmp);;All Files (*)",
            is_save_path=True,
            default_directory=_get_default_output_dir(),
        )
        self.add_parameter(
            "quality",
            param_type=ParameterType.INT,
            default=95,
            min_value=1,
            max_value=100,
            description="JPEG/WebP quality (1-100)",
        )
        self.add_parameter(
            "png_compression",
            param_type=ParameterType.INT,
            default=6,
            min_value=0,
            max_value=9,
            description="PNG compression level (0-9)",
        )

    def process(self) -> None:
        """Save the input image to disk."""
        path_str = self.get_parameter("path")

        if not path_str:
            raise ValueError("No output path specified")

        path = Path(path_str)

        # Get input image
        buffer: ImageBuffer = self.get_input_value("image")

        if buffer is None:
            raise ValueError("No input image")

        # Convert to uint8 HWC format
        uint8_data = buffer.to_uint8()

        # Handle single-channel (grayscale)
        if uint8_data.shape[2] == 1:
            uint8_data = uint8_data[:, :, 0]
            mode = "L"
        else:
            mode = "RGB"

        # Create PIL Image
        img = Image.fromarray(uint8_data, mode=mode)

        # Create output directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from extension
        ext = path.suffix.lower()
        quality = self.get_parameter("quality")
        png_compression = self.get_parameter("png_compression")

        save_kwargs = {}

        if ext in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif ext == ".png":
            save_kwargs["compress_level"] = png_compression
        elif ext == ".webp":
            save_kwargs["quality"] = quality
        elif ext in {".tiff", ".tif"}:
            save_kwargs["compression"] = "tiff_deflate"
        elif ext == ".bmp":
            pass  # No special options
        else:
            raise ValueError(
                f"Unsupported output format: {ext}. "
                f"Supported: .png, .jpg, .jpeg, .tiff, .tif, .webp, .bmp"
            )

        # Save the image
        img.save(path, **save_kwargs)
