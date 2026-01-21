"""
Image loader node.

Loads images from disk into ImageBuffer format.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from artifice.core.data_types import ColorSpace, ImageBuffer
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType
from artifice.core.registry import register_node


@register_node
class ImageLoaderNode(Node):
    """
    Load an image from disk.

    Supports common image formats: PNG, JPG, JPEG, TIFF, TIF, WebP, BMP, GIF.
    Output is an ImageBuffer in RGB colorspace with float32 values [0, 1].
    """

    name = "Image Loader"
    category = "I/O"
    description = "Load an image file from disk"
    icon = "file-image"
    _abstract = False

    def define_ports(self) -> None:
        """Define output port for loaded image."""
        self.add_output(
            "image",
            port_type=PortType.IMAGE,
            description="Loaded image",
        )

    def define_parameters(self) -> None:
        """Define file path parameter."""
        self.add_parameter(
            "path",
            param_type=ParameterType.FILEPATH,
            default="",
            description="Path to image file",
        )

    def process(self) -> None:
        """Load the image and output as ImageBuffer."""
        path_str = self.get_parameter("path")

        if not path_str:
            raise ValueError("No file path specified")

        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Check extension
        valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp", ".gif"}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(valid_extensions)}"
            )

        # Load with PIL
        with Image.open(path) as img:
            # Convert to RGB if necessary
            if img.mode == "RGBA":
                # Composite onto white background for now
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array
            data = np.array(img, dtype=np.float32) / 255.0

        # Create ImageBuffer (from HWC format)
        buffer = ImageBuffer.from_hwc(data, colorspace=ColorSpace.RGB)

        # Store image info in metadata
        buffer.metadata["source_path"] = str(path)
        buffer.metadata["original_size"] = (img.width, img.height)

        self.set_output_value("image", buffer)
