"""
Tests for I/O nodes (ImageLoader, ImageSaver).

End-to-end tests for loading and saving images through the node system.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from artifice.core.data_types import ColorSpace, ImageBuffer
from artifice.core.graph import NodeGraph
from artifice.nodes.io.loader import ImageLoaderNode
from artifice.nodes.io.saver import ImageSaverNode
from artifice.nodes.utility.passthrough import PassThroughNode


class TestImageLoaderNode:
    """Tests for ImageLoaderNode."""

    def test_load_png(self, sample_image_path):
        """Test loading PNG image."""
        node = ImageLoaderNode()
        node.set_parameter("path", str(sample_image_path))

        node.execute()

        output = node.outputs["image"].get_value()
        assert output is not None
        assert isinstance(output, ImageBuffer)
        assert output.colorspace == ColorSpace.RGB
        assert output.channels == 3

    def test_load_jpg(self, temp_dir, sample_image_data):
        """Test loading JPEG image."""
        # Create JPEG
        path = temp_dir / "test.jpg"
        uint8_data = (sample_image_data * 255).astype(np.uint8)
        Image.fromarray(uint8_data).save(path, quality=95)

        node = ImageLoaderNode()
        node.set_parameter("path", str(path))
        node.execute()

        output = node.outputs["image"].get_value()
        assert output is not None
        assert output.channels == 3

    def test_load_webp(self, temp_dir, sample_image_data):
        """Test loading WebP image."""
        path = temp_dir / "test.webp"
        uint8_data = (sample_image_data * 255).astype(np.uint8)
        Image.fromarray(uint8_data).save(path, quality=95)

        node = ImageLoaderNode()
        node.set_parameter("path", str(path))
        node.execute()

        output = node.outputs["image"].get_value()
        assert output is not None

    def test_load_tiff(self, temp_dir, sample_image_data):
        """Test loading TIFF image."""
        path = temp_dir / "test.tiff"
        uint8_data = (sample_image_data * 255).astype(np.uint8)
        Image.fromarray(uint8_data).save(path)

        node = ImageLoaderNode()
        node.set_parameter("path", str(path))
        node.execute()

        output = node.outputs["image"].get_value()
        assert output is not None

    def test_load_rgba_composites(self, temp_dir):
        """Test that RGBA images are composited to RGB."""
        # Create RGBA image with transparency
        path = temp_dir / "test_rgba.png"
        data = np.zeros((64, 64, 4), dtype=np.uint8)
        data[:, :, 0] = 255  # Red
        data[:, :, 3] = 128  # 50% alpha
        Image.fromarray(data, mode="RGBA").save(path)

        node = ImageLoaderNode()
        node.set_parameter("path", str(path))
        node.execute()

        output = node.outputs["image"].get_value()
        assert output is not None
        assert output.channels == 3  # Converted to RGB

    def test_load_grayscale_converts(self, temp_dir):
        """Test that grayscale images are converted to RGB."""
        path = temp_dir / "test_gray.png"
        data = np.zeros((64, 64), dtype=np.uint8)
        data[16:48, 16:48] = 255
        Image.fromarray(data, mode="L").save(path)

        node = ImageLoaderNode()
        node.set_parameter("path", str(path))
        node.execute()

        output = node.outputs["image"].get_value()
        assert output is not None
        assert output.channels == 3

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        node = ImageLoaderNode()
        node.set_parameter("path", "/nonexistent/path.png")

        result = node.execute()

        assert not result
        assert node._error is not None
        assert "not found" in node._error.lower()

    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported format raises error."""
        path = temp_dir / "test.xyz"
        path.write_text("not an image")

        node = ImageLoaderNode()
        node.set_parameter("path", str(path))

        result = node.execute()

        assert not result
        assert "Unsupported" in node._error

    def test_metadata_preserved(self, sample_image_path):
        """Test that metadata is stored in output buffer."""
        node = ImageLoaderNode()
        node.set_parameter("path", str(sample_image_path))
        node.execute()

        output = node.outputs["image"].get_value()
        assert "source_path" in output.metadata
        assert output.metadata["source_path"] == str(sample_image_path)


class TestImageSaverNode:
    """Tests for ImageSaverNode."""

    def test_save_png(self, temp_dir, sample_image_buffer):
        """Test saving PNG image."""
        path = temp_dir / "output.png"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))

        node.execute()

        assert path.exists()

        # Verify image can be loaded back
        with Image.open(path) as img:
            assert img.size == (sample_image_buffer.width, sample_image_buffer.height)

    def test_save_jpg(self, temp_dir, sample_image_buffer):
        """Test saving JPEG image."""
        path = temp_dir / "output.jpg"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))
        node.set_parameter("quality", 90)

        node.execute()

        assert path.exists()

    def test_save_webp(self, temp_dir, sample_image_buffer):
        """Test saving WebP image."""
        path = temp_dir / "output.webp"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))

        node.execute()

        assert path.exists()

    def test_save_tiff(self, temp_dir, sample_image_buffer):
        """Test saving TIFF image."""
        path = temp_dir / "output.tiff"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))

        node.execute()

        assert path.exists()

    def test_save_bmp(self, temp_dir, sample_image_buffer):
        """Test saving BMP image."""
        path = temp_dir / "output.bmp"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))

        node.execute()

        assert path.exists()

    def test_creates_parent_directory(self, temp_dir, sample_image_buffer):
        """Test that parent directories are created."""
        path = temp_dir / "subdir" / "nested" / "output.png"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))

        node.execute()

        assert path.exists()

    def test_save_unsupported_format(self, temp_dir, sample_image_buffer):
        """Test saving to unsupported format raises error."""
        path = temp_dir / "output.xyz"

        node = ImageSaverNode()
        node.inputs["image"].default = sample_image_buffer
        node.set_parameter("path", str(path))

        result = node.execute()

        assert not result
        assert "Unsupported" in node._error


class TestIOPipelineIntegration:
    """Integration tests for I/O pipeline."""

    def test_load_process_save_roundtrip(self, temp_dir, sample_image_path):
        """Test complete load -> process -> save pipeline."""
        graph = NodeGraph()

        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", str(sample_image_path))

        passthrough = graph.add_node(PassThroughNode())

        output_path = temp_dir / "output.png"
        saver = graph.add_node(ImageSaverNode())
        saver.set_parameter("path", str(output_path))

        graph.connect(loader, "image", passthrough, "image")
        graph.connect(passthrough, "image", saver, "image")

        results = graph.execute()

        assert all(results.values())
        assert output_path.exists()

        # Compare original and output
        original = np.array(Image.open(sample_image_path))
        result = np.array(Image.open(output_path))

        np.testing.assert_array_equal(original, result)

    def test_format_conversion(self, temp_dir, sample_image_path):
        """Test converting between formats through pipeline."""
        graph = NodeGraph()

        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", str(sample_image_path))  # PNG

        jpg_path = temp_dir / "output.jpg"
        saver = graph.add_node(ImageSaverNode())
        saver.set_parameter("path", str(jpg_path))
        saver.set_parameter("quality", 95)

        graph.connect(loader, "image", saver, "image")

        results = graph.execute()

        assert all(results.values())
        assert jpg_path.exists()

        # JPEG should be smaller than PNG (usually)
        # and loadable
        with Image.open(jpg_path) as img:
            assert img.format == "JPEG"

    def test_image_dimensions_preserved(self, temp_dir):
        """Test that image dimensions are preserved through pipeline."""
        # Create specific size image
        size = (123, 456)  # Odd dimensions
        data = np.random.rand(size[1], size[0], 3).astype(np.float32)
        input_path = temp_dir / "input.png"
        uint8_data = (data * 255).astype(np.uint8)
        Image.fromarray(uint8_data).save(input_path)

        graph = NodeGraph()

        loader = graph.add_node(ImageLoaderNode())
        loader.set_parameter("path", str(input_path))

        output_path = temp_dir / "output.png"
        saver = graph.add_node(ImageSaverNode())
        saver.set_parameter("path", str(output_path))

        graph.connect(loader, "image", saver, "image")
        graph.execute()

        with Image.open(output_path) as img:
            assert img.size == size
