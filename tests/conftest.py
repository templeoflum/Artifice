"""
Pytest configuration and shared fixtures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from artifice.core.data_types import ImageBuffer
from artifice.core.registry import NodeRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the node registry before and after each test."""
    NodeRegistry.clear()
    # Re-register the built-in nodes
    from artifice.nodes.io.loader import ImageLoaderNode
    from artifice.nodes.io.saver import ImageSaverNode
    from artifice.nodes.utility.passthrough import PassThroughNode

    NodeRegistry.register(ImageLoaderNode)
    NodeRegistry.register(ImageSaverNode)
    NodeRegistry.register(PassThroughNode)

    yield
    NodeRegistry.clear()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_data():
    """Create sample RGB image data."""
    # Create a gradient test pattern
    height, width = 64, 64
    data = np.zeros((height, width, 3), dtype=np.float32)

    # Red gradient horizontal
    data[:, :, 0] = np.linspace(0, 1, width)[np.newaxis, :]
    # Green gradient vertical
    data[:, :, 1] = np.linspace(0, 1, height)[:, np.newaxis]
    # Blue constant
    data[:, :, 2] = 0.5

    return data


@pytest.fixture
def sample_image_buffer(sample_image_data):
    """Create a sample ImageBuffer."""
    return ImageBuffer.from_hwc(sample_image_data)


@pytest.fixture
def sample_image_path(temp_dir, sample_image_data):
    """Create a sample image file and return its path."""
    path = temp_dir / "test_image.png"

    # Convert to uint8 and save
    uint8_data = (sample_image_data * 255).astype(np.uint8)
    img = Image.fromarray(uint8_data, mode="RGB")
    img.save(path)

    return path


@pytest.fixture
def large_image_data():
    """Create larger image data for performance tests."""
    height, width = 256, 256
    return np.random.rand(height, width, 3).astype(np.float32)


@pytest.fixture
def large_image_buffer(large_image_data):
    """Create a larger ImageBuffer."""
    return ImageBuffer.from_hwc(large_image_data)
