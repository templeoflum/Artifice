# Node Development Guide

This guide explains how to create custom nodes for Artifice Engine. Whether you want to implement a new glitch effect, add support for a new file format, or integrate an external library, this guide covers everything you need to know.

## Table of Contents

- [Node Anatomy](#node-anatomy)
- [Quick Start](#quick-start)
- [Ports and Data Types](#ports-and-data-types)
- [Parameters](#parameters)
- [The Process Method](#the-process-method)
- [Working with Images](#working-with-images)
- [Node Registration](#node-registration)
- [Testing Your Node](#testing-your-node)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)
- [Examples](#examples)

## Node Anatomy

Every node in Artifice Engine consists of:

1. **Metadata**: Name, category, description
2. **Input Ports**: Data the node receives
3. **Output Ports**: Data the node produces
4. **Parameters**: User-configurable settings
5. **Process Method**: The actual processing logic

```python
from artifice.core.node import Node
from artifice.core.port import PortType

class MyNode(Node):
    # Metadata
    CATEGORY = "transform"
    DESCRIPTION = "Brief description for the UI"

    def __init__(self):
        super().__init__("MyNode")

        # Input ports
        self.add_input("image", PortType.IMAGE)

        # Output ports
        self.add_output("image", PortType.IMAGE)

        # Parameters
        self.add_parameter("strength", 0.5, min_value=0.0, max_value=1.0)

    def process(self) -> None:
        # Processing logic
        pass
```

## Quick Start

Let's create a simple brightness adjustment node:

```python
"""Brightness adjustment node."""

from __future__ import annotations

import numpy as np

from artifice.core.node import Node
from artifice.core.port import PortType
from artifice.core.data_types import ImageBuffer


class BrightnessNode(Node):
    """
    Adjusts the brightness of an image.

    Multiplies all pixel values by a brightness factor.
    Values above 1.0 increase brightness, below 1.0 decrease it.
    """

    CATEGORY = "color"
    DESCRIPTION = "Adjust image brightness"

    def __init__(self):
        super().__init__("BrightnessNode")

        # One image input, one image output
        self.add_input("image", PortType.IMAGE)
        self.add_output("image", PortType.IMAGE)

        # Brightness multiplier parameter
        self.add_parameter(
            "brightness",
            default=1.0,
            min_value=0.0,
            max_value=3.0,
            description="Brightness multiplier"
        )

    def process(self) -> None:
        """Apply brightness adjustment."""
        # Get input image
        image = self.get_input_value("image")
        if image is None:
            return

        # Get parameter
        brightness = self.get_parameter("brightness")

        # Apply effect
        result_data = np.clip(image.data * brightness, 0.0, 1.0)

        # Create output
        result = ImageBuffer(
            data=result_data.astype(np.float32),
            colorspace=image.colorspace
        )
        self.set_output_value("image", result)
```

Save this as `src/artifice/nodes/color/brightness.py` and you have a working node!

## Ports and Data Types

### Port Types

Artifice Engine supports several port types:

```python
from artifice.core.port import PortType

PortType.IMAGE    # ImageBuffer - the most common type
PortType.REGIONS  # RegionMap - segmentation results
PortType.FLOAT    # Single float value
PortType.INT      # Single integer value
PortType.STRING   # Text data
PortType.ANY      # Accepts any type (use sparingly)
```

### Adding Ports

```python
def __init__(self):
    super().__init__("MyNode")

    # Input ports (left side of node)
    self.add_input("image", PortType.IMAGE)
    self.add_input("mask", PortType.IMAGE)
    self.add_input("amount", PortType.FLOAT)

    # Output ports (right side of node)
    self.add_output("result", PortType.IMAGE)
    self.add_output("difference", PortType.IMAGE)
```

### Accessing Port Values

```python
def process(self) -> None:
    # Get input values (may be None if not connected)
    image = self.get_input_value("image")
    mask = self.get_input_value("mask")
    amount = self.get_input_value("amount")

    # Check for required inputs
    if image is None:
        return

    # ... processing ...

    # Set output values
    self.set_output_value("result", result_image)
    self.set_output_value("difference", diff_image)
```

## Parameters

Parameters are user-configurable values that control node behavior.

### Basic Parameters

```python
# Float parameter with range
self.add_parameter(
    name="threshold",
    default=0.5,
    min_value=0.0,
    max_value=1.0,
    description="Detection threshold"
)

# Integer parameter
self.add_parameter(
    name="iterations",
    default=3,
    min_value=1,
    max_value=10,
    description="Number of iterations"
)

# Choice parameter (dropdown)
self.add_parameter(
    name="mode",
    default="normal",
    choices=["normal", "multiply", "screen", "overlay"],
    description="Blending mode"
)

# Boolean parameter (checkbox)
self.add_parameter(
    name="invert",
    default=False,
    description="Invert the result"
)

# String parameter
self.add_parameter(
    name="label",
    default="untitled",
    description="Output label"
)
```

### Accessing Parameters

```python
def process(self) -> None:
    threshold = self.get_parameter("threshold")
    iterations = self.get_parameter("iterations")
    mode = self.get_parameter("mode")
    invert = self.get_parameter("invert")
```

### Setting Parameters Programmatically

```python
node = MyNode()
node.set_parameter("threshold", 0.75)
node.set_parameter("mode", "multiply")
```

## The Process Method

The `process()` method is where your node's logic lives. It's called during graph execution.

### Basic Structure

```python
def process(self) -> None:
    """Process inputs and produce outputs."""
    # 1. Get inputs
    image = self.get_input_value("image")

    # 2. Validate inputs
    if image is None:
        return

    # 3. Get parameters
    strength = self.get_parameter("strength")

    # 4. Do processing
    result_data = self._apply_effect(image.data, strength)

    # 5. Set outputs
    result = ImageBuffer(data=result_data, colorspace=image.colorspace)
    self.set_output_value("image", result)
```

### Helper Methods

Keep `process()` clean by extracting algorithms into helper methods:

```python
def process(self) -> None:
    image = self.get_input_value("image")
    if image is None:
        return

    result_data = self._apply_glitch(
        image.data,
        self.get_parameter("intensity"),
        self.get_parameter("mode")
    )

    self.set_output_value("image", ImageBuffer(
        data=result_data,
        colorspace=image.colorspace
    ))

def _apply_glitch(self, data: np.ndarray, intensity: float, mode: str) -> np.ndarray:
    """Apply the glitch effect."""
    if mode == "shift":
        return self._shift_glitch(data, intensity)
    elif mode == "corrupt":
        return self._corrupt_glitch(data, intensity)
    return data

def _shift_glitch(self, data: np.ndarray, intensity: float) -> np.ndarray:
    # Implementation
    pass

def _corrupt_glitch(self, data: np.ndarray, intensity: float) -> np.ndarray:
    # Implementation
    pass
```

## Working with Images

### ImageBuffer Format

Images in Artifice Engine use the `ImageBuffer` class:

```python
from artifice.core.data_types import ImageBuffer

# Create from numpy array
# Shape: (channels, height, width)
# Dtype: float32
# Values: 0.0 to 1.0
data = np.random.rand(3, 256, 256).astype(np.float32)
image = ImageBuffer(data=data, colorspace="RGB")

# Access properties
print(image.shape)      # (3, 256, 256)
print(image.channels)   # 3
print(image.height)     # 256
print(image.width)      # 256
print(image.colorspace) # "RGB"

# Access raw data
raw = image.data  # numpy array
```

### Common Operations

```python
def process(self) -> None:
    image = self.get_input_value("image")
    if image is None:
        return

    data = image.data  # Shape: (C, H, W)

    # Access individual channels
    channel_0 = data[0]  # First channel (H, W)
    channel_1 = data[1]  # Second channel
    channel_2 = data[2]  # Third channel

    # Modify channels
    data[0] = data[0] * 1.2  # Boost first channel

    # Swap channels
    data = data[[2, 1, 0]]  # Reverse channel order

    # Apply per-pixel operation
    data = np.clip(data * 1.5, 0.0, 1.0)

    # Create output
    result = ImageBuffer(data=data.astype(np.float32), colorspace=image.colorspace)
    self.set_output_value("image", result)
```

### Working with Different Color Spaces

```python
def process(self) -> None:
    image = self.get_input_value("image")
    if image is None:
        return

    # Check color space
    if image.colorspace == "RGB":
        # Process as RGB
        pass
    elif image.colorspace == "HSV":
        # data[0] = Hue, data[1] = Saturation, data[2] = Value
        pass
    elif image.colorspace == "YCbCr":
        # data[0] = Luma, data[1] = Cb, data[2] = Cr
        pass
```

## Node Registration

Nodes must be registered to appear in the UI.

### Automatic Registration

Import your node in the category's `__init__.py`:

```python
# src/artifice/nodes/color/__init__.py
from .brightness import BrightnessNode
from .colorspace import ColorSpaceNode
# ... etc
```

Then in the main nodes `__init__.py`:

```python
# src/artifice/nodes/__init__.py
from . import color
from . import io
# ... etc
```

### Manual Registration

```python
from artifice.core.registry import NodeRegistry

# Register a node class
NodeRegistry.register("BrightnessNode", BrightnessNode, category="color")

# Or use decorator
from artifice.core.registry import register_node

@register_node(category="color")
class BrightnessNode(Node):
    ...
```

## Testing Your Node

Every node should have tests. Create a test file in `tests/`:

```python
# tests/test_brightness.py
"""Tests for BrightnessNode."""

import numpy as np
import pytest

from artifice.nodes.color.brightness import BrightnessNode
from artifice.core.data_types import ImageBuffer


class TestBrightnessNode:
    """Tests for BrightnessNode."""

    @pytest.fixture
    def node(self):
        """Create a BrightnessNode instance."""
        return BrightnessNode()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        data = np.full((3, 64, 64), 0.5, dtype=np.float32)
        return ImageBuffer(data=data, colorspace="RGB")

    def test_default_parameters(self, node):
        """Test default parameter values."""
        assert node.get_parameter("brightness") == 1.0

    def test_no_input_returns_none(self, node):
        """Test that processing with no input does nothing."""
        node.process()
        assert node.get_output_value("image") is None

    def test_brightness_increase(self, node, sample_image):
        """Test that brightness > 1 increases values."""
        node.set_parameter("brightness", 2.0)
        node.inputs["image"].set_value(sample_image)
        node.process()

        result = node.get_output_value("image")
        assert result is not None
        # Original was 0.5, doubled should be 1.0 (clamped)
        assert np.allclose(result.data, 1.0)

    def test_brightness_decrease(self, node, sample_image):
        """Test that brightness < 1 decreases values."""
        node.set_parameter("brightness", 0.5)
        node.inputs["image"].set_value(sample_image)
        node.process()

        result = node.get_output_value("image")
        assert result is not None
        # Original was 0.5, halved should be 0.25
        assert np.allclose(result.data, 0.25)

    def test_preserves_colorspace(self, node, sample_image):
        """Test that colorspace is preserved."""
        node.inputs["image"].set_value(sample_image)
        node.process()

        result = node.get_output_value("image")
        assert result.colorspace == sample_image.colorspace

    def test_output_clamped(self, node, sample_image):
        """Test that output values are clamped to [0, 1]."""
        node.set_parameter("brightness", 10.0)
        node.inputs["image"].set_value(sample_image)
        node.process()

        result = node.get_output_value("image")
        assert result.data.max() <= 1.0
        assert result.data.min() >= 0.0
```

Run tests with:

```bash
pytest tests/test_brightness.py -v
```

## Best Practices

### 1. Handle Missing Inputs Gracefully

```python
def process(self) -> None:
    image = self.get_input_value("image")
    if image is None:
        return  # Don't crash, just do nothing
```

### 2. Preserve Metadata

```python
result = ImageBuffer(
    data=result_data,
    colorspace=image.colorspace,  # Preserve color space
    metadata=image.metadata.copy() if image.metadata else {}
)
```

### 3. Ensure Correct Output Type

```python
# Always ensure float32
result_data = result_data.astype(np.float32)

# Always clip to valid range
result_data = np.clip(result_data, 0.0, 1.0)
```

### 4. Use Vectorized Operations

```python
# GOOD: Vectorized
result = data * 2.0

# BAD: Loop-based
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            result[i, j, k] = data[i, j, k] * 2.0
```

### 5. Document Your Node

```python
class MyGlitchNode(Node):
    """
    Apply a custom glitch effect.

    This node creates visual artifacts by manipulating pixel data
    in specific patterns. The effect is controlled by intensity
    and pattern parameters.

    Inputs:
        image: The input image to process.

    Outputs:
        image: The glitched result.

    Parameters:
        intensity: Strength of the effect (0.0-1.0).
        pattern: Type of glitch pattern to apply.
    """
```

### 6. Make Copies When Needed

```python
def process(self) -> None:
    image = self.get_input_value("image")
    if image is None:
        return

    # Make a copy if you need to modify
    data = image.data.copy()
    data[0] *= 1.5  # Now safe to modify

    # Or use operations that create new arrays
    data = image.data * 1.5  # This creates a new array
```

## Advanced Topics

### Multiple Inputs

```python
class BlendNode(Node):
    def __init__(self):
        super().__init__("BlendNode")
        self.add_input("image_a", PortType.IMAGE)
        self.add_input("image_b", PortType.IMAGE)
        self.add_input("mask", PortType.IMAGE)  # Optional
        self.add_output("image", PortType.IMAGE)
        self.add_parameter("blend_mode", "normal", choices=["normal", "multiply", "screen"])
        self.add_parameter("opacity", 1.0, min_value=0.0, max_value=1.0)

    def process(self) -> None:
        a = self.get_input_value("image_a")
        b = self.get_input_value("image_b")
        mask = self.get_input_value("mask")  # May be None

        if a is None or b is None:
            return

        # Use mask if provided, otherwise full blend
        if mask is not None:
            blend_mask = mask.data[0]  # Use first channel
        else:
            blend_mask = np.ones((a.height, a.width))

        # Apply blend
        result = self._blend(a.data, b.data, blend_mask, ...)
```

### Working with Regions

```python
from artifice.core.data_types import RegionMap

class RegionProcessNode(Node):
    def __init__(self):
        super().__init__("RegionProcessNode")
        self.add_input("image", PortType.IMAGE)
        self.add_input("regions", PortType.REGIONS)
        self.add_output("image", PortType.IMAGE)

    def process(self) -> None:
        image = self.get_input_value("image")
        regions = self.get_input_value("regions")

        if image is None or regions is None:
            return

        result = image.data.copy()

        # Process each region differently
        for region_id in range(regions.num_regions):
            mask = regions.labels == region_id
            result[:, mask] = self._process_region(result[:, mask])
```

### Stateful Nodes

Some nodes need to maintain state (use carefully):

```python
class AccumulatorNode(Node):
    def __init__(self):
        super().__init__("AccumulatorNode")
        self._accumulator = None
        self._frame_count = 0

    def process(self) -> None:
        image = self.get_input_value("image")
        if image is None:
            return

        if self._accumulator is None:
            self._accumulator = np.zeros_like(image.data)

        self._accumulator += image.data
        self._frame_count += 1

        result_data = self._accumulator / self._frame_count
        # ...

    def reset(self) -> None:
        """Reset the accumulator state."""
        self._accumulator = None
        self._frame_count = 0
```

## Examples

### Complete Example: Threshold Node

```python
"""Threshold node implementation."""

from __future__ import annotations

import numpy as np

from artifice.core.node import Node
from artifice.core.port import PortType
from artifice.core.data_types import ImageBuffer


class ThresholdNode(Node):
    """
    Apply thresholding to an image.

    Pixels above the threshold become white (1.0),
    pixels below become black (0.0).
    """

    CATEGORY = "color"
    DESCRIPTION = "Binary threshold"

    def __init__(self):
        super().__init__("ThresholdNode")

        self.add_input("image", PortType.IMAGE)
        self.add_output("image", PortType.IMAGE)
        self.add_output("mask", PortType.IMAGE)

        self.add_parameter("threshold", 0.5, min_value=0.0, max_value=1.0)
        self.add_parameter("channel", "luminance",
                          choices=["luminance", "red", "green", "blue", "alpha"])
        self.add_parameter("invert", False)

    def process(self) -> None:
        image = self.get_input_value("image")
        if image is None:
            return

        threshold = self.get_parameter("threshold")
        channel = self.get_parameter("channel")
        invert = self.get_parameter("invert")

        # Get comparison channel
        if channel == "luminance":
            compare = np.mean(image.data, axis=0)
        elif channel == "red":
            compare = image.data[0]
        elif channel == "green":
            compare = image.data[1]
        elif channel == "blue":
            compare = image.data[2]
        else:
            compare = image.data[0]

        # Create mask
        mask = compare > threshold
        if invert:
            mask = ~mask

        # Create outputs
        mask_data = np.stack([mask.astype(np.float32)] * 3)

        result_data = np.where(mask, 1.0, 0.0)
        result_data = np.stack([result_data] * image.channels)

        self.set_output_value("image", ImageBuffer(
            data=result_data.astype(np.float32),
            colorspace=image.colorspace
        ))

        self.set_output_value("mask", ImageBuffer(
            data=mask_data,
            colorspace="RGB"
        ))
```

---

For more examples, explore the existing nodes in `src/artifice/nodes/`.

For API details, see the [API Reference](api-reference.md).
