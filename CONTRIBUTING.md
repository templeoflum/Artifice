# Contributing to Artifice

Thank you for your interest in contributing to Artifice! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Creating Nodes](#creating-nodes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We're building a creative tool for artists and developers - let's maintain a welcoming environment for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes
5. Make your changes with tests
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- A virtual environment manager (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Artifice.git
cd Artifice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or install dependencies manually
pip install -r requirements.txt
pip install pytest pytest-cov

# Verify installation
pytest
```

### IDE Setup

We recommend using an IDE with Python support:

- **VS Code**: Install the Python extension
- **PyCharm**: Professional or Community edition
- **Vim/Neovim**: With LSP support (pyright or pylsp)

## Code Style

### General Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public classes and methods
- Keep functions focused and reasonably sized
- Prefer clarity over cleverness

### Formatting

We use the following tools (run before committing):

```bash
# Format code with black (if installed)
black src/ tests/

# Sort imports with isort (if installed)
isort src/ tests/

# Type checking with mypy (if installed)
mypy src/
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NodeGraph`, `ImageBuffer`)
- **Functions/Methods**: `snake_case` (e.g., `add_node`, `get_connections`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_DEPTH`, `DEFAULT_THRESHOLD`)
- **Private members**: `_leading_underscore` (e.g., `_internal_state`)
- **Node classes**: End with `Node` (e.g., `PixelSortNode`, `DCTNode`)

### Documentation Style

Use Google-style docstrings:

```python
def process_image(image: ImageBuffer, threshold: float = 0.5) -> ImageBuffer:
    """Process an image with the given threshold.

    Args:
        image: The input image buffer to process.
        threshold: Processing threshold value (0.0 to 1.0).

    Returns:
        The processed image buffer.

    Raises:
        ValueError: If threshold is out of range.
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_graph.py

# Run specific test class
pytest tests/test_graph.py::TestNodeGraph

# Run with coverage report
pytest --cov=artifice --cov-report=html

# Run only fast tests (skip slow/integration tests)
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for high coverage of new code

Example test structure:

```python
"""Tests for the FooNode."""

import pytest
import numpy as np

from artifice.nodes.category.foo import FooNode
from artifice.core.data_types import ImageBuffer


class TestFooNode:
    """Tests for FooNode functionality."""

    @pytest.fixture
    def node(self):
        """Create a FooNode instance."""
        return FooNode()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data=data, colorspace="RGB")

    def test_default_parameters(self, node):
        """Test that default parameters are set correctly."""
        assert node.get_parameter("threshold") == 0.5

    def test_process_basic(self, node, sample_image):
        """Test basic processing works."""
        node.inputs["image"].set_value(sample_image)
        node.process()

        result = node.outputs["image"].get_value()
        assert result is not None
        assert result.shape == sample_image.shape
```

## Submitting Changes

### Pull Request Process

1. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b feature/add-blur-node
   git checkout -b fix/colorspace-conversion-bug
   ```

2. **Make your changes** with clear, atomic commits:
   ```bash
   git commit -m "Add GaussianBlurNode with configurable radius"
   git commit -m "Add tests for GaussianBlurNode"
   ```

3. **Ensure tests pass**:
   ```bash
   pytest
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/add-blur-node
   ```

5. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what and why
   - Link to any related issues
   - Screenshots for UI changes

### PR Review Checklist

Before submitting, ensure:

- [ ] All tests pass
- [ ] New code has tests
- [ ] Code follows style guidelines
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear
- [ ] No unrelated changes included

## Creating Nodes

One of the best ways to contribute is by creating new processing nodes. See the [Node Development Guide](docs/node-development.md) for detailed instructions.

### Quick Node Template

```python
"""My custom node implementation."""

from __future__ import annotations

import numpy as np

from artifice.core.node import Node
from artifice.core.port import PortType
from artifice.core.data_types import ImageBuffer


class MyCustomNode(Node):
    """
    Brief description of what this node does.

    Longer description with details about the algorithm,
    use cases, and any important notes.
    """

    # Node metadata
    CATEGORY = "transform"  # or: io, color, segmentation, prediction, etc.
    DESCRIPTION = "One-line description for the palette"

    def __init__(self):
        super().__init__("MyCustomNode")

        # Define inputs
        self.add_input("image", PortType.IMAGE)

        # Define outputs
        self.add_output("image", PortType.IMAGE)

        # Define parameters with defaults
        self.add_parameter("strength", 0.5, min_value=0.0, max_value=1.0)
        self.add_parameter("mode", "normal", choices=["normal", "intense"])

    def process(self) -> None:
        """Process the input and produce output."""
        # Get input
        image = self.get_input_value("image")
        if image is None:
            return

        # Get parameters
        strength = self.get_parameter("strength")
        mode = self.get_parameter("mode")

        # Process
        result_data = self._apply_effect(image.data, strength, mode)

        # Set output
        result = ImageBuffer(data=result_data, colorspace=image.colorspace)
        self.set_output_value("image", result)

    def _apply_effect(self, data: np.ndarray, strength: float, mode: str) -> np.ndarray:
        """Apply the effect to the data."""
        # Your algorithm here
        return data * strength
```

### Node Categories

Place your node in the appropriate category:

- `nodes/io/` - File loading/saving, streams
- `nodes/color/` - Color space, channel operations
- `nodes/segmentation/` - Image segmentation algorithms
- `nodes/prediction/` - Prediction algorithms
- `nodes/quantization/` - Quantization methods
- `nodes/transform/` - Frequency transforms, spatial transforms
- `nodes/corruption/` - Data manipulation, glitch effects
- `nodes/utility/` - Helper nodes, debugging tools

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal steps to trigger the bug
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, relevant package versions
6. **Screenshots/Logs**: If applicable

### Feature Requests

For feature requests, please describe:

1. **Use Case**: What you're trying to accomplish
2. **Proposed Solution**: Your idea for the feature
3. **Alternatives**: Other approaches you've considered
4. **Context**: How this fits into your workflow

## Questions?

- Open a [GitHub Discussion](https://github.com/templeoflum/Artifice/discussions) for general questions
- Check existing [Issues](https://github.com/templeoflum/Artifice/issues) for known problems
- Review the [Documentation](docs/) for guides and references

---

Thank you for contributing to Artifice!

*Converse with Chaos, Sculpt Emergence.*
