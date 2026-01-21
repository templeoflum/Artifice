# Artifice Engine

**Converse with Chaos, Sculpt Emergence.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-277%20passing-brightgreen.svg)]()

A next-generation glitch art co-creative environment built on a node-based architecture for emergent complexity, semantic awareness, and generative processes.

![Artifice Engine Screenshot](docs/images/screenshot-placeholder.png)

## Overview

Artifice Engine evolves beyond traditional glitch tools into a unified platform where codec design, glitch art, generative art, and AI-assisted creation converge. The system enables users to cultivate and guide digital ecologies that produce aesthetically rich, often unpredictable visual and auditory experiences.

### Key Features

- **Node-Based Visual Programming** - Intuitive drag-and-drop interface for building complex image processing pipelines
- **GLIC-Inspired Processing** - Advanced prediction, segmentation, and quantization algorithms derived from cutting-edge glitch research
- **Real-Time Preview** - See your glitch effects as you build them
- **Extensible Architecture** - Create custom nodes with Python
- **Multiple Color Spaces** - Work in RGB, HSV, LAB, XYZ, YCbCr, and more
- **Comprehensive Transform Library** - DCT, FFT, Wavelets, Pixel Sorting, and data corruption tools
- **Undo/Redo Support** - Full history management for non-destructive experimentation

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/templeoflum/Artifice-Engine.git
cd Artifice-Engine

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- **NumPy** - Array operations and numerical computing
- **Pillow** - Image loading and saving
- **SciPy** - Scientific computing (wavelets, signal processing)
- **PySide6** - Qt-based user interface
- **PyWavelets** - Wavelet transforms

## Quick Start

### Launch the Application

```bash
python -m artifice
```

### Programmatic Usage

```python
from artifice.core.graph import NodeGraph
from artifice.nodes.io.loader import ImageLoaderNode
from artifice.nodes.io.saver import ImageSaverNode
from artifice.nodes.color.colorspace import ColorSpaceNode
from artifice.nodes.segmentation.quadtree import QuadtreeSegmentNode
from artifice.nodes.prediction.predict_node import PredictNode
from artifice.nodes.quantization.quantize_node import QuantizeNode

# Create a processing graph
graph = NodeGraph()

# Add nodes
loader = ImageLoaderNode()
loader.set_parameter("file_path", "input.png")

colorspace = ColorSpaceNode()
colorspace.set_parameter("target_space", "YCbCr")

segment = QuadtreeSegmentNode()
segment.set_parameter("threshold", 15.0)
segment.set_parameter("max_depth", 6)

predict = PredictNode()
predict.set_parameter("predictor_type", "paeth")

quantize = QuantizeNode()
quantize.set_parameter("levels", 8)

saver = ImageSaverNode()
saver.set_parameter("file_path", "output.png")

# Add to graph
for node in [loader, colorspace, segment, predict, quantize, saver]:
    graph.add_node(node)

# Connect the pipeline
graph.connect(loader, "image", colorspace, "image")
graph.connect(colorspace, "image", segment, "image")
graph.connect(segment, "regions", predict, "regions")
graph.connect(colorspace, "image", predict, "image")
graph.connect(predict, "residual", quantize, "image")
graph.connect(quantize, "image", saver, "image")

# Execute
graph.execute()
```

## Node Families

Artifice Engine provides a rich library of processing nodes organized by function:

### Input/Output
- **ImageLoaderNode** - Load images (PNG, JPG, TIFF, WebP, BMP, EXR)
- **ImageSaverNode** - Save processed images

### Color Processing
- **ColorSpaceNode** - Convert between color spaces (RGB, HSV, LAB, XYZ, YCbCr, LUV, YIQ)
- **ChannelSplitNode** / **ChannelMergeNode** - Separate and combine color channels
- **ChannelSwapNode** - Reorder color channels

### Segmentation
- **QuadtreeSegmentNode** - Adaptive quadtree image segmentation with multiple criteria

### Prediction
- **PredictNode** - GLIC-style predictors (H, V, DC, Paeth, Average, Gradient)

### Quantization
- **QuantizeNode** - Scalar and adaptive quantization

### Transforms
- **DCTNode** - Discrete Cosine Transform (block-based, full image)
- **FFTNode** - Fast Fourier Transform with frequency manipulation
- **WaveletNode** - Multi-level wavelet decomposition (Haar, Daubechies, etc.)
- **PixelSortNode** - Glitch-style pixel sorting with multiple modes

### Data Corruption
- **BitShiftNode** / **BitFlipNode** - Bit-level manipulation
- **ByteSwapNode** / **ByteShiftNode** - Byte-level corruption
- **DataRepeaterNode** / **DataDropperNode** - Structural data manipulation

### Utility
- **PassThroughNode** - Pass data unchanged (useful for debugging)

## Documentation

- [Getting Started Guide](docs/getting-started.md) - Tutorial for new users
- [Architecture Overview](docs/architecture.md) - System design and concepts
- [Node Development Guide](docs/node-development.md) - Create your own nodes
- [API Reference](docs/api-reference.md) - Complete API documentation

## Project Structure

```
Artifice-Engine/
├── src/artifice/
│   ├── core/           # Node system, graph, data types
│   ├── nodes/          # Node implementations
│   │   ├── io/         # Input/output nodes
│   │   ├── color/      # Color processing
│   │   ├── segmentation/
│   │   ├── prediction/
│   │   ├── quantization/
│   │   ├── transform/  # DCT, FFT, wavelets, pixel sort
│   │   ├── corruption/ # Bit/byte manipulation
│   │   └── utility/
│   └── ui/             # Qt-based user interface
├── tests/              # Comprehensive test suite
├── docs/               # Documentation
└── examples/           # Example projects and scripts
```

## Development Status

Artifice Engine is under active development. Current implementation status:

- [x] **Phase 1**: Core node system and data flow
- [x] **Phase 2**: GLIC-style processing nodes
- [x] **Phase 3**: Transform and corruption nodes
- [x] **Phase 4**: Qt-based user interface
- [ ] **Phase 5**: Video/temporal processing
- [ ] **Phase 6**: AI integration
- [ ] **Phase 7**: Audio reactivity

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=artifice

# Run specific test file
pytest tests/test_graph.py -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [GLIC](https://github.com/example/glic) and glitch art research
- Built with [PySide6](https://www.qt.io/qt-for-python) (Qt for Python)
- Node editor concepts influenced by Blender, Nuke, and TouchDesigner

## Contact

- **Repository**: [github.com/templeoflum/Artifice-Engine](https://github.com/templeoflum/Artifice-Engine)
- **Issues**: [GitHub Issues](https://github.com/templeoflum/Artifice-Engine/issues)

---

*Converse with Chaos, Sculpt Emergence.*
