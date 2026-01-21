# Artifice

**Converse with Chaos, Sculpt Emergence.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-277%20passing-brightgreen.svg)]()

A next-generation glitch art co-creative environment built on a node-based architecture for emergent complexity, semantic awareness, and generative processes.

![Artifice Screenshot](docs/images/screenshot.png)

## Overview

Artifice evolves beyond traditional glitch tools into a unified platform where codec design, glitch art, generative art, and AI-assisted creation converge. The system enables users to cultivate and guide digital ecologies that produce aesthetically rich, often unpredictable visual and auditory experiences.

### Key Features

- **Node-Based Visual Programming** - Intuitive drag-and-drop interface for building complex image processing pipelines
- **GLIC-Inspired Processing** - Advanced prediction, segmentation, and quantization algorithms derived from cutting-edge glitch research
- **Real-Time Preview** - See your glitch effects as you build them
- **Modular Algorithm Chaining** - Connect different algorithms in unique ways (e.g., bit corruption → pixel sorting)
- **Extensible Architecture** - Create custom nodes with Python
- **Multiple Color Spaces** - Work in RGB, HSV, LAB, XYZ, YCbCr, and more
- **Comprehensive Transform Library** - DCT, FFT, Wavelets, Pixel Sorting, and data corruption tools
- **File Browser Integration** - Easy file selection with native dialogs
- **Context Menus** - Right-click for quick access to common actions

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/templeoflum/Artifice.git
cd Artifice

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

### Using the UI

1. **Add nodes** - Drag nodes from the palette on the left, or right-click on the canvas
2. **Connect nodes** - Click an output port (right side), then click an input port (left side)
3. **Configure parameters** - Select a node to edit its parameters in the inspector panel
4. **Load an image** - Select the Image Loader node and click "Browse..." to select a file
5. **Execute** - Click the "Execute" button in the toolbar to process the graph
6. **Save output** - Configure the Image Saver node with an output path

### Example Pipeline: Glitchy Pixel Sort

1. Add **Image Loader** → **Bit Flip** → **Pixel Sort** → **Image Saver**
2. Connect them in sequence (output → input)
3. Configure Bit Flip: set probability to 0.001
4. Configure Pixel Sort: choose "brightness" mode, set threshold
5. Execute and view the result in the preview panel

### Programmatic Usage

```python
from artifice.core.graph import NodeGraph
from artifice.nodes.io.loader import ImageLoaderNode
from artifice.nodes.io.saver import ImageSaverNode
from artifice.nodes.transform.pixelsort import PixelSortNode
from artifice.nodes.corruption.bit_ops import BitFlipNode

# Create a processing graph
graph = NodeGraph()

# Add nodes
loader = ImageLoaderNode()
loader.set_parameter("path", "input.png")

bitflip = BitFlipNode()
bitflip.set_parameter("probability", 0.001)

pixelsort = PixelSortNode()
pixelsort.set_parameter("sort_by", "brightness")
pixelsort.set_parameter("direction", "horizontal")

saver = ImageSaverNode()
saver.set_parameter("path", "output.png")

# Add to graph
for node in [loader, bitflip, pixelsort, saver]:
    graph.add_node(node)

# Connect the pipeline
graph.connect(loader, "image", bitflip, "image")
graph.connect(bitflip, "image", pixelsort, "image")
graph.connect(pixelsort, "image", saver, "image")

# Execute
graph.execute()
```

## Node Families

Artifice provides a rich library of processing nodes organized by function:

### Input/Output
- **Image Loader** - Load images with file browser (PNG, JPG, TIFF, WebP, BMP, GIF)
- **Image Saver** - Save processed images with format selection

### Color Processing
- **Color Space** - Convert between RGB, HSV, LAB, XYZ, YCbCr, LUV, YIQ
- **Channel Split** / **Channel Merge** - Separate and combine color channels
- **Channel Swap** - Reorder color channels

### Segmentation
- **Quadtree Segment** - Adaptive image segmentation with variance/edge/gradient criteria

### Prediction
- **Predict** - GLIC-style predictors (Horizontal, Vertical, DC, Paeth, Average, Gradient)

### Quantization
- **Quantize** - Reduce precision with uniform, adaptive, or per-channel modes

### Transforms
- **DCT** - Discrete Cosine Transform (block-based or full image)
- **FFT** - Fast Fourier Transform with frequency manipulation
- **Wavelet Transform** - Multi-level wavelet decomposition (Haar, Daubechies, Symlets, etc.)
- **Pixel Sort** - Glitch-style pixel sorting by brightness, hue, saturation, or color channel

### Data Corruption
- **Bit Shift** / **Bit Flip** - Bit-level manipulation with configurable probability
- **Byte Swap** - Byte-level channel swapping
- **XOR Noise** - XOR-based noise injection
- **Data Repeat** / **Data Drop** - Structural data manipulation
- **Data Weave** / **Data Scramble** - Row/column interleaving and scrambling

### Utility
- **Pass Through** - Pass data unchanged (useful for debugging)

## Documentation

See [CLAUDE.md](CLAUDE.md) for:
- Architecture overview
- Node development guide
- Port and parameter types
- UI/UX features

## Project Structure

```
Artifice/
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

Artifice is under active development. Current implementation status:

- [x] **Phase 1**: Core node system and data flow
- [x] **Phase 2**: GLIC-style processing nodes
- [x] **Phase 3**: Transform and corruption nodes
- [x] **Phase 4**: Qt-based user interface with drag-and-drop
- [x] **Phase 4.5**: UX improvements (context menus, file browsers, connection management)
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

- **Repository**: [github.com/templeoflum/Artifice](https://github.com/templeoflum/Artifice)
- **Issues**: [GitHub Issues](https://github.com/templeoflum/Artifice/issues)

---

*Converse with Chaos, Sculpt Emergence.*
