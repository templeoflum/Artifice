# Artifice

**Motto:** *Converse with Chaos, Sculpt Emergence.*

A next-generation glitch art co-creative environment built on a node-based architecture for emergent complexity, semantic awareness, and generative processes.

## Project Overview

Artifice evolves beyond traditional glitch tools (like GLIC) into a unified platform where codec design, glitch art, generative art, and AI-assisted creation converge. The system enables users to cultivate and guide digital ecologies that produce aesthetically rich, often unpredictable visual and auditory experiences.

## Current Implementation Status

### What's Implemented (v0.1.0)
- ✅ Core node system with typed ports and parameters
- ✅ Node graph with topological execution and caching
- ✅ PySide6/Qt UI with node editor, inspector, preview panel
- ✅ Drag-and-drop node creation from palette
- ✅ Visual connection drawing with bezier curves
- ✅ Context menus for disconnect/delete operations
- ✅ File browser dialogs for image loading/saving
- ✅ Real-time preview of processed images
- ✅ Multiple node families: I/O, Color, Segmentation, Prediction, Quantization, Transform, Corruption

### Planned Features
- ⏳ Video/temporal processing
- ⏳ AI integration (segmentation, style transfer)
- ⏳ Audio reactivity
- ⏳ GPU acceleration
- ⏳ Plugin ecosystem

## Core Architecture

### Technology Stack
- **Language:** Python 3.10+
- **UI Framework:** PySide6 (Qt for Python)
- **Image Processing:** NumPy, OpenCV, Pillow, SciPy
- **Transforms:** PyWavelets (wavelet transforms)
- **Data Format:** NumPy float32 arrays (H, W, C) normalized 0.0-1.0

### Key Design Principles
1. **Node-Based System:** All processing stages are nodes with connectable inputs/outputs
2. **Typed Ports:** Ports have explicit types (IMAGE, ARRAY, FLOAT, INT, BOOL, STRING, REGIONS)
3. **Lazy Evaluation:** Nodes only execute when dirty and needed
4. **Extensibility:** Easy to add new nodes by subclassing Node base class

## Implemented Node Families

### 1. Input/Output Nodes (`nodes/io/`)
- **ImageLoaderNode** - Load images with file browser (PNG, JPG, TIFF, WebP, BMP, GIF). Defaults to user's Pictures directory.
- **ImageSaverNode** - Save processed images with format selection. Defaults to project `output/` folder.

### 2. Color Processing (`nodes/color/`)
- **ColorSpaceNode** - Convert between RGB, HSV, LAB, XYZ, YCbCr, LUV, YIQ
- **ChannelSplitNode** - Separate image into individual channels
- **ChannelMergeNode** - Combine channels back into image
- **ChannelSwapNode** - Reorder/swap color channels

### 3. Segmentation (`nodes/segmentation/`)
- **QuadtreeSegmentNode** - Adaptive image segmentation with multiple criteria:
  - Variance, edge detection, gradient magnitude
  - Configurable threshold, max depth, min block size

### 4. Prediction (`nodes/prediction/`)
- **PredictNode** - GLIC-style predictors for residual generation:
  - Horizontal, Vertical, DC (mean), Paeth, Average, Gradient
  - Works with quadtree regions for adaptive prediction

### 5. Quantization (`nodes/quantization/`)
- **QuantizeNode** - Reduce precision with multiple modes:
  - Uniform, adaptive, per-channel quantization
  - Configurable levels (2-256)

### 6. Transform (`nodes/transform/`)
- **DCTNode** - Discrete Cosine Transform (block-based or full image)
- **FFTNode** - Fast Fourier Transform with frequency manipulation
- **WaveletNode** - Multi-level wavelet decomposition (Haar, Daubechies, etc.)
- **PixelSortNode** - Glitch-style pixel sorting with multiple modes:
  - Sort by brightness, hue, saturation, red/green/blue
  - Horizontal, vertical, or diagonal sorting
  - Threshold-based region selection

### 7. Data Corruption (`nodes/corruption/`)
- **BitShiftNode** / **BitFlipNode** - Bit-level manipulation
- **ByteSwapNode** / **ByteShiftNode** - Byte-level corruption
- **DataRepeaterNode** / **DataDropperNode** - Structural data manipulation

### 8. Utility (`nodes/utility/`)
- **PassThroughNode** - Pass data unchanged (debugging)

### 9. Pipeline (`nodes/pipeline/`)
- **GLICPipelineNode** - Combined GLIC processing in single node

## Planned Node Families

### Temporal Processing (Video)
- FrameBlend/Difference, FrameBuffer, FrameScrambler
- TimeStretcher/Squeezer, OpticalFlowToolkit

### AI Integration
- ObjectSegmenter, SemanticSegmenter
- StyleTransfer, EffectEmulator

### Audio Reactivity
- AudioDrive, ParameterModulator

## Development Guidelines

### Code Organization
```
Artifice/
├── output/                # Default output directory for saved images
├── src/artifice/
│   ├── core/              # Node system, graph, ports, registry
│   │   ├── node.py        # Base Node class, Parameter, ParameterType
│   │   ├── port.py        # InputPort, OutputPort, PortType
│   │   ├── graph.py       # NodeGraph, Connection, execution engine
│   │   ├── registry.py    # Node auto-registration system
│   │   └── data_types.py  # Shared data types
│   ├── nodes/             # Node implementations by family
│   │   ├── io/            # ImageLoaderNode, ImageSaverNode
│   │   ├── color/         # ColorSpaceNode, Channel operations
│   │   ├── segmentation/  # QuadtreeSegmentNode
│   │   ├── prediction/    # PredictNode, predictors
│   │   ├── quantization/  # QuantizeNode
│   │   ├── transform/     # DCT, FFT, Wavelet, PixelSort
│   │   ├── corruption/    # Bit/byte manipulation nodes
│   │   ├── pipeline/      # GLICPipelineNode
│   │   └── utility/       # PassThroughNode
│   └── ui/                # Qt-based user interface
│       ├── main_window.py # Main application window
│       ├── node_editor.py # Node graph canvas (QGraphicsView)
│       ├── node_widget.py # Visual node representation
│       ├── connection.py  # Connection lines (bezier curves)
│       ├── palette.py     # Draggable node palette
│       ├── inspector.py   # Parameter editing panel
│       └── preview.py     # Image preview panel
├── tests/                 # Test suite
└── docs/                  # Documentation
```

### Node Implementation Pattern
To create a new node:
```python
from artifice.core.node import Node, ParameterType
from artifice.core.port import PortType

class MyNode(Node):
    name = "My Node"           # Display name
    category = "Transform"     # Category for palette
    description = "Does something cool"
    _abstract = False          # Must be False to register

    def define_ports(self) -> None:
        self.add_input("image", PortType.IMAGE, "Input image")
        self.add_output("image", PortType.IMAGE, "Output image")

    def define_parameters(self) -> None:
        self.add_parameter(
            "strength",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            description="Effect strength"
        )

    def process(self) -> None:
        image = self.get_input_value("image")
        strength = self.get_parameter("strength")
        # Process the image...
        result = image * strength
        self.set_output_value("image", result)
```

### Port Types
- `PortType.IMAGE` - NumPy array (H, W, C) float32 0.0-1.0
- `PortType.ARRAY` - Generic NumPy array
- `PortType.FLOAT` - Single float value
- `PortType.INT` - Single integer value
- `PortType.BOOL` - Boolean value
- `PortType.STRING` - Text string
- `PortType.REGIONS` - QuadTree regions for segmentation

### Parameter Types
- `ParameterType.FLOAT` - Slider for float values
- `ParameterType.INT` - Slider for integer values
- `ParameterType.BOOL` - Checkbox
- `ParameterType.STRING` - Text field
- `ParameterType.ENUM` - Dropdown with choices
- `ParameterType.FILEPATH` - File browser button (supports `file_filter`, `is_save_path`, `default_directory`)
- `ParameterType.COLOR` - Color picker (planned)
- `ParameterType.CURVE` - Curve editor (planned)

### FILEPATH Parameter Options
When using `ParameterType.FILEPATH`, these additional options are available:
- `file_filter` - Qt file filter string (e.g., `"Images (*.png *.jpg);;All Files (*)"`)
- `is_save_path` - If `True`, shows save dialog; if `False`, shows open dialog
- `default_directory` - Default directory when file browser opens (if no path is set)

## UI/UX Features

### Node Editor
- **Drag-and-drop** node creation from palette
- **Zoomable/pannable** canvas with mouse wheel and middle-click
- **Visual connections** with bezier curves colored by port type
- **Grid snapping** for node alignment
- **Multi-select** nodes with Shift+click or box selection
- **Delete key** removes selected nodes and connections

### Context Menus (Right-click)
- On **empty space**: Create new nodes
- On **node**: Delete, duplicate node
- On **connection**: Delete connection
- On **port**: Disconnect all connections

### Connection Workflow
- Click output port, then click input port to connect
- Click connected input port to disconnect and start new connection
- Compatible ports highlight green when dragging
- Type checking prevents invalid connections

### Inspector Panel
- Auto-populates with selected node's parameters
- Float/Int parameters: sliders with min/max
- Bool parameters: checkboxes
- Enum parameters: dropdown menus
- Filepath parameters: text field with Browse button

### Preview Panel
- Real-time display of last processed image
- Updates after each graph execution

## Running the Application

```bash
# Launch the GUI
python -m artifice

# Or using the entry point
artifice
```

## File Formats

- `.json` - Node graph save/load (implemented)
- `.artifice` - Project files (planned)
- `.anode` - Individual node presets (planned)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=artifice

# Run specific test
pytest tests/test_graph.py -v
```
