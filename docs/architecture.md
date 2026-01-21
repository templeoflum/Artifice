# Artifice Engine Architecture

This document describes the system architecture of Artifice Engine, explaining the core concepts, data flow, and design decisions.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Data Flow](#data-flow)
- [Component Architecture](#component-architecture)
- [Node System](#node-system)
- [User Interface](#user-interface)
- [Extension Points](#extension-points)
- [Future Architecture](#future-architecture)

## Overview

Artifice Engine is built on a **node-based processing architecture** where complex image transformations are constructed by connecting simple, focused processing units (nodes) into directed acyclic graphs (DAGs).

```
┌─────────────────────────────────────────────────────────────────┐
│                     Artifice Engine                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   UI Layer  │  │  Core Layer │  │     Processing Layer    │ │
│  │             │  │             │  │                         │ │
│  │ MainWindow  │  │  NodeGraph  │  │  IO Nodes               │ │
│  │ NodeEditor  │◄─┤  Node       │◄─┤  Color Nodes            │ │
│  │ Inspector   │  │  Port       │  │  Segmentation Nodes     │ │
│  │ Preview     │  │  DataTypes  │  │  Prediction Nodes       │ │
│  │ Palette     │  │  Registry   │  │  Transform Nodes        │ │
│  └─────────────┘  └─────────────┘  │  Corruption Nodes       │ │
│                                     └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each node is self-contained with clear inputs and outputs
2. **Data Agnostic**: Core system works with any data type through port typing
3. **Non-Destructive**: Original data is preserved; transformations create new data
4. **Extensible**: New nodes can be added without modifying core code
5. **Testable**: Each component can be tested in isolation

## Core Concepts

### NodeGraph

The `NodeGraph` is the central orchestrator that manages nodes and their connections.

```python
class NodeGraph:
    nodes: list[Node]           # All nodes in the graph
    connections: list[Connection]  # All connections between ports

    def add_node(node: Node) -> None
    def remove_node(node: Node) -> None
    def connect(source_node, source_port, target_node, target_port) -> bool
    def disconnect(connection) -> None
    def execute() -> None       # Process all nodes in topological order
    def save(path) -> None      # Serialize to file
    def load(path) -> NodeGraph # Deserialize from file
```

### Node

A `Node` represents a single processing operation with typed inputs, outputs, and parameters.

```python
class Node:
    id: str                     # Unique identifier
    name: str                   # Display name
    inputs: dict[str, InputPort]
    outputs: dict[str, OutputPort]
    parameters: dict[str, Any]
    position: tuple[float, float]  # UI position

    def process() -> None       # Execute the node's algorithm
    def validate() -> bool      # Check if inputs are valid
```

### Port

Ports are connection points on nodes with type information.

```python
class PortType(Enum):
    IMAGE = "image"             # ImageBuffer data
    REGIONS = "regions"         # Segmentation regions
    FLOAT = "float"             # Single float value
    INT = "int"                 # Single integer
    STRING = "string"           # Text data
    ANY = "any"                 # Accepts any type

class InputPort:
    name: str
    port_type: PortType
    connected_output: OutputPort | None

class OutputPort:
    name: str
    port_type: PortType
    value: Any                  # Cached output value
```

### DataTypes

#### ImageBuffer

The primary data type for image data:

```python
class ImageBuffer:
    data: np.ndarray           # Shape: (channels, height, width), dtype: float32
    colorspace: str            # "RGB", "HSV", "LAB", etc.
    metadata: dict             # Optional metadata

    @property
    def shape(self) -> tuple[int, int, int]
    @property
    def channels(self) -> int
    @property
    def height(self) -> int
    @property
    def width(self) -> int
```

**Important**: Image data is stored in **channel-first format** (C, H, W) with **float32** values in the range [0.0, 1.0] for normalized data.

#### RegionMap

For segmentation results:

```python
class RegionMap:
    labels: np.ndarray         # Integer labels for each pixel
    num_regions: int
    bounds: list[tuple]        # Bounding boxes for each region
```

## Data Flow

### Execution Model

The graph executes nodes in **topological order**, ensuring all inputs are ready before a node processes:

```
1. Build dependency graph from connections
2. Sort nodes topologically
3. For each node in sorted order:
   a. Pull values from connected input ports
   b. Call node.process()
   c. Output values are cached in output ports
4. Return final outputs
```

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Loader  │────►│ ColorSp. │────►│ Segment  │────►│ Predict  │
│          │     │          │     │          │     │          │
│ [image]──┼────►│──[image] │────►│──[image] │     │──[image] │
└──────────┘     └──────────┘     │          │     │          │
                                  │ [regions]┼────►│──[regions│
                                  └──────────┘     └──────────┘
```

### Connection Rules

- Each input port can have **at most one** connection
- Each output port can connect to **multiple** inputs
- Connections must match port types (or use `ANY` type)
- Cycles are not allowed (DAG only)

## Component Architecture

### Core Layer (`artifice/core/`)

```
core/
├── __init__.py
├── node.py          # Base Node class
├── port.py          # Port and PortType definitions
├── graph.py         # NodeGraph orchestrator
├── data_types.py    # ImageBuffer, RegionMap, etc.
└── registry.py      # Node type registry
```

### Node Layer (`artifice/nodes/`)

```
nodes/
├── io/              # Input/Output
│   ├── loader.py    # ImageLoaderNode
│   └── saver.py     # ImageSaverNode
├── color/           # Color Processing
│   ├── colorspace.py
│   ├── conversions.py
│   └── channel_ops.py
├── segmentation/    # Image Segmentation
│   └── quadtree.py
├── prediction/      # Prediction Algorithms
│   ├── predictors.py
│   └── predict_node.py
├── quantization/    # Quantization
│   └── quantize_node.py
├── transform/       # Transforms
│   ├── dct.py
│   ├── fft.py
│   ├── wavelet.py
│   └── pixelsort.py
├── corruption/      # Data Corruption
│   ├── bit_ops.py
│   └── data_ops.py
└── utility/         # Utilities
    └── passthrough.py
```

### UI Layer (`artifice/ui/`)

```
ui/
├── __init__.py
├── main_window.py   # Application window
├── node_editor.py   # Graph canvas
├── node_widget.py   # Visual node representation
├── connection.py    # Connection lines
├── inspector.py     # Parameter editing
├── preview.py       # Image preview
├── palette.py       # Node selection
└── undo.py          # Undo/redo system
```

## Node System

### Node Lifecycle

```
1. Construction: __init__() sets up ports and parameters
2. Connection: Inputs connected to other outputs
3. Validation: validate() checks if processing can proceed
4. Processing: process() executes the algorithm
5. Output: Results cached in output ports
```

### Parameter System

Nodes declare parameters with metadata for UI generation:

```python
self.add_parameter(
    name="threshold",
    default=0.5,
    min_value=0.0,
    max_value=1.0,
    description="Processing threshold"
)

self.add_parameter(
    name="mode",
    default="normal",
    choices=["normal", "intense", "subtle"]
)
```

### Node Registration

Nodes register themselves for discovery:

```python
from artifice.core.registry import register_node

@register_node
class MyNode(Node):
    CATEGORY = "transform"
    ...
```

Or manual registration:

```python
from artifice.core.registry import NodeRegistry

NodeRegistry.register("MyNode", MyNode, category="transform")
```

## User Interface

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  File  Edit  View  Graph  Help                                  │
├─────────┬───────────────────────────────────────┬───────────────┤
│         │                                       │               │
│  Node   │         Node Editor Canvas            │   Preview     │
│ Palette │                                       │               │
│         │    ┌─────┐      ┌─────┐              │               │
│ ─────── │    │Node1│──────│Node2│              ├───────────────┤
│ IO      │    └─────┘      └─────┘              │               │
│ Color   │         │                             │  Inspector    │
│ Segment │         ▼                             │               │
│ ...     │    ┌─────┐                           │  [Parameters] │
│         │    │Node3│                           │               │
│         │    └─────┘                           │               │
└─────────┴───────────────────────────────────────┴───────────────┘
```

### Undo/Redo System

The undo system uses the **Command Pattern**:

```python
class Command:
    def execute(self) -> None
    def undo(self) -> None

class AddNodeCommand(Command): ...
class RemoveNodeCommand(Command): ...
class ConnectCommand(Command): ...
class DisconnectCommand(Command): ...
class ChangeParameterCommand(Command): ...
class CompositeCommand(Command): ...  # Groups multiple commands
```

## Extension Points

### Custom Nodes

The primary extension point. Create new processing algorithms by subclassing `Node`:

```python
from artifice.core.node import Node
from artifice.core.port import PortType

class MyGlitchNode(Node):
    CATEGORY = "corruption"

    def __init__(self):
        super().__init__("MyGlitchNode")
        self.add_input("image", PortType.IMAGE)
        self.add_output("image", PortType.IMAGE)
        self.add_parameter("intensity", 0.5)

    def process(self):
        image = self.get_input_value("image")
        # Apply glitch effect
        result = self._glitch(image, self.get_parameter("intensity"))
        self.set_output_value("image", result)
```

### Custom Data Types

Add new data types for specialized processing:

```python
class AudioBuffer:
    data: np.ndarray
    sample_rate: int
    channels: int

# Register new port type
class PortType(Enum):
    AUDIO = "audio"
```

### Custom Color Spaces

Add new color space conversions:

```python
from artifice.nodes.color.conversions import register_colorspace

@register_colorspace("CUSTOM")
def rgb_to_custom(rgb: np.ndarray) -> np.ndarray:
    # Conversion logic
    return custom_data

@register_colorspace("CUSTOM", inverse=True)
def custom_to_rgb(custom: np.ndarray) -> np.ndarray:
    # Inverse conversion
    return rgb_data
```

## Future Architecture

### GPU Acceleration (Planned)

```
┌─────────────────┐
│   Node Graph    │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Executor │
    └────┬────┘
         │
   ┌─────┴─────┐
   │           │
┌──▼──┐    ┌──▼──┐
│ CPU │    │ GPU │
│Nodes│    │Nodes│
└─────┘    └─────┘
```

Nodes will declare GPU compatibility:

```python
class GPUNode(Node):
    GPU_CAPABLE = True

    def process_gpu(self, context: GPUContext) -> None:
        # CUDA/OpenCL implementation
        pass
```

### Video Processing (Planned)

```python
class VideoBuffer:
    frames: list[ImageBuffer]
    fps: float
    duration: float

class TemporalNode(Node):
    def process_frame(self, frame: ImageBuffer, time: float) -> ImageBuffer:
        pass
```

### AI Integration (Planned)

```python
class AINode(Node):
    MODEL_PATH: str

    def load_model(self) -> None:
        self._model = load_pytorch_model(self.MODEL_PATH)

    def process(self) -> None:
        with torch.no_grad():
            result = self._model(self.get_input_value("image"))
        self.set_output_value("image", result)
```

## Performance Considerations

### Current Optimizations

- **NumPy Vectorization**: All operations use vectorized NumPy operations
- **Lazy Evaluation**: Nodes only process when outputs are requested
- **Output Caching**: Results cached until inputs change
- **Topological Execution**: Minimal reprocessing through dependency tracking

### Future Optimizations

- **GPU Offloading**: Move computationally intensive operations to GPU
- **Parallel Node Execution**: Process independent branches concurrently
- **Memory Pooling**: Reuse image buffers to reduce allocation overhead
- **Incremental Updates**: Only reprocess affected portions of the graph

---

For implementation details, see the [API Reference](api-reference.md).

For creating custom nodes, see the [Node Development Guide](node-development.md).
