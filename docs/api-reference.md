# API Reference

Complete API documentation for Artifice.

## Table of Contents

- [Core Module](#core-module)
  - [NodeGraph](#nodegraph)
  - [Node](#node)
  - [Port](#port)
  - [DataTypes](#datatypes)
  - [Registry](#registry)
- [Node Reference](#node-reference)
  - [IO Nodes](#io-nodes)
  - [Color Nodes](#color-nodes)
  - [Segmentation Nodes](#segmentation-nodes)
  - [Prediction Nodes](#prediction-nodes)
  - [Quantization Nodes](#quantization-nodes)
  - [Transform Nodes](#transform-nodes)
  - [Corruption Nodes](#corruption-nodes)
  - [Utility Nodes](#utility-nodes)

---

## Core Module

### NodeGraph

`artifice.core.graph.NodeGraph`

The main container for nodes and their connections.

#### Constructor

```python
NodeGraph()
```

Creates an empty node graph.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodes` | `list[Node]` | All nodes in the graph |
| `connections` | `list[Connection]` | All connections between ports |

#### Methods

##### add_node

```python
def add_node(self, node: Node) -> None
```

Add a node to the graph.

**Parameters:**
- `node`: The node instance to add

**Example:**
```python
graph = NodeGraph()
loader = ImageLoaderNode()
graph.add_node(loader)
```

##### remove_node

```python
def remove_node(self, node: Node) -> None
```

Remove a node and all its connections from the graph.

**Parameters:**
- `node`: The node to remove

##### get_node

```python
def get_node(self, node_id: str) -> Node | None
```

Find a node by its ID.

**Parameters:**
- `node_id`: The unique identifier of the node

**Returns:** The node if found, None otherwise

##### connect

```python
def connect(
    self,
    source_node: Node,
    source_port: str,
    target_node: Node,
    target_port: str
) -> bool
```

Create a connection between two ports.

**Parameters:**
- `source_node`: Node with the output port
- `source_port`: Name of the output port
- `target_node`: Node with the input port
- `target_port`: Name of the input port

**Returns:** True if connection was successful

**Example:**
```python
graph.connect(loader, "image", colorspace, "image")
```

##### disconnect

```python
def disconnect(
    self,
    source_node: Node,
    source_port: str,
    target_node: Node,
    target_port: str
) -> bool
```

Remove a connection between two ports.

##### get_connections

```python
def get_connections(self) -> list[Connection]
```

Get all connections in the graph.

##### execute

```python
def execute(self) -> None
```

Execute all nodes in topological order.

##### clear

```python
def clear(self) -> None
```

Remove all nodes and connections.

##### save

```python
def save(self, path: Path | str) -> None
```

Save the graph to a file.

**Parameters:**
- `path`: File path (typically .artifice extension)

##### load (class method)

```python
@classmethod
def load(cls, path: Path | str) -> NodeGraph
```

Load a graph from a file.

**Parameters:**
- `path`: File path to load

**Returns:** New NodeGraph instance

---

### Node

`artifice.core.node.Node`

Base class for all processing nodes.

#### Constructor

```python
Node(name: str)
```

**Parameters:**
- `name`: Display name for the node

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique identifier (auto-generated) |
| `name` | `str` | Display name |
| `inputs` | `dict[str, InputPort]` | Input ports |
| `outputs` | `dict[str, OutputPort]` | Output ports |
| `parameters` | `dict[str, Any]` | Parameter values |
| `position` | `tuple[float, float]` | UI position (x, y) |

#### Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `CATEGORY` | `str` | Node category for UI grouping |
| `DESCRIPTION` | `str` | Brief description for UI |

#### Methods

##### add_input

```python
def add_input(self, name: str, port_type: PortType) -> InputPort
```

Add an input port to the node.

**Parameters:**
- `name`: Port name (must be unique within inputs)
- `port_type`: The type of data this port accepts

**Returns:** The created InputPort

##### add_output

```python
def add_output(self, name: str, port_type: PortType) -> OutputPort
```

Add an output port to the node.

##### add_parameter

```python
def add_parameter(
    self,
    name: str,
    default: Any,
    min_value: float | None = None,
    max_value: float | None = None,
    choices: list | None = None,
    description: str = ""
) -> None
```

Add a configurable parameter.

**Parameters:**
- `name`: Parameter name
- `default`: Default value
- `min_value`: Minimum value (for numeric parameters)
- `max_value`: Maximum value (for numeric parameters)
- `choices`: List of allowed values (creates dropdown)
- `description`: Help text for UI

##### get_parameter

```python
def get_parameter(self, name: str) -> Any
```

Get the current value of a parameter.

##### set_parameter

```python
def set_parameter(self, name: str, value: Any) -> None
```

Set a parameter value.

##### get_input_value

```python
def get_input_value(self, name: str) -> Any | None
```

Get the value from a connected input port.

**Returns:** The connected value, or None if not connected

##### set_output_value

```python
def set_output_value(self, name: str, value: Any) -> None
```

Set the value of an output port.

##### process

```python
def process(self) -> None
```

Execute the node's processing logic. **Override this in subclasses.**

##### validate

```python
def validate(self) -> bool
```

Check if the node can process (all required inputs connected).

---

### Port

`artifice.core.port`

#### PortType Enum

```python
class PortType(Enum):
    IMAGE = "image"      # ImageBuffer
    REGIONS = "regions"  # RegionMap
    FLOAT = "float"      # float
    INT = "int"          # int
    STRING = "string"    # str
    ANY = "any"          # Any type
```

#### InputPort

```python
class InputPort:
    name: str
    port_type: PortType
    node: Node
    connected_output: OutputPort | None
```

##### Methods

- `get_value() -> Any | None`: Get the connected value
- `set_value(value: Any) -> None`: Set value directly (for testing)
- `is_connected() -> bool`: Check if port has a connection

#### OutputPort

```python
class OutputPort:
    name: str
    port_type: PortType
    node: Node
    value: Any
```

##### Methods

- `get_value() -> Any`: Get the cached output value
- `set_value(value: Any) -> None`: Set the output value

---

### DataTypes

`artifice.core.data_types`

#### ImageBuffer

The primary image data container.

```python
class ImageBuffer:
    data: np.ndarray       # Shape: (C, H, W), dtype: float32
    colorspace: str        # e.g., "RGB", "HSV", "LAB"
    metadata: dict         # Optional metadata
```

##### Constructor

```python
ImageBuffer(
    data: np.ndarray,
    colorspace: str = "RGB",
    metadata: dict | None = None
)
```

**Parameters:**
- `data`: NumPy array with shape (channels, height, width)
- `colorspace`: Color space identifier
- `metadata`: Optional metadata dictionary

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `tuple[int, int, int]` | (channels, height, width) |
| `channels` | `int` | Number of channels |
| `height` | `int` | Image height in pixels |
| `width` | `int` | Image width in pixels |
| `dtype` | `np.dtype` | Data type (float32) |

##### Methods

```python
def copy(self) -> ImageBuffer
```

Create a deep copy of the image buffer.

```python
def to_uint8(self) -> np.ndarray
```

Convert to uint8 format (0-255) for display/saving.

```python
@classmethod
def from_pil(cls, pil_image: PIL.Image) -> ImageBuffer
```

Create ImageBuffer from PIL Image.

```python
def to_pil(self) -> PIL.Image
```

Convert to PIL Image.

#### RegionMap

Container for segmentation results.

```python
class RegionMap:
    labels: np.ndarray     # Shape: (H, W), dtype: int32
    num_regions: int       # Total number of regions
    bounds: list[tuple]    # Bounding boxes per region
```

##### Constructor

```python
RegionMap(
    labels: np.ndarray,
    num_regions: int | None = None,
    bounds: list | None = None
)
```

---

### Registry

`artifice.core.registry`

#### NodeRegistry

Singleton registry for node types.

##### Methods

```python
@classmethod
def register(cls, name: str, node_class: type, category: str = "utility") -> None
```

Register a node class.

```python
@classmethod
def get(cls, name: str) -> type | None
```

Get a node class by name.

```python
@classmethod
def create(cls, name: str) -> Node | None
```

Create a node instance by name.

```python
@classmethod
def list_nodes(cls) -> list[str]
```

List all registered node names.

```python
@classmethod
def list_by_category(cls) -> dict[str, list[str]]
```

List nodes grouped by category.

##### Decorator

```python
@register_node(category: str = "utility")
def register_node(cls):
    """Decorator to register a node class."""
```

---

## Node Reference

### IO Nodes

#### ImageLoaderNode

Load images from disk.

**Inputs:** None

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Loaded image |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | "" | Path to image file |

**Supported Formats:** PNG, JPG, JPEG, TIFF, TIF, WebP, BMP, EXR

---

#### ImageSaverNode

Save images to disk.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image to save |

**Outputs:** None

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | "" | Output path |
| `format` | choice | "PNG" | Output format |
| `quality` | int | 95 | JPEG quality (1-100) |

---

### Color Nodes

#### ColorSpaceNode

Convert between color spaces.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Converted image |

**Parameters:**
| Parameter | Type | Default | Choices |
|-----------|------|---------|---------|
| `target_space` | choice | "RGB" | RGB, HSV, LAB, XYZ, YCbCr, LUV, YIQ, YXY |

---

#### ChannelSplitNode

Split image into separate channels.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `channel_0` | IMAGE | First channel |
| `channel_1` | IMAGE | Second channel |
| `channel_2` | IMAGE | Third channel |

---

#### ChannelMergeNode

Merge separate channels into image.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `channel_0` | IMAGE | First channel |
| `channel_1` | IMAGE | Second channel |
| `channel_2` | IMAGE | Third channel |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Merged image |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `colorspace` | string | "RGB" | Output color space |

---

#### ChannelSwapNode

Reorder color channels.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Swapped image |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `order` | choice | "RGB" | Channel order (RGB, RBG, GRB, GBR, BRG, BGR) |

---

### Segmentation Nodes

#### QuadtreeSegmentNode

Adaptive quadtree segmentation.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `regions` | REGIONS | Segmentation result |
| `visualization` | IMAGE | Debug visualization |

**Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `threshold` | float | 10.0 | 0-255 | Split threshold |
| `min_size` | int | 4 | 1-64 | Minimum region size |
| `max_depth` | int | 8 | 1-12 | Maximum tree depth |
| `criterion` | choice | "variance" | variance, range, gradient | Split criterion |

---

### Prediction Nodes

#### PredictNode

GLIC-style prediction.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |
| `regions` | REGIONS | Optional segmentation |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `residual` | IMAGE | Prediction residual |
| `prediction` | IMAGE | Predicted values |

**Parameters:**
| Parameter | Type | Default | Choices |
|-----------|------|---------|---------|
| `predictor_type` | choice | "paeth" | h, v, dc, paeth, average, gradient |
| `per_region` | bool | True | Apply per-region |

**Predictor Types:**
- `h`: Horizontal (left pixel)
- `v`: Vertical (top pixel)
- `dc`: Average of neighbors
- `paeth`: PNG Paeth predictor
- `average`: Average of H and V
- `gradient`: Gradient-based prediction

---

### Quantization Nodes

#### QuantizeNode

Value quantization.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Quantized image |

**Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `levels` | int | 8 | 2-256 | Quantization levels |
| `mode` | choice | "uniform" | uniform, adaptive | Quantization mode |
| `dither` | bool | False | Apply dithering |

---

### Transform Nodes

#### DCTNode

Discrete Cosine Transform.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `coefficients` | IMAGE | DCT coefficients |
| `image` | IMAGE | Reconstructed image |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | int | 8 | DCT block size |
| `keep_ratio` | float | 1.0 | Coefficient retention ratio |
| `inverse` | bool | False | Apply inverse DCT |

---

#### FFTNode

Fast Fourier Transform.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `magnitude` | IMAGE | FFT magnitude |
| `phase` | IMAGE | FFT phase |
| `image` | IMAGE | Reconstructed image |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shift` | bool | True | Center zero frequency |
| `log_scale` | bool | True | Log scale magnitude |
| `inverse` | bool | False | Apply inverse FFT |

---

#### WaveletNode

Wavelet transform.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `coefficients` | IMAGE | Wavelet coefficients |
| `image` | IMAGE | Reconstructed image |

**Parameters:**
| Parameter | Type | Default | Choices |
|-----------|------|---------|---------|
| `wavelet` | choice | "haar" | haar, db1-db10, sym2-sym8, bior1.1-bior3.3 |
| `level` | int | 3 | Decomposition levels |
| `mode` | choice | "symmetric" | symmetric, periodic, reflect |
| `inverse` | bool | False | Apply inverse |

---

#### PixelSortNode

Pixel sorting glitch effect.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |
| `mask` | IMAGE | Optional mask |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Sorted image |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | choice | "horizontal" | Sort direction |
| `sort_by` | choice | "brightness" | Sort criterion |
| `threshold_low` | float | 0.25 | Lower threshold |
| `threshold_high` | float | 0.75 | Upper threshold |
| `reverse` | bool | False | Reverse sort order |

**Direction Options:** horizontal, vertical, diagonal_down, diagonal_up

**Sort By Options:** brightness, hue, saturation, red, green, blue

---

### Corruption Nodes

#### BitShiftNode

Shift bits in pixel values.

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Corrupted image |

**Parameters:**
| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `amount` | int | 1 | -7 to 7 |
| `direction` | choice | "left" | left, right |
| `wrap` | bool | False | Wrap shifted bits |

---

#### BitFlipNode

Flip specific bits.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bit_mask` | int | 1 | Bits to flip (0-255) |
| `probability` | float | 0.1 | Flip probability |

---

#### ByteSwapNode

Swap bytes in data.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swap_distance` | int | 1 | Distance between swapped bytes |
| `probability` | float | 0.5 | Swap probability |

---

#### ByteShiftNode

Shift bytes in data stream.

**Parameters:**
| Parameter | Type | Default |
|-----------|------|---------|
| `amount` | int | 1 |
| `direction` | choice | "forward" |

---

#### DataRepeaterNode

Repeat sections of data.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repeat_length` | int | 10 | Bytes to repeat |
| `repeat_count` | int | 2 | Number of repetitions |
| `interval` | int | 100 | Bytes between repeats |

---

#### DataDropperNode

Drop sections of data.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drop_length` | int | 10 | Bytes to drop |
| `interval` | int | 100 | Bytes between drops |

---

### Utility Nodes

#### PassThroughNode

Pass data unchanged (useful for debugging).

**Inputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image |

**Outputs:**
| Port | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Same image |

---

## Color Space Reference

### Supported Color Spaces

| Name | Channels | Range | Description |
|------|----------|-------|-------------|
| RGB | R, G, B | 0-1 | Standard RGB |
| HSV | H, S, V | H: 0-1, S: 0-1, V: 0-1 | Hue, Saturation, Value |
| LAB | L, a, b | L: 0-1, a/b: varies | CIELAB perceptual |
| XYZ | X, Y, Z | varies | CIE 1931 XYZ |
| YCbCr | Y, Cb, Cr | 0-1 | Luma + Chroma |
| LUV | L, u, v | varies | CIELUV |
| YIQ | Y, I, Q | varies | NTSC color |
| YXY | Y, x, y | varies | CIE xyY |

---

For more details on implementation, see the [Architecture Guide](architecture.md).

For creating custom nodes, see the [Node Development Guide](node-development.md).
