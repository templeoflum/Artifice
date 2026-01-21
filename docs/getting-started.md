# Getting Started with Artifice

Welcome to Artifice! This guide will walk you through your first steps with the application, from installation to creating your first glitch art project.

## Table of Contents

- [Installation](#installation)
- [Launching the Application](#launching-the-application)
- [Understanding the Interface](#understanding-the-interface)
- [Your First Project](#your-first-project)
- [Working with Nodes](#working-with-nodes)
- [Saving and Loading](#saving-and-loading)
- [Next Steps](#next-steps)

## Installation

### Prerequisites

Ensure you have Python 3.10 or higher installed:

```bash
python --version  # Should show 3.10 or higher
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/templeoflum/Artifice.git
cd Artifice

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the package
pip install -e .
```

### Verify Installation

```bash
# Run the test suite
pytest

# You should see all tests passing
```

## Launching the Application

### GUI Mode

```bash
python -m artifice
```

This opens the main application window.

### Script Mode

You can also use Artifice programmatically:

```python
from artifice.core.graph import NodeGraph
from artifice.nodes.io.loader import ImageLoaderNode

# Create and work with nodes in code
graph = NodeGraph()
loader = ImageLoaderNode()
# ... more code
```

## Understanding the Interface

When you launch Artifice, you'll see the main window divided into several panels:

```
┌─────────────────────────────────────────────────────────────────┐
│  [Menu Bar]  File  Edit  View  Graph  Help                      │
├─────────┬───────────────────────────────────────┬───────────────┤
│         │                                       │               │
│   (1)   │              (2)                      │     (3)       │
│  Node   │         Node Editor                   │   Preview     │
│ Palette │           Canvas                      │    Panel      │
│         │                                       │               │
│         │                                       ├───────────────┤
│         │                                       │               │
│         │                                       │     (4)       │
│         │                                       │  Inspector    │
│         │                                       │    Panel      │
│         │                                       │               │
└─────────┴───────────────────────────────────────┴───────────────┘
```

### 1. Node Palette (Left)

The Node Palette shows all available nodes organized by category:

- **IO** - Image loading and saving
- **Color** - Color space conversion and channel operations
- **Segmentation** - Image segmentation algorithms
- **Prediction** - GLIC-style prediction
- **Quantization** - Value quantization
- **Transform** - DCT, FFT, Wavelets, Pixel Sorting
- **Corruption** - Bit and byte manipulation
- **Utility** - Helper nodes

**To add a node**: Double-click a node in the palette, or drag it onto the canvas.

### 2. Node Editor Canvas (Center)

The main workspace where you build your processing pipeline:

- **Pan**: Middle-click and drag, or hold Space and drag
- **Zoom**: Scroll wheel, or Ctrl+Plus/Minus
- **Select**: Click on nodes, or drag a selection box
- **Multi-select**: Hold Ctrl/Shift while clicking

### 3. Preview Panel (Top Right)

Shows the output of your processing pipeline:

- Updates when you execute the graph (F5)
- Zoom with scroll wheel
- Pan by dragging

### 4. Inspector Panel (Bottom Right)

Shows parameters for the currently selected node:

- Adjust sliders and values to modify node behavior
- Changes take effect on the next execution

## Your First Project

Let's create a simple glitch effect pipeline.

### Step 1: Add an Image Loader

1. In the Node Palette, expand the **IO** category
2. Double-click **ImageLoaderNode** to add it to the canvas
3. Select the node by clicking on it
4. In the Inspector, click the file path field and enter your image path:
   ```
   C:/path/to/your/image.png
   ```

### Step 2: Add Color Space Conversion

1. Add a **ColorSpaceNode** from the **Color** category
2. In the Inspector, set **target_space** to `YCbCr`
3. Connect the nodes:
   - Click on the **image** output port (right side) of ImageLoaderNode
   - Drag to the **image** input port (left side) of ColorSpaceNode
   - Release to create the connection

### Step 3: Add Pixel Sorting

1. Add a **PixelSortNode** from the **Transform** category
2. Connect ColorSpaceNode's **image** output to PixelSortNode's **image** input
3. Configure the parameters:
   - **direction**: `horizontal`
   - **sort_by**: `brightness`
   - **threshold_low**: `0.2`
   - **threshold_high**: `0.8`

### Step 4: Convert Back and Save

1. Add another **ColorSpaceNode**
2. Set **target_space** to `RGB`
3. Connect PixelSortNode → ColorSpaceNode

4. Add an **ImageSaverNode** from **IO**
5. Set the output path: `C:/path/to/output.png`
6. Connect the final ColorSpaceNode → ImageSaverNode

### Step 5: Execute

Press **F5** or go to **Graph → Execute** to run the pipeline.

Your processed image will appear in the Preview panel and be saved to the specified path!

### Your Pipeline Should Look Like:

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Image    │    │ ColorSpace │    │  PixelSort │    │ ColorSpace │    │   Image    │
│   Loader   │───►│  (YCbCr)   │───►│            │───►│   (RGB)    │───►│   Saver    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘    └────────────┘
```

## Working with Nodes

### Selecting Nodes

- **Single select**: Click on a node
- **Multi-select**: Ctrl+Click or drag a selection rectangle
- **Select all**: Ctrl+A

### Moving Nodes

- Drag selected nodes to reposition them
- The canvas auto-scrolls near edges

### Connecting Nodes

1. Click and drag from an output port (right side of node)
2. Drop on a compatible input port (left side of another node)
3. Connections are color-coded by data type

### Disconnecting

- Right-click on a connection line and select "Delete"
- Or select a connection and press Delete

### Deleting Nodes

- Select nodes and press Delete
- Or right-click and select "Delete"

### Copying Nodes

- Select nodes and press Ctrl+C to copy
- Ctrl+V to paste
- Ctrl+D to duplicate in place

## Common Node Patterns

### Basic Glitch Pipeline

```
Loader → ColorSpace → Segment → Predict → Quantize → ColorSpace → Saver
```

### Frequency Manipulation

```
Loader → DCT → [manipulate coefficients] → Inverse DCT → Saver
```

### Data Corruption

```
Loader → BitShift → ByteSwap → DataRepeater → Saver
```

### Wavelet Glitching

```
Loader → Wavelet → [modify coefficients] → Inverse Wavelet → Saver
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Project | Ctrl+N |
| Open Project | Ctrl+O |
| Save Project | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |
| Cut | Ctrl+X |
| Copy | Ctrl+C |
| Paste | Ctrl+V |
| Delete | Delete |
| Select All | Ctrl+A |
| Execute Graph | F5 |
| Zoom In | Ctrl++ |
| Zoom Out | Ctrl+- |
| Reset Zoom | Ctrl+0 |
| Fit to View | F |

## Saving and Loading

### Save Project

- **Ctrl+S** or **File → Save**
- Projects are saved as `.artifice` files
- Contains all nodes, connections, and parameter values

### Load Project

- **Ctrl+O** or **File → Open**
- Select a `.artifice` file

### New Project

- **Ctrl+N** or **File → New**
- Clears the current graph

## Troubleshooting

### Image Not Loading

- Verify the file path is correct
- Check that the image format is supported (PNG, JPG, TIFF, WebP, BMP, EXR)
- Ensure the file exists and is readable

### Preview Not Updating

- Press F5 to manually execute the graph
- Check that all nodes are properly connected
- Look for error messages in the status bar

### Nodes Not Connecting

- Ensure port types are compatible (IMAGE to IMAGE, etc.)
- Only one connection per input port is allowed
- Check that you're connecting output → input (not input → input)

### Application Won't Start

- Verify PySide6 is installed: `pip install PySide6`
- Check Python version: `python --version` (need 3.10+)
- Try reinstalling: `pip install -e . --force-reinstall`

## Next Steps

Now that you understand the basics:

1. **Explore the Node Library**: Try different nodes and combinations
2. **Read the Architecture Guide**: Understand how the system works
3. **Create Custom Nodes**: Build your own processing algorithms
4. **Join the Community**: Share your creations and get help

### Recommended Tutorials

1. [Architecture Overview](architecture.md) - Deep dive into system design
2. [Node Development Guide](node-development.md) - Create custom nodes
3. [API Reference](api-reference.md) - Complete API documentation

---

*Converse with Chaos, Sculpt Emergence.*
