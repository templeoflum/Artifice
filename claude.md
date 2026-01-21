# Artifice Engine

**Motto:** *Converse with Chaos, Sculpt Emergence.*

A next-generation glitch art co-creative environment built on a node-based architecture for emergent complexity, semantic awareness, and generative processes.

## Project Overview

Artifice Engine evolves beyond traditional glitch tools (like GLIC) into a unified platform where codec design, glitch art, generative art, and AI-assisted creation converge. The system enables users to cultivate and guide digital ecologies that produce aesthetically rich, often unpredictable visual and auditory experiences.

## Core Architecture

### Technology Stack
- **Primary Language:** Python (for AI/ML libraries, OpenCV, PyTorch, NumPy, MoviePy, Librosa) or C++ (for performance-critical components)
- **GPU Compute:** CUDA / OpenCL / Vulkan Compute / Metal for GPU nodes; GLSL/HLSL for shaders
- **UI Framework:** Qt (Python/C++) or web-based (Electron + React Flow)
- **Data Flow:** Multi-channel float buffers, GPU textures, designed for real-time/near real-time processing

### Key Design Principles
1. **Node-Based System:** All processing stages are nodes with connectable inputs/outputs
2. **Data Agnostic:** Internal representation is flexible (multi-channel float buffers, GPU textures)
3. **Real-Time Processing:** GPU acceleration, multi-threading, async operations
4. **Extensibility:** Plugin API for custom nodes, integrated Python scripting

## Node Families

### 1. Input/Output Nodes
- Loaders (PNG, JPG, EXR, TIFF, WebP, MP4, MOV, WebM, ProRes, GIF, image sequences)
- Savers (various formats, glitch-native project files)
- Live Feeds (camera, screen capture, NDI/Spout/Syphon)
- Data Feeds (OSC, MIDI, text files, web APIs)

### 2. Color Space Engine
- All standard color spaces plus spectral/custom mathematical spaces
- Dynamic color spaces with animation/modulation
- GlitchColorMatrix, ChannelMixer/Swizzler operations

### 3. Region Definition / Segmentation
- Evolved Quadtree with multiple subdivision criteria
- Geometric (Superpixel, VectorMask)
- Content-Aware AI (ObjectSegmenter, SemanticSegmenter)
- Temporal (MotionRegion for video)

### 4. Prediction / Data Generation
- GLIC predictors (H, V, DC, Paeth, Angle, Ref, SAD, BSAD)
- Temporal predictors (MotionVector, FrameDifference, OpticalFlowWarp)
- Generative (TextureSynthesis, RuleBasedFill with cellular automata/L-Systems)
- External data (ImageSampler, NoiseGenerator)
- Audio-Reactive (AudioDrive)

### 5. Quantization & Value Mapping
- Scalar, Vector, Adaptive quantizers
- PaletteRemapper, CurveMap, GradientMap, ImageColormap

### 6. Transformation
- Wavelet/Frequency transforms (DCT, DFT/FFT)
- Geometric (PixelSorter, SpatialWarp, FeedbackDisplace, SlitScan)
- Convolution/Filtering with glitchable parameters
- ShaderNode (GLSL/HLSL/ISF custom shaders)
- AI-Driven (StyleTransfer, EffectEmulator)

### 7. Data Manipulation & Corruption
- BitShifter/Flipper, ByteSwapper/Shifter
- DataRepeater/Dropper, StreamWeaver
- HeaderCorruptor (simulated codec corruption)

### 8. Temporal Processing (Video)
- FrameBlend/Difference, FrameBuffer, FrameScrambler
- TimeStretcher/Squeezer, OpticalFlowToolkit

### 9. Compositing & Utility
- BlendNode, MaskingNode, FeedbackLoop
- ParameterModulator (LFOs, envelopes, audio, OSC, MIDI)
- ScriptNode (Python/Lua)

## Development Guidelines

### Code Organization
```
artifice-engine/
├── src/
│   ├── core/           # Node system, data flow, graph execution
│   ├── nodes/          # Node implementations by family
│   │   ├── io/
│   │   ├── color/
│   │   ├── segmentation/
│   │   ├── prediction/
│   │   ├── quantization/
│   │   ├── transform/
│   │   ├── corruption/
│   │   ├── temporal/
│   │   └── utility/
│   ├── gpu/            # GPU compute kernels, shader infrastructure
│   ├── ai/             # AI model integration (segmentation, style transfer)
│   ├── audio/          # Audio analysis, reactivity
│   └── ui/             # Node editor, preview, inspector panels
├── plugins/            # Community/user plugins
├── presets/            # Node graph presets
├── tests/
└── docs/
```

### Implementation Priority
1. **Phase 1:** Core node system, data flow, basic I/O nodes
2. **Phase 2:** Evolved GLIC nodes (color, segment, predict, quantize, transform)
3. **Phase 3:** Video/temporal nodes, real-time preview
4. **Phase 4:** AI integration, audio reactivity
5. **Phase 5:** Community features, plugin ecosystem

### Performance Considerations
- Profile continuously for real-time video performance
- GPU-accelerate computationally intensive nodes first
- Use lazy evaluation where possible in node graphs
- Implement node caching for unchanged upstream data

### Node Implementation Pattern
Each node should:
- Define clear typed inputs/outputs
- Support parameter modulation
- Handle both image and video data where applicable
- Be serializable for preset saving/loading
- Include metadata for UI display (name, category, description, parameter ranges)

## UI/UX Guidelines

- Node editor: drag-and-drop, zoomable/pannable canvas
- Real-time preview panel with video playback controls
- Inspector panel with sliders, curve editors, gradient editors
- Right-click parameter modulation linking
- Robust undo/redo history

## File Formats

- `.artifice` - Project files (node graph + state)
- `.anode` - Individual node presets
- `.amacro` - Macro node packages (grouped nodes)

## Testing

- Unit tests for each node's core algorithm
- Integration tests for node graph execution
- Performance benchmarks for video processing
- Visual regression tests for effect consistency
