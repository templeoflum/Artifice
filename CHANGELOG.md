# Changelog

All notable changes to Artifice will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- **Video Synthesis** - Oscillators, signal generators, colorizers, keyers, feedback systems (LZX/Vidiot/Fairlight CVI inspired)
- Real-time preview infrastructure for live synthesis
- Video/temporal processing nodes
- **Audio Processing** - Audio codecs, compression artifacts, sonification, spectrogram manipulation
- Audio reactivity system
- AI-powered segmentation and style transfer
- GPU acceleration via CUDA/OpenCL
- Plugin system for community nodes

### Changed
- Updated roadmap to prioritize Video Synthesis before video codec work
- Reorganized development phases (new Phase 5 for synthesis)

---

## [0.1.0] - 2025-01-20

### Added

#### Core System
- Node-based processing architecture with typed ports
- `NodeGraph` class for managing nodes and connections
- Topological sorting for correct execution order
- Project save/load in `.artifice` format
- Comprehensive node registry system

#### Data Types
- `ImageBuffer` class for image data (channel-first format, float32)
- `RegionMap` class for segmentation results
- Support for multiple color spaces

#### IO Nodes
- `ImageLoaderNode` - Load PNG, JPG, TIFF, WebP, BMP, EXR images
- `ImageSaverNode` - Save images with format options

#### Color Nodes
- `ColorSpaceNode` - Convert between 8 color spaces (RGB, HSV, LAB, XYZ, YCbCr, LUV, YIQ, YXY)
- `ChannelSplitNode` - Split image into separate channels
- `ChannelMergeNode` - Merge channels into image
- `ChannelSwapNode` - Reorder color channels

#### Segmentation Nodes
- `QuadtreeSegmentNode` - Adaptive quadtree segmentation with multiple criteria

#### Prediction Nodes
- `PredictNode` - GLIC-style predictors (H, V, DC, Paeth, Average, Gradient)

#### Quantization Nodes
- `QuantizeNode` - Scalar and adaptive quantization with optional dithering

#### Transform Nodes
- `DCTNode` - Discrete Cosine Transform (block and full-image modes)
- `FFTNode` - Fast Fourier Transform with magnitude/phase outputs
- `WaveletNode` - Multi-level wavelet decomposition (Haar, Daubechies, Symlets, Biorthogonal)
- `PixelSortNode` - Glitch-style pixel sorting with multiple modes

#### Corruption Nodes
- `BitShiftNode` - Bit shifting with wrap option
- `BitFlipNode` - Random bit flipping
- `ByteSwapNode` - Byte swapping
- `ByteShiftNode` - Byte stream shifting
- `DataRepeaterNode` - Data section repetition
- `DataDropperNode` - Data section removal

#### Utility Nodes
- `PassThroughNode` - Pass data unchanged

#### User Interface
- Qt-based main window with dockable panels
- Visual node editor with drag-and-drop
- Real-time preview panel with zoom/pan
- Inspector panel for parameter editing
- Node palette with category organization
- Full undo/redo system with command pattern
- Keyboard shortcuts for common operations

#### Testing
- 277 unit tests covering all components
- Test fixtures for common setups
- Integration tests for node pipelines

#### Documentation
- README with quick start guide
- Architecture documentation
- Getting started tutorial
- Node development guide
- API reference
- Contributing guidelines

### Technical Details
- Python 3.10+ required
- PySide6 for UI
- NumPy for array operations
- Pillow for image I/O
- SciPy for scientific computing
- PyWavelets for wavelet transforms

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2025-01-20 | Initial release with core node system, UI, and 30+ nodes |

---

[Unreleased]: https://github.com/templeoflum/Artifice/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/templeoflum/Artifice/releases/tag/v0.1.0
