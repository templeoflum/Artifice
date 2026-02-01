"""GPU-accelerated node implementations.

This package contains GPU versions of standard nodes that execute
via compute shaders for real-time performance.
"""

from artifice.nodes.gpu.generator import TestCardGPUNode, NoiseGPUNode
from artifice.nodes.gpu.corruption import (
    BitFlipGPUNode,
    BitShiftGPUNode,
    XORNoiseGPUNode,
    DataRepeatGPUNode,
    DataDropGPUNode,
    DataScrambleGPUNode,
    DataWeaveGPUNode,
)
from artifice.nodes.gpu.color import (
    ColorSpaceGPUNode,
    BlendGPUNode,
    InvertGPUNode,
    BrightnessContrastGPUNode,
    ThresholdGPUNode,
    PosterizeGPUNode,
    ChannelSplitGPUNode,
    ChannelMergeGPUNode,
    ChannelSwapGPUNode,
)
from artifice.nodes.gpu.quantization import QuantizeGPUNode
from artifice.nodes.gpu.transform import (
    PixelSortGPUNode,
    MirrorGPUNode,
    RotateGPUNode,
    BlurGPUNode,
    SharpenGPUNode,
    EdgeDetectGPUNode,
    DCTGPUNode,
    FFTGPUNode,
    WaveletGPUNode,
)
from artifice.nodes.gpu.glic import (
    GLICPredictGPUNode,
    GLICResidualGPUNode,
    GLICReconstructGPUNode,
)

__all__ = [
    # Generators
    "TestCardGPUNode",
    "NoiseGPUNode",
    # Corruption
    "BitFlipGPUNode",
    "BitShiftGPUNode",
    "XORNoiseGPUNode",
    "DataRepeatGPUNode",
    "DataDropGPUNode",
    "DataScrambleGPUNode",
    "DataWeaveGPUNode",
    # Color
    "ColorSpaceGPUNode",
    "BlendGPUNode",
    "InvertGPUNode",
    "BrightnessContrastGPUNode",
    "ThresholdGPUNode",
    "PosterizeGPUNode",
    "ChannelSplitGPUNode",
    "ChannelMergeGPUNode",
    "ChannelSwapGPUNode",
    # Quantization
    "QuantizeGPUNode",
    # Transform
    "PixelSortGPUNode",
    "MirrorGPUNode",
    "RotateGPUNode",
    "BlurGPUNode",
    "SharpenGPUNode",
    "EdgeDetectGPUNode",
    "DCTGPUNode",
    "FFTGPUNode",
    "WaveletGPUNode",
    # GLIC
    "GLICPredictGPUNode",
    "GLICResidualGPUNode",
    "GLICReconstructGPUNode",
]
