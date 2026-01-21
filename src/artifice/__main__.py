"""
Entry point for running Artifice as a module.

Usage:
    python -m artifice
"""

import sys
from PySide6.QtWidgets import QApplication
from artifice.ui.main_window import MainWindow
from artifice.core.registry import NodeRegistry


def register_all_nodes():
    """Register all built-in nodes."""
    # IO nodes
    from artifice.nodes.io.loader import ImageLoaderNode
    from artifice.nodes.io.saver import ImageSaverNode
    NodeRegistry.register(ImageLoaderNode)
    NodeRegistry.register(ImageSaverNode)

    # Generator nodes
    from artifice.nodes.generator.testcard import TestCardNode
    NodeRegistry.register(TestCardNode)

    # Color nodes
    from artifice.nodes.color.colorspace import ColorSpaceNode
    from artifice.nodes.color.channel_ops import (
        ChannelSplitNode,
        ChannelMergeNode,
        ChannelSwapNode,
    )
    NodeRegistry.register(ColorSpaceNode)
    NodeRegistry.register(ChannelSplitNode)
    NodeRegistry.register(ChannelMergeNode)
    NodeRegistry.register(ChannelSwapNode)

    # Segmentation nodes
    from artifice.nodes.segmentation.quadtree import QuadtreeSegmentNode
    NodeRegistry.register(QuadtreeSegmentNode)

    # Prediction nodes
    from artifice.nodes.prediction.predict_node import (
        PredictNode,
        ResidualNode,
        ReconstructNode,
    )
    NodeRegistry.register(PredictNode)
    NodeRegistry.register(ResidualNode)
    NodeRegistry.register(ReconstructNode)

    # Quantization nodes
    from artifice.nodes.quantization.quantize_node import (
        QuantizeNode,
        DequantizeNode,
    )
    NodeRegistry.register(QuantizeNode)
    NodeRegistry.register(DequantizeNode)

    # Transform nodes
    from artifice.nodes.transform.dct import DCTNode, InverseDCTNode
    from artifice.nodes.transform.fft import FFTNode, InverseFFTNode, FFTFilterNode
    from artifice.nodes.transform.wavelet import (
        WaveletTransformNode,
        InverseWaveletNode,
        WaveletCompressNode,
    )
    from artifice.nodes.transform.pixelsort import PixelSortNode
    NodeRegistry.register(DCTNode)
    NodeRegistry.register(InverseDCTNode)
    NodeRegistry.register(FFTNode)
    NodeRegistry.register(InverseFFTNode)
    NodeRegistry.register(FFTFilterNode)
    NodeRegistry.register(WaveletTransformNode)
    NodeRegistry.register(InverseWaveletNode)
    NodeRegistry.register(WaveletCompressNode)
    NodeRegistry.register(PixelSortNode)

    # Corruption nodes
    from artifice.nodes.corruption.bit_ops import (
        BitShiftNode,
        BitFlipNode,
        ByteSwapNode,
        XORNoiseNode,
    )
    from artifice.nodes.corruption.data_ops import (
        DataRepeatNode,
        DataDropNode,
        DataWeaveNode,
        DataScrambleNode,
    )
    NodeRegistry.register(BitShiftNode)
    NodeRegistry.register(BitFlipNode)
    NodeRegistry.register(ByteSwapNode)
    NodeRegistry.register(XORNoiseNode)
    NodeRegistry.register(DataRepeatNode)
    NodeRegistry.register(DataDropNode)
    NodeRegistry.register(DataWeaveNode)
    NodeRegistry.register(DataScrambleNode)

    # Pipeline nodes
    from artifice.nodes.pipeline.glic_pipeline import (
        GLICEncodeNode,
        GLICDecodeNode,
    )
    NodeRegistry.register(GLICEncodeNode)
    NodeRegistry.register(GLICDecodeNode)

    # Utility nodes
    from artifice.nodes.utility.passthrough import PassThroughNode
    NodeRegistry.register(PassThroughNode)


def main():
    """Launch the Artifice application."""
    # Register all nodes before creating the UI
    register_all_nodes()

    app = QApplication(sys.argv)
    app.setApplicationName("Artifice")
    app.setOrganizationName("Artifice")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
