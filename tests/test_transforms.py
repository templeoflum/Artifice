"""
Tests for transformation nodes.

Validates V3.1-V3.3 from DELIVERABLES.md
"""

import numpy as np
import pytest

from artifice.core.data_types import ImageBuffer
from artifice.nodes.transform.wavelet import (
    WaveletTransformNode,
    InverseWaveletNode,
    WaveletCompressNode,
    wavelet_transform,
    inverse_wavelet,
    compress_coefficients,
    list_wavelets,
)
from artifice.nodes.transform.dct import (
    DCTNode,
    InverseDCTNode,
    dct_2d,
    idct_2d,
)
from artifice.nodes.transform.fft import (
    FFTNode,
    InverseFFTNode,
    FFTFilterNode,
    fft_2d,
    ifft_2d,
)


class TestWaveletTransform:
    """V3.1 - Test wavelet transforms."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def grayscale_image(self):
        """Create grayscale test image."""
        data = np.random.rand(64, 64).astype(np.float32)
        return data

    def test_wavelet_roundtrip_fwt(self, grayscale_image):
        """V3.1 - FWT wavelet transform inverts correctly."""
        wavelets = ["haar", "db2", "db4", "sym4", "bior2.2", "coif2"]

        for wavelet in wavelets:
            coeffs = wavelet_transform(grayscale_image, wavelet, mode="fwt", level=2)
            reconstructed = inverse_wavelet(coeffs, wavelet, mode="fwt")

            # Should match within tolerance
            np.testing.assert_allclose(
                grayscale_image, reconstructed, atol=1e-5,
                err_msg=f"FWT roundtrip failed for {wavelet}"
            )

    def test_wavelet_roundtrip_wpt(self, grayscale_image):
        """V3.1 - WPT wavelet transform inverts correctly."""
        wavelets = ["haar", "db2", "sym4"]

        for wavelet in wavelets:
            coeffs = wavelet_transform(grayscale_image, wavelet, mode="wpt", level=2)
            reconstructed = inverse_wavelet(coeffs, wavelet, mode="wpt")

            np.testing.assert_allclose(
                grayscale_image, reconstructed, atol=1e-5,
                err_msg=f"WPT roundtrip failed for {wavelet}"
            )

    def test_wavelet_list(self):
        """68+ wavelets available via PyWavelets."""
        wavelets = list_wavelets()
        assert len(wavelets) >= 68, f"Expected 68+ wavelets, got {len(wavelets)}"

    def test_wavelet_node_roundtrip(self, test_image):
        """Wavelet nodes produce invertible output."""
        # Forward transform
        fwd_node = WaveletTransformNode()
        fwd_node.inputs["image"].default = test_image
        fwd_node.set_parameter("wavelet", "haar")
        fwd_node.set_parameter("mode", "fwt")
        fwd_node.set_parameter("level", 2)
        fwd_node.execute()

        coeffs = fwd_node.outputs["coefficients"].get_value()
        assert coeffs is not None

        # Inverse transform (uses wavelet/mode from coeffs metadata)
        inv_node = InverseWaveletNode()
        inv_node.inputs["coefficients"].default = coeffs
        inv_node.execute()

        result = inv_node.outputs["image"].get_value()
        assert result is not None

        # Should match original
        np.testing.assert_allclose(test_image.data, result.data, atol=1e-4)


class TestWaveletCompression:
    """V3.2 - Test wavelet compression effects."""

    @pytest.fixture
    def test_image(self):
        """Create test image with structure."""
        # Create image with gradients and edges
        data = np.zeros((64, 64), dtype=np.float32)
        data[:32, :] = np.linspace(0, 1, 64)[np.newaxis, :]
        data[32:, :] = np.linspace(1, 0, 64)[np.newaxis, :]
        # Add some edges
        data[16:48, 16:48] = 0.5
        return data

    def test_compression_is_lossy(self, test_image):
        """V3.2 - Coefficient zeroing creates expected artifacts."""
        coeffs = wavelet_transform(test_image, "haar", "fwt", level=3)

        # Compress by zeroing small coefficients
        compressed = compress_coefficients(coeffs, threshold=0.1)

        reconstructed = inverse_wavelet(compressed, "haar", "fwt")

        # Should be lossy
        assert not np.allclose(test_image, reconstructed)

    def test_compression_preserves_structure(self, test_image):
        """V3.2 - Compressed image maintains structure."""
        coeffs = wavelet_transform(test_image, "haar", "fwt", level=3)
        compressed = compress_coefficients(coeffs, threshold=0.05)
        reconstructed = inverse_wavelet(compressed, "haar", "fwt")

        # Calculate correlation as simple structural similarity proxy
        correlation = np.corrcoef(test_image.ravel(), reconstructed.ravel())[0, 1]
        assert correlation > 0.9, f"Structure not preserved: correlation={correlation}"

    def test_compression_node(self):
        """WaveletCompressNode reduces data."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        buffer = ImageBuffer(data, colorspace="RGB")

        # First do wavelet transform
        fwd_node = WaveletTransformNode()
        fwd_node.inputs["image"].default = buffer
        fwd_node.set_parameter("wavelet", "db4")
        fwd_node.execute()
        coeffs = fwd_node.outputs["coefficients"].get_value()

        # Then compress
        compress_node = WaveletCompressNode()
        compress_node.inputs["coefficients"].default = coeffs
        compress_node.set_parameter("threshold", 0.1)
        compress_node.execute()
        compressed = compress_node.outputs["coefficients"].get_value()

        # Then reconstruct
        inv_node = InverseWaveletNode()
        inv_node.inputs["coefficients"].default = compressed
        inv_node.execute()
        result = inv_node.outputs["image"].get_value()

        assert result is not None

        # Should be different from original (lossy compression)
        assert not np.allclose(data, result.data)

    def test_higher_threshold_more_compression(self):
        """Higher threshold zeroes more coefficients."""
        data = np.random.rand(64, 64).astype(np.float32)

        coeffs = wavelet_transform(data, "haar", "fwt", level=3)

        compressed_low = compress_coefficients(coeffs, threshold=0.01)
        compressed_high = compress_coefficients(coeffs, threshold=0.2)

        recon_low = inverse_wavelet(compressed_low, "haar", "fwt")
        recon_high = inverse_wavelet(compressed_high, "haar", "fwt")

        error_low = np.abs(data - recon_low).mean()
        error_high = np.abs(data - recon_high).mean()

        # Higher threshold should have more error
        assert error_high > error_low


class TestDCT:
    """V3.3 - Test DCT transforms."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def grayscale_image(self):
        """Create grayscale test image."""
        return np.random.rand(64, 64).astype(np.float32)

    def test_dct_roundtrip(self, grayscale_image):
        """V3.3 - DCT transform inverts correctly (without normalization)."""
        # Without normalization, DCT should be perfectly invertible
        dct_coeffs = dct_2d(grayscale_image, normalize=False)
        restored = idct_2d(dct_coeffs)

        np.testing.assert_allclose(grayscale_image, restored, atol=1e-5)

    def test_dct_roundtrip_3d(self):
        """DCT works on 3D (CHW) data."""
        data = np.random.rand(3, 64, 64).astype(np.float32)

        dct_coeffs = dct_2d(data, normalize=False)
        restored = idct_2d(dct_coeffs)

        np.testing.assert_allclose(data, restored, atol=1e-5)

    def test_dct_node_roundtrip(self, test_image):
        """DCT nodes produce invertible output (with normalize=False)."""
        # Forward DCT without normalization
        fwd_node = DCTNode()
        fwd_node.inputs["image"].default = test_image
        fwd_node.set_parameter("normalize", False)
        fwd_node.execute()

        coeffs = fwd_node.outputs["coefficients"].get_value()
        assert coeffs is not None

        # Inverse DCT
        inv_node = InverseDCTNode()
        inv_node.inputs["coefficients"].default = coeffs
        inv_node.execute()

        result = inv_node.outputs["image"].get_value()
        assert result is not None

        np.testing.assert_allclose(test_image.data, result.data, atol=1e-4)

    def test_dct_block_based(self):
        """DCT works with different block sizes."""
        data = np.random.rand(64, 64).astype(np.float32)

        for block_size in [4, 8, 16]:
            coeffs = dct_2d(data, block_size=block_size, normalize=False)
            restored = idct_2d(coeffs, block_size=block_size)

            np.testing.assert_allclose(
                data, restored, atol=1e-5,
                err_msg=f"DCT roundtrip failed for block_size={block_size}"
            )


class TestFFT:
    """V3.3 - Test FFT transforms."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        return ImageBuffer(data, colorspace="RGB")

    @pytest.fixture
    def grayscale_image(self):
        """Create grayscale test image."""
        return np.random.rand(64, 64).astype(np.float32)

    def test_fft_roundtrip(self, grayscale_image):
        """V3.3 - FFT transform inverts correctly."""
        fft_result = fft_2d(grayscale_image)
        restored = ifft_2d(fft_result)

        np.testing.assert_allclose(grayscale_image, restored, atol=1e-5)

    def test_fft_roundtrip_3d(self):
        """FFT works on 3D (CHW) data."""
        data = np.random.rand(3, 64, 64).astype(np.float32)

        fft_result = fft_2d(data)
        restored = ifft_2d(fft_result)

        np.testing.assert_allclose(data, restored, atol=1e-5)

    def test_fft_magnitude_phase(self, grayscale_image):
        """FFT separates magnitude and phase correctly."""
        fft_result = fft_2d(grayscale_image)

        assert "magnitude" in fft_result
        assert "phase" in fft_result
        assert "complex" in fft_result

        # Reconstruct from magnitude and phase
        restored = ifft_2d(fft_result, use_magnitude_phase=True)
        np.testing.assert_allclose(grayscale_image, restored, atol=1e-5)

    def test_fft_node_roundtrip(self, test_image):
        """FFT nodes produce invertible output."""
        # Forward FFT
        fwd_node = FFTNode()
        fwd_node.inputs["image"].default = test_image
        fwd_node.execute()

        fft_data = fwd_node.outputs["fft_data"].get_value()
        assert fft_data is not None

        # Inverse FFT
        inv_node = InverseFFTNode()
        inv_node.inputs["fft_data"].default = fft_data
        inv_node.execute()

        result = inv_node.outputs["image"].get_value()
        assert result is not None

        np.testing.assert_allclose(test_image.data, result.data, atol=1e-4)

    def test_fft_filter_node(self, test_image):
        """FFT filter modifies frequency content."""
        # Forward FFT
        fwd_node = FFTNode()
        fwd_node.inputs["image"].default = test_image
        fwd_node.execute()
        fft_data = fwd_node.outputs["fft_data"].get_value()

        # Apply low-pass filter
        filter_node = FFTFilterNode()
        filter_node.inputs["fft_data"].default = fft_data
        filter_node.set_parameter("low_pass", 0.5)
        filter_node.execute()
        filtered = filter_node.outputs["fft_data"].get_value()

        # Inverse FFT
        inv_node = InverseFFTNode()
        inv_node.inputs["fft_data"].default = filtered
        inv_node.execute()
        result = inv_node.outputs["image"].get_value()

        # Should be blurred (different from original)
        assert not np.allclose(test_image.data, result.data)

    def test_fft_shift(self, grayscale_image):
        """FFT shift parameter works."""
        # With shift (default)
        fft_shifted = fft_2d(grayscale_image, shift=True)
        # Without shift
        fft_unshifted = fft_2d(grayscale_image, shift=False)

        # Magnitude patterns should be different (shifted vs unshifted)
        assert not np.allclose(fft_shifted["magnitude"], fft_unshifted["magnitude"])

        # But both should round-trip correctly
        restored_shifted = ifft_2d(fft_shifted)
        restored_unshifted = ifft_2d(fft_unshifted)

        np.testing.assert_allclose(grayscale_image, restored_shifted, atol=1e-5)
        np.testing.assert_allclose(grayscale_image, restored_unshifted, atol=1e-5)
