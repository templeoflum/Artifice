"""Tests for the GPU subsystem.

These tests verify that the GPU backend, textures, and compute shaders
work correctly.
"""

import pytest
import numpy as np

# Skip all tests if ModernGL is not available
moderngl = pytest.importorskip("moderngl")


@pytest.fixture
def gpu_backend():
    """Create a GPU backend for testing."""
    from artifice.gpu.moderngl_backend import ModernGLBackend

    backend = ModernGLBackend(standalone=True)
    backend.initialize()
    yield backend
    backend.shutdown()


class TestModernGLBackend:
    """Tests for the ModernGL backend."""

    def test_initialization(self, gpu_backend):
        """Test backend initializes correctly."""
        assert gpu_backend.is_initialized
        assert gpu_backend.ctx is not None

    def test_create_texture(self, gpu_backend):
        """Test texture creation."""
        from artifice.gpu.backend import TextureFormat

        texture = gpu_backend.create_texture(256, 256, TextureFormat.RGBA32F)

        assert texture.width == 256
        assert texture.height == 256
        assert texture.channels == 4

        gpu_backend.destroy_texture(texture)

    def test_texture_upload_download(self, gpu_backend):
        """Test uploading and downloading texture data."""
        from artifice.gpu.backend import TextureFormat

        texture = gpu_backend.create_texture(64, 64, TextureFormat.RGBA32F)

        # Create test data
        data = np.random.rand(64, 64, 4).astype(np.float32)

        # Upload
        texture.upload(data)

        # Download
        result = texture.download()

        # Verify
        np.testing.assert_array_almost_equal(data, result, decimal=5)

        gpu_backend.destroy_texture(texture)

    def test_create_buffer(self, gpu_backend):
        """Test buffer creation."""
        from artifice.gpu.backend import BufferUsage

        buffer = gpu_backend.create_buffer(1024, BufferUsage.UNIFORM)

        assert buffer.size == 1024

        gpu_backend.destroy_buffer(buffer)

    def test_buffer_write_read(self, gpu_backend):
        """Test buffer write and read operations."""
        from artifice.gpu.backend import BufferUsage

        buffer = gpu_backend.create_buffer(16, BufferUsage.UNIFORM)

        # Write floats
        buffer.write_floats([1.0, 2.0, 3.0, 4.0])

        # Read back
        data = buffer.read()
        values = np.frombuffer(data, dtype=np.float32)

        np.testing.assert_array_almost_equal(values, [1.0, 2.0, 3.0, 4.0])

        gpu_backend.destroy_buffer(buffer)


class TestComputeShaders:
    """Tests for compute shader execution."""

    def test_compile_simple_shader(self, gpu_backend):
        """Test compiling a simple compute shader."""
        shader_src = """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;

        layout(rgba32f, binding = 0) readonly uniform image2D input_image;
        layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(input_image);

            if (pixel.x >= size.x || pixel.y >= size.y) return;

            vec4 color = imageLoad(input_image, pixel);
            imageStore(output_image, pixel, color);
        }
        """

        shader = gpu_backend.compile_compute_shader(shader_src)
        assert shader is not None

    def test_passthrough_shader(self, gpu_backend):
        """Test a passthrough compute shader."""
        from artifice.gpu.backend import TextureFormat

        # Create textures
        input_tex = gpu_backend.create_texture(64, 64, TextureFormat.RGBA32F)
        output_tex = gpu_backend.create_texture(64, 64, TextureFormat.RGBA32F)

        # Create test data
        data = np.random.rand(64, 64, 4).astype(np.float32)
        input_tex.upload(data)

        # Compile shader
        shader_src = """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;

        layout(rgba32f, binding = 0) readonly uniform image2D input_image;
        layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(input_image);

            if (pixel.x >= size.x || pixel.y >= size.y) return;

            vec4 color = imageLoad(input_image, pixel);
            imageStore(output_image, pixel, color);
        }
        """
        shader = gpu_backend.compile_compute_shader(shader_src)

        # Bind textures
        input_tex.bind_as_image(0, "read")
        output_tex.bind_as_image(1, "write")

        # Dispatch
        gpu_backend.dispatch_compute(shader, 4, 4, 1)
        gpu_backend.sync()

        # Verify output
        result = output_tex.download()
        np.testing.assert_array_almost_equal(data, result, decimal=5)

        gpu_backend.destroy_texture(input_tex)
        gpu_backend.destroy_texture(output_tex)

    def test_invert_shader(self, gpu_backend):
        """Test a color invert compute shader."""
        from artifice.gpu.backend import TextureFormat

        # Create textures
        input_tex = gpu_backend.create_texture(32, 32, TextureFormat.RGBA32F)
        output_tex = gpu_backend.create_texture(32, 32, TextureFormat.RGBA32F)

        # Create solid color test data
        data = np.full((32, 32, 4), [0.2, 0.4, 0.6, 1.0], dtype=np.float32)
        input_tex.upload(data)

        # Compile invert shader
        shader_src = """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;

        layout(rgba32f, binding = 0) readonly uniform image2D input_image;
        layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(input_image);

            if (pixel.x >= size.x || pixel.y >= size.y) return;

            vec4 color = imageLoad(input_image, pixel);
            vec4 inverted = vec4(1.0 - color.rgb, color.a);
            imageStore(output_image, pixel, inverted);
        }
        """
        shader = gpu_backend.compile_compute_shader(shader_src)

        # Bind and dispatch
        input_tex.bind_as_image(0, "read")
        output_tex.bind_as_image(1, "write")
        gpu_backend.dispatch_compute(shader, 2, 2, 1)
        gpu_backend.sync()

        # Verify output
        result = output_tex.download()
        expected = np.full((32, 32, 4), [0.8, 0.6, 0.4, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        gpu_backend.destroy_texture(input_tex)
        gpu_backend.destroy_texture(output_tex)


class TestTexturePool:
    """Tests for texture pooling."""

    def test_texture_pool_acquire_release(self, gpu_backend):
        """Test acquiring and releasing textures from pool."""
        from artifice.gpu.texture import TexturePool

        pool = TexturePool(gpu_backend)

        # Acquire texture
        tex1 = pool.acquire(256, 256, 4)
        assert tex1.width == 256
        assert tex1.height == 256

        # Release back to pool
        pool.release(tex1)

        # Acquire again - should get same texture back
        tex2 = pool.acquire(256, 256, 4)

        # Stats should show pooling worked
        stats = pool.stats
        assert stats["total"] >= 1

        pool.clear()

    def test_texture_pool_different_sizes(self, gpu_backend):
        """Test pool handles different sizes correctly."""
        from artifice.gpu.texture import TexturePool

        pool = TexturePool(gpu_backend)

        tex1 = pool.acquire(128, 128, 4)
        tex2 = pool.acquire(256, 256, 4)

        assert tex1.size != tex2.size

        pool.release(tex1)
        pool.release(tex2)

        stats = pool.stats
        assert stats["pooled"] == 2

        pool.clear()


class TestTripleBuffer:
    """Tests for triple buffering."""

    def test_triple_buffer_creation(self, gpu_backend):
        """Test triple buffer creation."""
        from artifice.core.stream_executor import TripleBuffer

        buffer = TripleBuffer(gpu_backend, 640, 480)

        assert buffer.size == (640, 480)
        assert buffer.get_write_texture() is not None
        assert buffer.get_display_texture() is not None

        buffer.release()

    def test_triple_buffer_swap(self, gpu_backend):
        """Test triple buffer swapping."""
        from artifice.core.stream_executor import TripleBuffer

        buffer = TripleBuffer(gpu_backend, 320, 240)

        # Get initial textures
        write1 = buffer.get_write_texture()
        display1 = buffer.get_display_texture()

        # Swap
        buffer.swap()

        # Textures should have rotated
        write2 = buffer.get_write_texture()
        display2 = buffer.get_display_texture()

        assert write1 != write2
        assert display1 != display2

        buffer.release()


class TestGPUNodes:
    """Tests for GPU node implementations."""

    def test_bitflip_node_creation(self):
        """Test BitFlip GPU node can be created."""
        from artifice.core.gpu_node import BitFlipGPUNode

        node = BitFlipGPUNode()

        assert node.name == "Bit Flip (GPU)"
        assert node.category == "Corruption"
        assert "probability" in node.parameters

    def test_quantize_node_creation(self):
        """Test Quantize GPU node can be created."""
        from artifice.core.gpu_node import QuantizeGPUNode

        node = QuantizeGPUNode()

        assert node.name == "Quantize (GPU)"
        assert node.category == "Quantization"
        assert "levels" in node.parameters

    def test_colorspace_node_creation(self):
        """Test ColorSpace GPU node can be created."""
        from artifice.core.gpu_node import ColorSpaceGPUNode

        node = ColorSpaceGPUNode()

        assert node.name == "Color Space (GPU)"
        assert node.category == "Color"
        assert "from_space" in node.parameters
        assert "to_space" in node.parameters


class TestShaderLoading:
    """Tests for shader loading from files."""

    def test_load_bitflip_shader(self, gpu_backend):
        """Test loading the bitflip shader."""
        shader_src = gpu_backend.load_shader("corruption/bitflip.glsl")

        assert "#version 430" in shader_src
        assert "probability" in shader_src
        assert "void main()" in shader_src

    def test_load_quantize_shader(self, gpu_backend):
        """Test loading the quantize shader."""
        shader_src = gpu_backend.load_shader("quantization/quantize.glsl")

        assert "#version 430" in shader_src
        assert "levels" in shader_src

    def test_load_colorspace_shader(self, gpu_backend):
        """Test loading the colorspace shader."""
        shader_src = gpu_backend.load_shader("color/colorspace.glsl")

        assert "#version 430" in shader_src
        assert "rgb_to_hsv" in shader_src
        assert "from_space" in shader_src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
