#version 430

// XOR Noise compute shader
// XORs image data with noise pattern

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/output images
layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

// Parameters
uniform float intensity;    // Noise intensity [0, 1]
uniform uint seed;          // Random seed
uniform int mode;           // 0=per-pixel, 1=per-row, 2=per-block
uniform int block_size;     // Block size for mode 2
uniform bool affect_alpha;  // Whether to affect alpha channel

// Hash function
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint hash2(uvec2 v) {
    return hash(v.x ^ hash(v.y));
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    // Bounds check
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    // Load pixel
    vec4 color = imageLoad(input_image, pixel);

    // Determine noise coordinate based on mode
    uvec2 noiseCoord;
    if (mode == 0) {
        // Per-pixel noise
        noiseCoord = uvec2(pixel);
    } else if (mode == 1) {
        // Per-row noise (same noise across row)
        noiseCoord = uvec2(0, pixel.y);
    } else {
        // Per-block noise
        noiseCoord = uvec2(pixel) / uint(max(block_size, 1));
    }

    // Generate noise value
    uint noiseHash = hash2(noiseCoord) ^ seed;

    // Scale noise by intensity
    uint noiseMask = uint(intensity * 255.0);

    // Convert color to bytes
    uvec4 bytes = uvec4(clamp(color * 255.0, 0.0, 255.0));

    // Determine which channels to process
    int numChannels = affect_alpha ? 4 : 3;

    // XOR each channel with noise
    for (int c = 0; c < numChannels; c++) {
        // Different noise for each channel
        uint channelNoise = hash(noiseHash + uint(c)) & noiseMask;
        bytes[c] ^= channelNoise;
    }

    // Convert back to float
    vec4 result = vec4(bytes) / 255.0;

    // Preserve original alpha if not affected
    if (!affect_alpha) {
        result.a = color.a;
    }

    imageStore(output_image, pixel, result);
}
