#version 430

// Quantize compute shader
// Reduces color precision to specified number of levels

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/output images
layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

// Parameters
uniform int levels;         // Number of quantization levels (2-256)
uniform int mode;           // 0=uniform, 1=adaptive, 2=per-channel
uniform bool dither;        // Apply ordered dithering
uniform float dither_strength;  // Dithering strength [0, 1]

// 8x8 Bayer dithering matrix
const float BAYER[64] = float[64](
     0.0/64.0, 32.0/64.0,  8.0/64.0, 40.0/64.0,  2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
    48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
    12.0/64.0, 44.0/64.0,  4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0,  6.0/64.0, 38.0/64.0,
    60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, 22.0/64.0,
     3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0,  1.0/64.0, 33.0/64.0,  9.0/64.0, 41.0/64.0,
    51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
    15.0/64.0, 47.0/64.0,  7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0,  5.0/64.0, 37.0/64.0,
    63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0, 21.0/64.0
);

float getBayerValue(ivec2 pixel) {
    int x = pixel.x & 7;  // mod 8
    int y = pixel.y & 7;
    return BAYER[y * 8 + x];
}

float quantize(float value, int numLevels, float ditherOffset) {
    // Add dither offset before quantization
    float v = value + ditherOffset;
    v = clamp(v, 0.0, 1.0);

    // Quantize
    float step = 1.0 / float(numLevels - 1);
    return floor(v / step + 0.5) * step;
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

    // Calculate dither offset
    float ditherOffset = 0.0;
    if (dither) {
        float bayerValue = getBayerValue(pixel) - 0.5;  // Center around 0
        float step = 1.0 / float(levels - 1);
        ditherOffset = bayerValue * step * dither_strength;
    }

    vec4 result;

    if (mode == 0) {
        // Uniform quantization - same levels for all channels
        result.r = quantize(color.r, levels, ditherOffset);
        result.g = quantize(color.g, levels, ditherOffset);
        result.b = quantize(color.b, levels, ditherOffset);
    } else if (mode == 1) {
        // Adaptive - more levels for green (human eye sensitivity)
        int rLevels = max(2, levels * 3 / 4);
        int gLevels = levels;
        int bLevels = max(2, levels / 2);

        result.r = quantize(color.r, rLevels, ditherOffset);
        result.g = quantize(color.g, gLevels, ditherOffset);
        result.b = quantize(color.b, bLevels, ditherOffset);
    } else {
        // Per-channel - uniform but independent
        result.r = quantize(color.r, levels, ditherOffset);
        result.g = quantize(color.g, levels, ditherOffset);
        result.b = quantize(color.b, levels, ditherOffset);
    }

    // Preserve alpha
    result.a = color.a;

    imageStore(output_image, pixel, result);
}
