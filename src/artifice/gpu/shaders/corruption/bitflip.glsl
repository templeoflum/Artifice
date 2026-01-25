#version 430

// BitFlip compute shader
// Randomly flips bits in the image data based on probability

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/output images
layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

// Parameters
uniform float probability;  // Probability of flipping each bit [0, 1]
uniform uint seed;          // Random seed for reproducibility
uniform int bits_per_channel;  // Bits to consider (1-8)
uniform bool affect_alpha;  // Whether to affect alpha channel

// Hash function for pseudo-random numbers
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Random float in [0, 1]
float randomFloat(uint seed) {
    return float(hash(seed)) / float(0xFFFFFFFFu);
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

    // Convert to bytes [0, 255]
    uvec4 bytes = uvec4(clamp(color * 255.0, 0.0, 255.0));

    // Unique seed for this pixel
    uint pixelSeed = seed ^ uint(pixel.x + pixel.y * size.x);

    // Determine which channels to process
    int numChannels = affect_alpha ? 4 : 3;

    // Process each channel
    for (int c = 0; c < numChannels; c++) {
        uint value = bytes[c];

        // Process each bit
        for (int b = 0; b < bits_per_channel; b++) {
            // Unique seed for this bit
            uint bitSeed = hash(pixelSeed + uint(c * 8 + b));

            // Random chance to flip
            if (randomFloat(bitSeed) < probability) {
                value ^= (1u << b);
            }
        }

        bytes[c] = value;
    }

    // Convert back to float
    vec4 result = vec4(bytes) / 255.0;

    // Preserve original alpha if not affected
    if (!affect_alpha) {
        result.a = color.a;
    }

    imageStore(output_image, pixel, result);
}
