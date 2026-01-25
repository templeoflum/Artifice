#version 430

// BitShift compute shader
// Shifts bits in each color channel

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/output images
layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

// Parameters
uniform int shift_amount;   // Number of bits to shift (-7 to 7)
uniform bool wrap;          // Wrap bits around or fill with zeros
uniform bool affect_alpha;  // Whether to affect alpha channel

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

    // Determine which channels to process
    int numChannels = affect_alpha ? 4 : 3;

    // Process each channel
    for (int c = 0; c < numChannels; c++) {
        uint value = bytes[c];
        uint shifted;

        if (shift_amount > 0) {
            // Left shift
            if (wrap) {
                // Rotate left
                uint sh = uint(shift_amount) & 7u;
                shifted = ((value << sh) | (value >> (8u - sh))) & 0xFFu;
            } else {
                // Simple left shift
                shifted = (value << uint(shift_amount)) & 0xFFu;
            }
        } else if (shift_amount < 0) {
            // Right shift
            uint sh = uint(-shift_amount);
            if (wrap) {
                // Rotate right
                sh = sh & 7u;
                shifted = ((value >> sh) | (value << (8u - sh))) & 0xFFu;
            } else {
                // Simple right shift
                shifted = value >> sh;
            }
        } else {
            shifted = value;
        }

        bytes[c] = shifted;
    }

    // Convert back to float
    vec4 result = vec4(bytes) / 255.0;

    // Preserve original alpha if not affected
    if (!affect_alpha) {
        result.a = color.a;
    }

    imageStore(output_image, pixel, result);
}
