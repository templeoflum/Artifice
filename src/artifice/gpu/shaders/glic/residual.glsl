#version 430

/*
 * GLIC Residual Calculation Shader
 *
 * Calculates: residual = original - prediction
 * Then quantizes the result.
 *
 * For glitch effects, we can also apply the "worst" predictor
 * or corrupt the residuals.
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) readonly uniform image2D prediction;
layout(rgba32f, binding = 2) writeonly uniform image2D output_residual;

uniform int quantization_bits;   // 1-8 bits
uniform int residual_mode;       // 0=normal, 1=clamp_mod256 (GLIC glitch)
uniform float residual_scale;    // Scale factor for visualization

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 original = imageLoad(input_image, pos);
    vec4 pred = imageLoad(prediction, pos);

    // Calculate residual
    vec4 residual = original - pred;

    // Quantization levels
    float levels = pow(2.0, float(quantization_bits));
    float half_levels = levels * 0.5;

    vec4 quantized;
    for (int c = 0; c < 3; c++) {
        // Convert to signed integer range centered at 0
        float val = residual[c] * 255.0;  // -255 to 255 range

        if (residual_mode == 1) {
            // CLAMP_MOD256 mode - GLIC glitch style
            // Clamp to 0-255, then modulo
            val = mod(val + 256.0, 256.0);
        }

        // Quantize to specified bit depth
        // Map to 0-levels range, round, map back
        float normalized = (val + 255.0) / 510.0;  // Map to 0-1
        float quantized_val = floor(normalized * levels + 0.5) / levels;
        val = quantized_val * 510.0 - 255.0;  // Map back to -255 to 255

        quantized[c] = val / 255.0;  // Back to -1 to 1 range
    }
    quantized.a = 1.0;

    // Scale for visualization (residuals are often small)
    vec4 output_val = quantized * residual_scale + 0.5;  // Center at 0.5 for display

    imageStore(output_residual, pos, output_val);
}
