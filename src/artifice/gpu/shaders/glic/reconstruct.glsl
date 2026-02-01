#version 430

/*
 * GLIC Reconstruction Shader
 *
 * Reconstructs image: output = prediction + residual
 *
 * For glitch effects, we can use "wrong" predictions,
 * corrupted residuals, or mismatched parameters.
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D prediction;
layout(rgba32f, binding = 1) readonly uniform image2D residual;
layout(rgba32f, binding = 2) writeonly uniform image2D output_image;

uniform float residual_strength;  // 1.0 = normal, 0.0 = prediction only, >1.0 = exaggerated
uniform float prediction_blend;   // Mix between prediction and full reconstruction
uniform int clamp_output;         // 0 = no clamp (wrap artifacts), 1 = clamp to 0-1

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(prediction);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pred = imageLoad(prediction, pos);
    vec4 res = imageLoad(residual, pos);

    // Residual is stored centered at 0.5, convert back to signed
    vec4 signed_residual = (res - 0.5) * 2.0;  // -1 to 1 range

    // Reconstruct
    vec4 reconstructed = pred + signed_residual * residual_strength;

    // Blend between pure prediction and full reconstruction
    vec4 output_val = mix(pred, reconstructed, prediction_blend);

    // Optionally clamp or allow wrap-around artifacts
    if (clamp_output == 1) {
        output_val = clamp(output_val, 0.0, 1.0);
    }

    output_val.a = 1.0;
    imageStore(output_image, pos, output_val);
}
