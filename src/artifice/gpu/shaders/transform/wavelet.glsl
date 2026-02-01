#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int levels;       // Decomposition levels (1-4)
uniform int inverse;      // 0=forward, 1=inverse
uniform int wavelet_type; // 0=Haar, 1=Daubechies-2
uniform float threshold;  // Coefficient threshold for compression

// Haar wavelet coefficients
const float H0 = 0.7071067811865476;  // 1/sqrt(2)
const float H1 = 0.7071067811865476;
const float G0 = 0.7071067811865476;
const float G1 = -0.7071067811865476;

// Daubechies-2 coefficients
const float D2_H0 = 0.4829629131445341;
const float D2_H1 = 0.8365163037378079;
const float D2_H2 = 0.2241438680420134;
const float D2_H3 = -0.1294095225512604;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // For a true multi-level wavelet, we'd need multiple passes
    // This is a simplified single-pass visualization

    if (inverse == 0) {
        // Forward wavelet - simplified Haar-like decomposition

        // Determine which quadrant we're in for the output
        int half_w = size.x / 2;
        int half_h = size.y / 2;

        vec4 result;

        if (pos.x < half_w && pos.y < half_h) {
            // LL (approximation) - top left
            ivec2 src = pos * 2;
            vec4 p00 = imageLoad(input_image, src);
            vec4 p10 = imageLoad(input_image, src + ivec2(1, 0));
            vec4 p01 = imageLoad(input_image, src + ivec2(0, 1));
            vec4 p11 = imageLoad(input_image, src + ivec2(1, 1));

            // Low-pass both directions
            result = (p00 + p10 + p01 + p11) * 0.25;

        } else if (pos.x >= half_w && pos.y < half_h) {
            // LH (horizontal detail) - top right
            ivec2 src = ivec2((pos.x - half_w) * 2, pos.y * 2);
            vec4 p00 = imageLoad(input_image, src);
            vec4 p10 = imageLoad(input_image, src + ivec2(1, 0));
            vec4 p01 = imageLoad(input_image, src + ivec2(0, 1));
            vec4 p11 = imageLoad(input_image, src + ivec2(1, 1));

            // Low-pass horizontal, high-pass vertical
            result = ((p00 + p10) - (p01 + p11)) * 0.25 + 0.5;

        } else if (pos.x < half_w && pos.y >= half_h) {
            // HL (vertical detail) - bottom left
            ivec2 src = ivec2(pos.x * 2, (pos.y - half_h) * 2);
            vec4 p00 = imageLoad(input_image, src);
            vec4 p10 = imageLoad(input_image, src + ivec2(1, 0));
            vec4 p01 = imageLoad(input_image, src + ivec2(0, 1));
            vec4 p11 = imageLoad(input_image, src + ivec2(1, 1));

            // High-pass horizontal, low-pass vertical
            result = ((p00 - p10) + (p01 - p11)) * 0.25 + 0.5;

        } else {
            // HH (diagonal detail) - bottom right
            ivec2 src = ivec2((pos.x - half_w) * 2, (pos.y - half_h) * 2);
            vec4 p00 = imageLoad(input_image, src);
            vec4 p10 = imageLoad(input_image, src + ivec2(1, 0));
            vec4 p01 = imageLoad(input_image, src + ivec2(0, 1));
            vec4 p11 = imageLoad(input_image, src + ivec2(1, 1));

            // High-pass both directions
            result = (p00 - p10 - p01 + p11) * 0.25 + 0.5;
        }

        // Apply threshold to detail coefficients (not LL)
        if (!(pos.x < half_w && pos.y < half_h) && threshold > 0.0) {
            vec4 centered = result - 0.5;
            vec4 sign_val = sign(centered);
            vec4 abs_val = abs(centered);
            abs_val = max(abs_val - threshold, vec4(0.0));
            result = sign_val * abs_val + 0.5;
        }

        imageStore(output_image, pos, result);

    } else {
        // Inverse wavelet

        int half_w = size.x / 2;
        int half_h = size.y / 2;

        // Determine which source quadrant pixels we need
        int src_x = pos.x / 2;
        int src_y = pos.y / 2;
        int sub_x = pos.x % 2;
        int sub_y = pos.y % 2;

        // Load from all four quadrants
        vec4 ll = imageLoad(input_image, ivec2(src_x, src_y));
        vec4 lh = imageLoad(input_image, ivec2(src_x + half_w, src_y)) - 0.5;
        vec4 hl = imageLoad(input_image, ivec2(src_x, src_y + half_h)) - 0.5;
        vec4 hh = imageLoad(input_image, ivec2(src_x + half_w, src_y + half_h)) - 0.5;

        vec4 result;

        if (sub_x == 0 && sub_y == 0) {
            result = ll + lh + hl + hh;
        } else if (sub_x == 1 && sub_y == 0) {
            result = ll + lh - hl - hh;
        } else if (sub_x == 0 && sub_y == 1) {
            result = ll - lh + hl - hh;
        } else {
            result = ll - lh - hl + hh;
        }

        result = clamp(result, 0.0, 1.0);
        imageStore(output_image, pos, result);
    }
}
