#version 430

/*
 * Data Weave Shader
 *
 * Interleaves two images together with configurable row/column weaving.
 * Creates scan-line and interlace-like glitch effects.
 *
 * Can also blend at boundaries for smoother transitions.
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_a;
layout(rgba32f, binding = 1) readonly uniform image2D input_b;
layout(rgba32f, binding = 2) writeonly uniform image2D output_image;

uniform int weave_size;      // How many rows/columns before switching
uniform int direction;       // 0=horizontal (rows), 1=vertical (columns), 2=checker
uniform float blend_width;   // Blend at boundaries (0 = hard cut, 1 = full blend)
uniform float mix_amount;    // Overall mix between A and B (0.5 = equal weave)
uniform int offset;          // Offset the weave pattern

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_a);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel_a = imageLoad(input_a, pos);
    vec4 pixel_b = imageLoad(input_b, pos);

    float blend_factor;

    if (direction == 0) {
        // Horizontal weave (alternating rows)
        int row_in_pattern = (pos.y + offset) % (weave_size * 2);
        float pattern_pos = float(row_in_pattern) / float(weave_size);

        if (pattern_pos < 1.0) {
            // First half: mostly A
            float dist_to_edge = min(pattern_pos, 1.0 - pattern_pos);
            blend_factor = smoothstep(0.0, blend_width * 0.5, dist_to_edge);
        } else {
            // Second half: mostly B
            float local_pos = pattern_pos - 1.0;
            float dist_to_edge = min(local_pos, 1.0 - local_pos);
            blend_factor = 1.0 - smoothstep(0.0, blend_width * 0.5, dist_to_edge);
        }
    } else if (direction == 1) {
        // Vertical weave (alternating columns)
        int col_in_pattern = (pos.x + offset) % (weave_size * 2);
        float pattern_pos = float(col_in_pattern) / float(weave_size);

        if (pattern_pos < 1.0) {
            float dist_to_edge = min(pattern_pos, 1.0 - pattern_pos);
            blend_factor = smoothstep(0.0, blend_width * 0.5, dist_to_edge);
        } else {
            float local_pos = pattern_pos - 1.0;
            float dist_to_edge = min(local_pos, 1.0 - local_pos);
            blend_factor = 1.0 - smoothstep(0.0, blend_width * 0.5, dist_to_edge);
        }
    } else {
        // Checkerboard weave
        int check_x = ((pos.x + offset) / weave_size) % 2;
        int check_y = ((pos.y + offset) / weave_size) % 2;
        bool use_a = (check_x ^ check_y) == 0;

        // Calculate distance from block edge for blending
        int local_x = (pos.x + offset) % weave_size;
        int local_y = (pos.y + offset) % weave_size;
        float dist_x = min(float(local_x), float(weave_size - 1 - local_x)) / float(weave_size);
        float dist_y = min(float(local_y), float(weave_size - 1 - local_y)) / float(weave_size);
        float dist_to_edge = min(dist_x, dist_y);

        if (use_a) {
            blend_factor = smoothstep(0.0, blend_width * 0.5, dist_to_edge);
        } else {
            blend_factor = 1.0 - smoothstep(0.0, blend_width * 0.5, dist_to_edge);
        }
    }

    // Apply overall mix amount bias
    blend_factor = mix(mix_amount, blend_factor, 0.5 + abs(mix_amount - 0.5));

    vec4 result = mix(pixel_b, pixel_a, blend_factor);
    result.a = 1.0;

    imageStore(output_image, pos, result);
}
