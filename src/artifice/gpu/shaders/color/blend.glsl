#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_a;
layout(rgba32f, binding = 1) readonly uniform image2D input_b;
layout(rgba32f, binding = 2) writeonly uniform image2D output_image;

uniform int blend_mode;  // 0=mix, 1=add, 2=multiply, 3=screen, 4=overlay, 5=difference
uniform float mix_factor;

vec3 blend_overlay(vec3 a, vec3 b) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        if (a[i] < 0.5) {
            result[i] = 2.0 * a[i] * b[i];
        } else {
            result[i] = 1.0 - 2.0 * (1.0 - a[i]) * (1.0 - b[i]);
        }
    }
    return result;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_a);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 a = imageLoad(input_a, pos);
    vec4 b = imageLoad(input_b, pos);

    vec3 result;

    if (blend_mode == 0) {
        // Mix/Lerp
        result = mix(a.rgb, b.rgb, mix_factor);
    } else if (blend_mode == 1) {
        // Add
        result = a.rgb + b.rgb * mix_factor;
    } else if (blend_mode == 2) {
        // Multiply
        result = mix(a.rgb, a.rgb * b.rgb, mix_factor);
    } else if (blend_mode == 3) {
        // Screen
        result = mix(a.rgb, 1.0 - (1.0 - a.rgb) * (1.0 - b.rgb), mix_factor);
    } else if (blend_mode == 4) {
        // Overlay
        result = mix(a.rgb, blend_overlay(a.rgb, b.rgb), mix_factor);
    } else {
        // Difference
        result = mix(a.rgb, abs(a.rgb - b.rgb), mix_factor);
    }

    imageStore(output_image, pos, vec4(clamp(result, 0.0, 1.0), a.a));
}
