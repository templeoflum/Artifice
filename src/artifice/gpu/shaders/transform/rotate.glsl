#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform float angle;  // In radians
uniform float center_x;
uniform float center_y;
uniform int interpolation;  // 0=nearest, 1=bilinear

vec4 sample_bilinear(ivec2 size, vec2 uv) {
    vec2 pixel = uv * vec2(size) - 0.5;
    ivec2 p0 = ivec2(floor(pixel));
    vec2 f = fract(pixel);

    // Clamp to bounds
    ivec2 p00 = clamp(p0, ivec2(0), size - 1);
    ivec2 p10 = clamp(p0 + ivec2(1, 0), ivec2(0), size - 1);
    ivec2 p01 = clamp(p0 + ivec2(0, 1), ivec2(0), size - 1);
    ivec2 p11 = clamp(p0 + ivec2(1, 1), ivec2(0), size - 1);

    vec4 c00 = imageLoad(input_image, p00);
    vec4 c10 = imageLoad(input_image, p10);
    vec4 c01 = imageLoad(input_image, p01);
    vec4 c11 = imageLoad(input_image, p11);

    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // Normalize coordinates
    vec2 uv = (vec2(pos) + 0.5) / vec2(size);
    vec2 center = vec2(center_x, center_y);

    // Rotate around center
    vec2 offset = uv - center;
    float c = cos(-angle);
    float s = sin(-angle);
    vec2 rotated = vec2(
        offset.x * c - offset.y * s,
        offset.x * s + offset.y * c
    );
    vec2 src_uv = rotated + center;

    vec4 pixel;

    // Check bounds
    if (src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0) {
        pixel = vec4(0.0, 0.0, 0.0, 1.0);
    } else if (interpolation == 0) {
        // Nearest neighbor
        ivec2 src_pos = ivec2(src_uv * vec2(size));
        src_pos = clamp(src_pos, ivec2(0), size - 1);
        pixel = imageLoad(input_image, src_pos);
    } else {
        // Bilinear
        pixel = sample_bilinear(size, src_uv);
    }

    imageStore(output_image, pos, pixel);
}
