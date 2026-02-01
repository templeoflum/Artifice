#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform float amount;
uniform int radius;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 center = imageLoad(input_image, pos);

    // Calculate local average (blur)
    vec4 blur_sum = vec4(0.0);
    int count = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            ivec2 sample_pos = clamp(pos + ivec2(dx, dy), ivec2(0), size - 1);
            blur_sum += imageLoad(input_image, sample_pos);
            count++;
        }
    }

    vec4 blur = blur_sum / float(count);

    // Unsharp mask: sharpen = original + amount * (original - blur)
    vec4 result = center + amount * (center - blur);

    imageStore(output_image, pos, vec4(clamp(result.rgb, 0.0, 1.0), center.a));
}
