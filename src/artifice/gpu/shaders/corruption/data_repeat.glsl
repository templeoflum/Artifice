#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int repeat_interval;
uniform int repeat_length;
uniform int direction;  // 0=horizontal, 1=vertical

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    ivec2 src_pos = pos;

    if (direction == 0) {
        // Horizontal repeat
        int section_start = (pos.x / repeat_interval) * repeat_interval;
        int offset_in_section = pos.x - section_start;

        if (offset_in_section >= repeat_length) {
            // We're in a repeat zone - sample from the original section
            src_pos.x = section_start + (offset_in_section % repeat_length);
        }
    } else {
        // Vertical repeat
        int section_start = (pos.y / repeat_interval) * repeat_interval;
        int offset_in_section = pos.y - section_start;

        if (offset_in_section >= repeat_length) {
            src_pos.y = section_start + (offset_in_section % repeat_length);
        }
    }

    vec4 pixel = imageLoad(input_image, src_pos);
    imageStore(output_image, pos, pixel);
}
