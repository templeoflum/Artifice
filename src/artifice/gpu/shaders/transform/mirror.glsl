#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int flip_horizontal;
uniform int flip_vertical;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    ivec2 src_pos = pos;

    if (flip_horizontal == 1) {
        src_pos.x = size.x - 1 - pos.x;
    }
    if (flip_vertical == 1) {
        src_pos.y = size.y - 1 - pos.y;
    }

    vec4 pixel = imageLoad(input_image, src_pos);
    imageStore(output_image, pos, pixel);
}
