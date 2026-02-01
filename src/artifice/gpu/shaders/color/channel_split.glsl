#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_r;
layout(rgba32f, binding = 2) writeonly uniform image2D output_g;
layout(rgba32f, binding = 3) writeonly uniform image2D output_b;
layout(rgba32f, binding = 4) writeonly uniform image2D output_a;

uniform int output_mode;  // 0=grayscale, 1=color tinted

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel = imageLoad(input_image, pos);

    if (output_mode == 0) {
        // Grayscale output
        imageStore(output_r, pos, vec4(vec3(pixel.r), 1.0));
        imageStore(output_g, pos, vec4(vec3(pixel.g), 1.0));
        imageStore(output_b, pos, vec4(vec3(pixel.b), 1.0));
        imageStore(output_a, pos, vec4(vec3(pixel.a), 1.0));
    } else {
        // Color tinted output
        imageStore(output_r, pos, vec4(pixel.r, 0.0, 0.0, 1.0));
        imageStore(output_g, pos, vec4(0.0, pixel.g, 0.0, 1.0));
        imageStore(output_b, pos, vec4(0.0, 0.0, pixel.b, 1.0));
        imageStore(output_a, pos, vec4(vec3(pixel.a), 1.0));
    }
}
