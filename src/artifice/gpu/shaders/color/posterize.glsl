#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int levels;  // Number of levels per channel (2-256)

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel = imageLoad(input_image, pos);

    // Quantize each channel to specified number of levels
    float scale = float(levels - 1);
    vec3 result;
    result.r = floor(pixel.r * scale + 0.5) / scale;
    result.g = floor(pixel.g * scale + 0.5) / scale;
    result.b = floor(pixel.b * scale + 0.5) / scale;

    imageStore(output_image, pos, vec4(clamp(result, 0.0, 1.0), pixel.a));
}
