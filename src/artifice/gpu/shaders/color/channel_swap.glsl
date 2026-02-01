#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int r_source;  // 0=R, 1=G, 2=B, 3=A, 4=Luma
uniform int g_source;
uniform int b_source;

float get_channel(vec4 pixel, int src) {
    if (src == 0) return pixel.r;
    if (src == 1) return pixel.g;
    if (src == 2) return pixel.b;
    if (src == 3) return pixel.a;
    // Luminance
    return dot(pixel.rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel = imageLoad(input_image, pos);

    vec4 result;
    result.r = get_channel(pixel, r_source);
    result.g = get_channel(pixel, g_source);
    result.b = get_channel(pixel, b_source);
    result.a = pixel.a;

    imageStore(output_image, pos, result);
}
