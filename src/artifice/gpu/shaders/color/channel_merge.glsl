#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_r;
layout(rgba32f, binding = 1) readonly uniform image2D input_g;
layout(rgba32f, binding = 2) readonly uniform image2D input_b;
layout(rgba32f, binding = 3) writeonly uniform image2D output_image;

uniform int r_channel;  // Which channel to use from R input (0=R, 1=G, 2=B, 3=Luma)
uniform int g_channel;  // Which channel to use from G input
uniform int b_channel;  // Which channel to use from B input

float get_channel(vec4 pixel, int ch) {
    if (ch == 0) return pixel.r;
    if (ch == 1) return pixel.g;
    if (ch == 2) return pixel.b;
    // Luminance
    return dot(pixel.rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_r);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel_r = imageLoad(input_r, pos);
    vec4 pixel_g = imageLoad(input_g, pos);
    vec4 pixel_b = imageLoad(input_b, pos);

    float r = get_channel(pixel_r, r_channel);
    float g = get_channel(pixel_g, g_channel);
    float b = get_channel(pixel_b, b_channel);

    imageStore(output_image, pos, vec4(r, g, b, 1.0));
}
