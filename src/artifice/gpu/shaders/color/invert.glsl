#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform float amount;
uniform int channels;  // Bitmask: 1=R, 2=G, 4=B

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel = imageLoad(input_image, pos);
    vec3 inverted = 1.0 - pixel.rgb;

    vec3 result = pixel.rgb;

    if ((channels & 1) != 0) {
        result.r = mix(pixel.r, inverted.r, amount);
    }
    if ((channels & 2) != 0) {
        result.g = mix(pixel.g, inverted.g, amount);
    }
    if ((channels & 4) != 0) {
        result.b = mix(pixel.b, inverted.b, amount);
    }

    imageStore(output_image, pos, vec4(result, pixel.a));
}
