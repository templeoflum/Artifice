#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform float threshold;
uniform int mode;  // 0=binary, 1=truncate, 2=to_zero, 3=adaptive_mean
uniform int use_luminance;  // 0=per channel, 1=luminance based

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel = imageLoad(input_image, pos);
    vec3 result;

    if (use_luminance == 1) {
        float luma = dot(pixel.rgb, vec3(0.299, 0.587, 0.114));

        if (mode == 0) {
            // Binary
            float v = luma > threshold ? 1.0 : 0.0;
            result = vec3(v);
        } else if (mode == 1) {
            // Truncate
            result = luma > threshold ? vec3(threshold) : pixel.rgb;
        } else {
            // To zero
            result = luma > threshold ? pixel.rgb : vec3(0.0);
        }
    } else {
        // Per channel
        if (mode == 0) {
            result.r = pixel.r > threshold ? 1.0 : 0.0;
            result.g = pixel.g > threshold ? 1.0 : 0.0;
            result.b = pixel.b > threshold ? 1.0 : 0.0;
        } else if (mode == 1) {
            result.r = pixel.r > threshold ? threshold : pixel.r;
            result.g = pixel.g > threshold ? threshold : pixel.g;
            result.b = pixel.b > threshold ? threshold : pixel.b;
        } else {
            result.r = pixel.r > threshold ? pixel.r : 0.0;
            result.g = pixel.g > threshold ? pixel.g : 0.0;
            result.b = pixel.b > threshold ? pixel.b : 0.0;
        }
    }

    imageStore(output_image, pos, vec4(result, pixel.a));
}
