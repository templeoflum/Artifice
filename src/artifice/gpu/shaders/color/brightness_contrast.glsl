#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform float brightness;  // -1 to 1
uniform float contrast;    // 0 to 2 (1 = normal)
uniform float saturation;  // 0 to 2 (1 = normal)
uniform float gamma;       // 0.1 to 3 (1 = normal)

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 pixel = imageLoad(input_image, pos);
    vec3 color = pixel.rgb;

    // Brightness
    color += brightness;

    // Contrast (around 0.5 midpoint)
    color = (color - 0.5) * contrast + 0.5;

    // Saturation
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luma), color, saturation);

    // Gamma
    color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));

    imageStore(output_image, pos, vec4(clamp(color, 0.0, 1.0), pixel.a));
}
