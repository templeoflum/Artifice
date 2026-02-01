#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int radius;
uniform int blur_type;  // 0=box, 1=gaussian

// Gaussian weight
float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 sum = vec4(0.0);
    float weight_sum = 0.0;

    float sigma = float(radius) / 2.0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            ivec2 sample_pos = pos + ivec2(dx, dy);

            // Clamp to image bounds
            sample_pos = clamp(sample_pos, ivec2(0), size - 1);

            vec4 sample_color = imageLoad(input_image, sample_pos);

            float weight;
            if (blur_type == 0) {
                // Box blur - uniform weight
                weight = 1.0;
            } else {
                // Gaussian blur
                float dist = length(vec2(dx, dy));
                weight = gaussian(dist, sigma);
            }

            sum += sample_color * weight;
            weight_sum += weight;
        }
    }

    vec4 result = sum / weight_sum;
    imageStore(output_image, pos, result);
}
