#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int block_size;  // 8 or 16
uniform int inverse;     // 0=forward, 1=inverse
uniform float quality;   // 0.0-1.0, affects coefficient preservation

const float PI = 3.14159265359;

// DCT-II coefficient
float dct_coeff(int u, int x, int N) {
    return cos((PI * float(u) * (2.0 * float(x) + 1.0)) / (2.0 * float(N)));
}

// Forward DCT for one 8x8 block
void forward_dct_block(ivec2 block_origin, ivec2 size) {
    ivec2 local_pos = ivec2(gl_LocalInvocationID.xy);
    int u = local_pos.x;
    int v = local_pos.y;

    vec4 sum = vec4(0.0);

    // Sum over all pixels in block
    for (int y = 0; y < block_size; y++) {
        for (int x = 0; x < block_size; x++) {
            ivec2 pixel_pos = block_origin + ivec2(x, y);
            if (pixel_pos.x < size.x && pixel_pos.y < size.y) {
                vec4 pixel = imageLoad(input_image, pixel_pos);
                float coeff = dct_coeff(u, x, block_size) * dct_coeff(v, y, block_size);
                sum += pixel * coeff;
            }
        }
    }

    // Normalization factors
    float cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
    float cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
    float norm = (2.0 / float(block_size)) * cu * cv;

    vec4 dct_value = sum * norm;

    // Apply quality-based coefficient attenuation
    // Higher frequencies get more attenuated at lower quality
    float freq_dist = sqrt(float(u * u + v * v)) / sqrt(float(2 * block_size * block_size));
    float attenuation = mix(1.0, 1.0 - freq_dist, 1.0 - quality);
    dct_value *= attenuation;

    ivec2 output_pos = block_origin + local_pos;
    if (output_pos.x < size.x && output_pos.y < size.y) {
        imageStore(output_image, output_pos, dct_value);
    }
}

// Inverse DCT for one 8x8 block
void inverse_dct_block(ivec2 block_origin, ivec2 size) {
    ivec2 local_pos = ivec2(gl_LocalInvocationID.xy);
    int x = local_pos.x;
    int y = local_pos.y;

    vec4 sum = vec4(0.0);

    // Sum over all frequencies
    for (int v = 0; v < block_size; v++) {
        for (int u = 0; u < block_size; u++) {
            ivec2 freq_pos = block_origin + ivec2(u, v);
            if (freq_pos.x < size.x && freq_pos.y < size.y) {
                vec4 dct_coeff_val = imageLoad(input_image, freq_pos);

                float cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
                float cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
                float coeff = cu * cv * dct_coeff(u, x, block_size) * dct_coeff(v, y, block_size);

                sum += dct_coeff_val * coeff;
            }
        }
    }

    float norm = 2.0 / float(block_size);
    vec4 pixel = sum * norm;

    // Clamp to valid range
    pixel = clamp(pixel, 0.0, 1.0);

    ivec2 output_pos = block_origin + local_pos;
    if (output_pos.x < size.x && output_pos.y < size.y) {
        imageStore(output_image, output_pos, pixel);
    }
}

void main() {
    ivec2 size = imageSize(input_image);

    // Each workgroup processes one block
    ivec2 block_id = ivec2(gl_WorkGroupID.xy);
    ivec2 block_origin = block_id * block_size;

    if (block_origin.x >= size.x || block_origin.y >= size.y) return;

    if (inverse == 0) {
        forward_dct_block(block_origin, size);
    } else {
        inverse_dct_block(block_origin, size);
    }
}
