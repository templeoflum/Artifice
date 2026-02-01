#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int mode;         // 0=forward, 1=inverse, 2=magnitude, 3=phase
uniform float scale;      // Output scaling for visualization

const float PI = 3.14159265359;

// Simple 2D DFT (not true FFT, but works for visualization)
// For production, would use Stockham FFT with multiple passes

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    if (mode == 0) {
        // Forward DFT - compute frequency at (pos.x, pos.y)
        vec4 sum_real = vec4(0.0);
        vec4 sum_imag = vec4(0.0);

        // Sample subset for performance (every 4th pixel for large images)
        int step = max(1, min(size.x, size.y) / 64);

        for (int y = 0; y < size.y; y += step) {
            for (int x = 0; x < size.x; x += step) {
                vec4 pixel = imageLoad(input_image, ivec2(x, y));

                float angle = -2.0 * PI * (float(pos.x * x) / float(size.x) +
                                            float(pos.y * y) / float(size.y));
                float c = cos(angle);
                float s = sin(angle);

                sum_real += pixel * c;
                sum_imag += pixel * s;
            }
        }

        // Store as complex: RG = real components, BA = imaginary components
        // Normalized and scaled for visualization
        float norm = float(step * step) / float(size.x * size.y);
        vec4 output_val;
        output_val.rg = sum_real.rg * norm * scale;
        output_val.ba = sum_imag.rg * norm * scale;
        output_val = clamp(output_val * 0.5 + 0.5, 0.0, 1.0);

        imageStore(output_image, pos, output_val);

    } else if (mode == 1) {
        // Inverse DFT
        vec4 sum_real = vec4(0.0);

        int step = max(1, min(size.x, size.y) / 64);

        for (int v = 0; v < size.y; v += step) {
            for (int u = 0; u < size.x; u += step) {
                vec4 freq = imageLoad(input_image, ivec2(u, v));
                // Decode from stored format
                vec2 real_part = freq.rg * 2.0 - 1.0;
                vec2 imag_part = freq.ba * 2.0 - 1.0;

                float angle = 2.0 * PI * (float(u * pos.x) / float(size.x) +
                                          float(v * pos.y) / float(size.y));
                float c = cos(angle);
                float s = sin(angle);

                sum_real.r += real_part.r * c - imag_part.r * s;
                sum_real.g += real_part.g * c - imag_part.g * s;
            }
        }

        vec4 output_val = vec4(sum_real.rg, sum_real.r, 1.0);
        output_val = clamp(output_val * scale, 0.0, 1.0);
        imageStore(output_image, pos, output_val);

    } else if (mode == 2) {
        // Magnitude spectrum
        vec4 pixel = imageLoad(input_image, pos);
        vec2 real_part = pixel.rg * 2.0 - 1.0;
        vec2 imag_part = pixel.ba * 2.0 - 1.0;

        float mag_r = sqrt(real_part.r * real_part.r + imag_part.r * imag_part.r);
        float mag_g = sqrt(real_part.g * real_part.g + imag_part.g * imag_part.g);

        // Log scale for better visualization
        mag_r = log(1.0 + mag_r * scale) / log(1.0 + scale);
        mag_g = log(1.0 + mag_g * scale) / log(1.0 + scale);

        vec4 output_val = vec4(mag_r, mag_g, (mag_r + mag_g) * 0.5, 1.0);
        imageStore(output_image, pos, output_val);

    } else {
        // Phase spectrum
        vec4 pixel = imageLoad(input_image, pos);
        vec2 real_part = pixel.rg * 2.0 - 1.0;
        vec2 imag_part = pixel.ba * 2.0 - 1.0;

        float phase_r = atan(imag_part.r, real_part.r) / PI * 0.5 + 0.5;
        float phase_g = atan(imag_part.g, real_part.g) / PI * 0.5 + 0.5;

        vec4 output_val = vec4(phase_r, phase_g, (phase_r + phase_g) * 0.5, 1.0);
        imageStore(output_image, pos, output_val);
    }
}
