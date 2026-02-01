#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int mode;  // 0=sobel, 1=prewitt, 2=laplacian, 3=roberts
uniform float strength;
uniform int show_edges_only;  // 0=overlay, 1=edges only

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float sample_luma(ivec2 pos, ivec2 size) {
    pos = clamp(pos, ivec2(0), size - 1);
    return luminance(imageLoad(input_image, pos).rgb);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 center = imageLoad(input_image, pos);

    // Sample 3x3 neighborhood
    float tl = sample_luma(pos + ivec2(-1, -1), size);
    float tc = sample_luma(pos + ivec2( 0, -1), size);
    float tr = sample_luma(pos + ivec2( 1, -1), size);
    float ml = sample_luma(pos + ivec2(-1,  0), size);
    float mc = sample_luma(pos, size);
    float mr = sample_luma(pos + ivec2( 1,  0), size);
    float bl = sample_luma(pos + ivec2(-1,  1), size);
    float bc = sample_luma(pos + ivec2( 0,  1), size);
    float br = sample_luma(pos + ivec2( 1,  1), size);

    float edge;

    if (mode == 0) {
        // Sobel
        float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
        float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
        edge = sqrt(gx*gx + gy*gy);
    } else if (mode == 1) {
        // Prewitt
        float gx = -tl - ml - bl + tr + mr + br;
        float gy = -tl - tc - tr + bl + bc + br;
        edge = sqrt(gx*gx + gy*gy);
    } else if (mode == 2) {
        // Laplacian
        edge = abs(-4.0*mc + tc + ml + mr + bc);
    } else {
        // Roberts cross
        float gx = mc - br;
        float gy = mr - bc;
        edge = sqrt(gx*gx + gy*gy);
    }

    edge = clamp(edge * strength, 0.0, 1.0);

    vec4 result;
    if (show_edges_only == 1) {
        result = vec4(vec3(edge), 1.0);
    } else {
        // Overlay edges on original
        result = vec4(mix(center.rgb, vec3(1.0), edge), center.a);
    }

    imageStore(output_image, pos, result);
}
