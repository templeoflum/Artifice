#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) writeonly uniform image2D output_image;

uniform int size;
uniform int noise_type;  // 0=white, 1=perlin, 2=simplex, 3=cellular
uniform float scale;
uniform int seed;
uniform int octaves;
uniform float persistence;
uniform int colored;  // 0=grayscale, 1=colored

// Hash functions
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

float random(uvec2 v, uint s) {
    return float(hash(v.x ^ hash(v.y ^ hash(s)))) / 4294967295.0;
}

// Gradient for Perlin noise
vec2 gradient(ivec2 p, uint s) {
    float angle = random(uvec2(p), s) * 6.28318530718;
    return vec2(cos(angle), sin(angle));
}

// Smoothstep interpolation
float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Perlin noise
float perlin(vec2 p, uint s) {
    ivec2 i = ivec2(floor(p));
    vec2 f = fract(p);

    float n00 = dot(gradient(i, s), f);
    float n10 = dot(gradient(i + ivec2(1, 0), s), f - vec2(1.0, 0.0));
    float n01 = dot(gradient(i + ivec2(0, 1), s), f - vec2(0.0, 1.0));
    float n11 = dot(gradient(i + ivec2(1, 1), s), f - vec2(1.0, 1.0));

    vec2 u = vec2(fade(f.x), fade(f.y));

    return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y) * 0.5 + 0.5;
}

// Fractal Brownian Motion
float fbm(vec2 p, uint s, int oct, float pers) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float total_amplitude = 0.0;

    for (int i = 0; i < oct; i++) {
        value += amplitude * perlin(p * frequency, s + uint(i * 1000));
        total_amplitude += amplitude;
        amplitude *= pers;
        frequency *= 2.0;
    }

    return value / total_amplitude;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

    if (pos.x >= size || pos.y >= size) return;

    vec2 uv = vec2(pos) / float(size);
    vec4 color;

    if (noise_type == 0) {
        // White noise
        if (colored == 1) {
            color.r = random(uvec2(pos), uint(seed));
            color.g = random(uvec2(pos), uint(seed + 1));
            color.b = random(uvec2(pos), uint(seed + 2));
        } else {
            float n = random(uvec2(pos), uint(seed));
            color = vec4(n, n, n, 1.0);
        }
    } else if (noise_type == 1) {
        // Perlin noise
        vec2 p = uv * scale;
        if (colored == 1) {
            color.r = fbm(p, uint(seed), octaves, persistence);
            color.g = fbm(p, uint(seed + 1000), octaves, persistence);
            color.b = fbm(p, uint(seed + 2000), octaves, persistence);
        } else {
            float n = fbm(p, uint(seed), octaves, persistence);
            color = vec4(n, n, n, 1.0);
        }
    } else {
        // Fallback to Perlin
        vec2 p = uv * scale;
        float n = fbm(p, uint(seed), octaves, persistence);
        color = vec4(n, n, n, 1.0);
    }

    color.a = 1.0;
    imageStore(output_image, pos, color);
}
