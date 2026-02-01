#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int block_size;
uniform float scramble_ratio;
uniform int seed;

// Hash function for pseudo-random
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

float random(uint x, uint y, uint s) {
    return float(hash(x ^ hash(y ^ hash(s)))) / 4294967295.0;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // Determine which block this pixel belongs to
    int block_x = pos.x / block_size;
    int block_y = pos.y / block_size;
    int num_blocks_x = size.x / block_size;
    int num_blocks_y = size.y / block_size;

    // Position within the block
    int local_x = pos.x % block_size;
    int local_y = pos.y % block_size;

    // Check if this block should be scrambled
    float r = random(uint(block_x), uint(block_y), uint(seed));

    ivec2 src_pos = pos;

    if (r < scramble_ratio && num_blocks_x > 1 && num_blocks_y > 1) {
        // Generate a pseudo-random target block
        uint block_hash = hash(uint(block_x + block_y * num_blocks_x + seed * 1000));
        int target_block = int(block_hash % uint(num_blocks_x * num_blocks_y));

        int target_block_x = target_block % num_blocks_x;
        int target_block_y = target_block / num_blocks_x;

        src_pos.x = target_block_x * block_size + local_x;
        src_pos.y = target_block_y * block_size + local_y;
    }

    vec4 pixel = imageLoad(input_image, src_pos);
    imageStore(output_image, pos, pixel);
}
