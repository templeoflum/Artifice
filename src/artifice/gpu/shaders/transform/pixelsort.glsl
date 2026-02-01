#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int threshold_mode;   // 0=brightness, 1=random, 2=none
uniform float threshold_low;
uniform float threshold_high;
uniform int sort_by;      // 0=brightness, 1=hue, 2=saturation, 3=red, 4=green, 5=blue
uniform int direction;    // 0=horizontal, 1=vertical
uniform int reverse_sort;
uniform int seed;

// Hash function for random mode
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

float random(uvec3 v) {
    return float(hash(v.x ^ hash(v.y ^ hash(v.z)))) / 4294967295.0;
}

// Get sort key based on sort_by mode
float get_sort_key(vec4 pixel) {
    if (sort_by == 0) {
        // Brightness (luminance)
        return dot(pixel.rgb, vec3(0.299, 0.587, 0.114));
    } else if (sort_by == 1) {
        // Hue
        float M = max(max(pixel.r, pixel.g), pixel.b);
        float m = min(min(pixel.r, pixel.g), pixel.b);
        float C = M - m;
        if (C < 0.001) return 0.0;
        float h;
        if (M == pixel.r) h = mod((pixel.g - pixel.b) / C, 6.0);
        else if (M == pixel.g) h = (pixel.b - pixel.r) / C + 2.0;
        else h = (pixel.r - pixel.g) / C + 4.0;
        return h / 6.0;
    } else if (sort_by == 2) {
        // Saturation
        float M = max(max(pixel.r, pixel.g), pixel.b);
        float m = min(min(pixel.r, pixel.g), pixel.b);
        if (M < 0.001) return 0.0;
        return (M - m) / M;
    } else if (sort_by == 3) {
        return pixel.r;
    } else if (sort_by == 4) {
        return pixel.g;
    } else {
        return pixel.b;
    }
}

// Get threshold value for a pixel based on threshold_mode
float get_threshold_value(ivec2 pos, vec4 pixel) {
    if (threshold_mode == 0) {
        // Brightness mode - use pixel brightness
        return dot(pixel.rgb, vec3(0.299, 0.587, 0.114));
    } else if (threshold_mode == 1) {
        // Random mode - random value per pixel
        return random(uvec3(pos.x, pos.y, seed));
    } else {
        // None mode - all pixels are in threshold (0.5 is always in range)
        return 0.5;
    }
}

// Check if pixel is within sort threshold
bool in_threshold(float thresh_val) {
    return thresh_val >= threshold_low && thresh_val <= threshold_high;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 my_pixel = imageLoad(input_image, pos);
    float my_thresh = get_threshold_value(pos, my_pixel);
    float my_key = get_sort_key(my_pixel);

    // If this pixel is not in threshold, just copy it
    if (!in_threshold(my_thresh)) {
        imageStore(output_image, pos, my_pixel);
        return;
    }

    // Find the span this pixel belongs to
    int line_length = (direction == 0) ? size.x : size.y;
    int my_pos_in_line = (direction == 0) ? pos.x : pos.y;
    int line_idx = (direction == 0) ? pos.y : pos.x;

    // Find span start (scan backwards)
    int span_start = my_pos_in_line;
    for (int i = my_pos_in_line - 1; i >= 0; i--) {
        ivec2 check_pos = (direction == 0) ? ivec2(i, line_idx) : ivec2(line_idx, i);
        vec4 check_pixel = imageLoad(input_image, check_pos);
        float check_thresh = get_threshold_value(check_pos, check_pixel);
        if (!in_threshold(check_thresh)) break;
        span_start = i;
    }

    // Find span end (scan forwards)
    int span_end = my_pos_in_line;
    for (int i = my_pos_in_line + 1; i < line_length; i++) {
        ivec2 check_pos = (direction == 0) ? ivec2(i, line_idx) : ivec2(line_idx, i);
        vec4 check_pixel = imageLoad(input_image, check_pos);
        float check_thresh = get_threshold_value(check_pos, check_pixel);
        if (!in_threshold(check_thresh)) break;
        span_end = i;
    }

    int span_length = span_end - span_start + 1;

    // Count how many pixels in span have smaller/larger key than me
    // This determines my position in the sorted output
    int count_less = 0;
    int count_equal_before = 0;  // For stable sort with equal keys

    for (int i = span_start; i <= span_end; i++) {
        ivec2 check_pos = (direction == 0) ? ivec2(i, line_idx) : ivec2(line_idx, i);
        vec4 check_pixel = imageLoad(input_image, check_pos);
        float check_key = get_sort_key(check_pixel);

        if (reverse_sort == 0) {
            // Ascending
            if (check_key < my_key) {
                count_less++;
            } else if (check_key == my_key && i < my_pos_in_line) {
                count_equal_before++;
            }
        } else {
            // Descending
            if (check_key > my_key) {
                count_less++;
            } else if (check_key == my_key && i < my_pos_in_line) {
                count_equal_before++;
            }
        }
    }

    // My new position in the span
    int new_pos_in_span = count_less + count_equal_before;
    int new_pos_in_line = span_start + new_pos_in_span;

    // Write to new position
    ivec2 output_pos = (direction == 0) ? ivec2(new_pos_in_line, line_idx) : ivec2(line_idx, new_pos_in_line);
    imageStore(output_image, output_pos, my_pixel);
}
