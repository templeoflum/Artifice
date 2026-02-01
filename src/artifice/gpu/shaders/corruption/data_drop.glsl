#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int drop_interval;
uniform int drop_length;
uniform int direction;   // 0=horizontal, 1=vertical
uniform int fill_mode;   // 0=shift, 1=black, 2=previous

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    ivec2 src_pos = pos;
    bool in_drop_zone = false;

    if (direction == 0) {
        // Horizontal
        int section_start = (pos.x / drop_interval) * drop_interval;
        int offset_in_section = pos.x - section_start;
        in_drop_zone = (offset_in_section < drop_length);

        if (fill_mode == 0) {
            // Shift mode - calculate shifted position
            int num_drops_before = pos.x / drop_interval;
            int total_dropped = num_drops_before * drop_length;
            if (in_drop_zone) {
                total_dropped += offset_in_section;
            }
            src_pos.x = pos.x + total_dropped;
            if (src_pos.x >= size.x) {
                in_drop_zone = true;  // Out of bounds, treat as drop
            }
        }
    } else {
        // Vertical
        int section_start = (pos.y / drop_interval) * drop_interval;
        int offset_in_section = pos.y - section_start;
        in_drop_zone = (offset_in_section < drop_length);

        if (fill_mode == 0) {
            int num_drops_before = pos.y / drop_interval;
            int total_dropped = num_drops_before * drop_length;
            if (in_drop_zone) {
                total_dropped += offset_in_section;
            }
            src_pos.y = pos.y + total_dropped;
            if (src_pos.y >= size.y) {
                in_drop_zone = true;
            }
        }
    }

    vec4 pixel;

    if (fill_mode == 0) {
        // Shift mode
        if (src_pos.x < size.x && src_pos.y < size.y) {
            pixel = imageLoad(input_image, src_pos);
        } else {
            pixel = vec4(0.0, 0.0, 0.0, 1.0);
        }
    } else if (fill_mode == 1) {
        // Black fill
        if (in_drop_zone) {
            pixel = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            pixel = imageLoad(input_image, pos);
        }
    } else {
        // Previous fill
        if (in_drop_zone) {
            if (direction == 0 && pos.x > 0) {
                int section_start = (pos.x / drop_interval) * drop_interval;
                src_pos.x = max(0, section_start - 1);
            } else if (direction == 1 && pos.y > 0) {
                int section_start = (pos.y / drop_interval) * drop_interval;
                src_pos.y = max(0, section_start - 1);
            }
            pixel = imageLoad(input_image, src_pos);
        } else {
            pixel = imageLoad(input_image, pos);
        }
    }

    imageStore(output_image, pos, pixel);
}
