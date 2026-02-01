#version 430

/*
 * GLIC Predictor Shader
 *
 * Implements all 16 GLIC predictors on GPU plus SAD/BSAD selection.
 *
 * Special modes:
 * - predictor_mode 14: SAD (Sum of Absolute Differences) - picks BEST predictor per block
 * - predictor_mode 15: BSAD (Bad SAD) - picks WORST predictor per block (for glitch art!)
 * - predictor_mode 16: Random - random predictor per block
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_prediction;

uniform int block_size;      // Size of prediction blocks (e.g., 8, 16, 32)
uniform int predictor_mode;  // 0-13 = specific, 14=SAD, 15=BSAD, 16=Random
uniform float border_value;  // Value for out-of-bounds pixels
uniform int seed;            // Random seed for random mode

const int NUM_PREDICTORS = 14;  // Number of predictors to test (0-13)

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

// Safe pixel fetch with border handling
vec4 getPixel(ivec2 pos, ivec2 size) {
    if (pos.x < 0 || pos.x >= size.x || pos.y < 0 || pos.y >= size.y) {
        return vec4(border_value, border_value, border_value, 1.0);
    }
    return imageLoad(input_image, pos);
}

// Median of three values
float median3(float a, float b, float c) {
    return max(min(a, b), min(max(a, b), c));
}

// Predictor 0: None (zeros)
vec4 pred_none(ivec2 pos, ivec2 block_origin, ivec2 size) {
    return vec4(0.0, 0.0, 0.0, 1.0);
}

// Predictor 1: Corner (top-left corner pixel)
vec4 pred_corner(ivec2 pos, ivec2 block_origin, ivec2 size) {
    return getPixel(ivec2(block_origin.x - 1, block_origin.y - 1), size);
}

// Predictor 2: Horizontal (left edge)
vec4 pred_h(ivec2 pos, ivec2 block_origin, ivec2 size) {
    return getPixel(ivec2(block_origin.x - 1, pos.y), size);
}

// Predictor 3: Vertical (top edge)
vec4 pred_v(ivec2 pos, ivec2 block_origin, ivec2 size) {
    return getPixel(ivec2(pos.x, block_origin.y - 1), size);
}

// Predictor 4: DC (mean of edges)
vec4 pred_dc(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 total = vec4(0.0);
    int count = 0;

    // Left edge
    for (int y = 0; y < block_size; y++) {
        total += getPixel(ivec2(block_origin.x - 1, block_origin.y + y), size);
        count++;
    }

    // Top edge
    for (int x = 0; x < block_size; x++) {
        total += getPixel(ivec2(block_origin.x + x, block_origin.y - 1), size);
        count++;
    }

    // Corner
    total += getPixel(ivec2(block_origin.x - 1, block_origin.y - 1), size);
    count++;

    return total / float(count);
}

// Predictor 5: DC Median
vec4 pred_dcmedian(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 dc = pred_dc(pos, block_origin, size);
    vec4 top = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 left = getPixel(ivec2(block_origin.x - 1, pos.y), size);

    return vec4(
        median3(dc.r, top.r, left.r),
        median3(dc.g, top.g, left.g),
        median3(dc.b, top.b, left.b),
        1.0
    );
}

// Predictor 6: Median (of corner and edges)
vec4 pred_median(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 corner = getPixel(ivec2(block_origin.x - 1, block_origin.y - 1), size);
    vec4 top = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 left = getPixel(ivec2(block_origin.x - 1, pos.y), size);

    return vec4(
        median3(corner.r, top.r, left.r),
        median3(corner.g, top.g, left.g),
        median3(corner.b, top.b, left.b),
        1.0
    );
}

// Predictor 7: Average of top and left
vec4 pred_avg(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 top = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 left = getPixel(ivec2(block_origin.x - 1, pos.y), size);
    return (top + left) * 0.5;
}

// Predictor 8: TrueMotion (top + left - corner)
vec4 pred_truemotion(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 corner = getPixel(ivec2(block_origin.x - 1, block_origin.y - 1), size);
    vec4 top = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 left = getPixel(ivec2(block_origin.x - 1, pos.y), size);
    return clamp(top + left - corner, 0.0, 1.0);
}

// Predictor 9: Paeth (PNG filter)
vec4 pred_paeth(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 corner = getPixel(ivec2(block_origin.x - 1, block_origin.y - 1), size);
    vec4 top = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 left = getPixel(ivec2(block_origin.x - 1, pos.y), size);

    vec4 pp = top + left - corner;
    vec4 pa = abs(pp - left);
    vec4 pb = abs(pp - top);
    vec4 pc = abs(pp - corner);

    vec4 result;
    for (int c = 0; c < 3; c++) {
        if (pa[c] <= pb[c] && pa[c] <= pc[c]) {
            result[c] = left[c];
        } else if (pb[c] <= pc[c]) {
            result[c] = top[c];
        } else {
            result[c] = corner[c];
        }
    }
    result.a = 1.0;
    return clamp(result, 0.0, 1.0);
}

// Predictor 10: Linear Diagonal
vec4 pred_ldiag(ivec2 pos, ivec2 block_origin, ivec2 size) {
    ivec2 local = pos - block_origin;
    int ss = local.x + local.y;
    int xx_idx = min(ss + 1, block_size - 1);
    int yy_idx = min(ss, block_size - 1);

    vec4 xx = getPixel(ivec2(block_origin.x + xx_idx, block_origin.y - 1), size);
    vec4 yy = getPixel(ivec2(block_origin.x - 1, block_origin.y + yy_idx), size);

    float weight_total = float(local.x + local.y + 2);
    return (float(local.x + 1) * xx + float(local.y + 1) * yy) / weight_total;
}

// Predictor 11: H/V based on position
vec4 pred_hv(ivec2 pos, ivec2 block_origin, ivec2 size) {
    ivec2 local = pos - block_origin;
    vec4 top = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 left = getPixel(ivec2(block_origin.x - 1, pos.y), size);

    if (local.x > local.y) {
        return top;
    } else if (local.y > local.x) {
        return left;
    } else {
        return (top + left) * 0.5;
    }
}

// Predictor 12: JPEG-LS
vec4 pred_jpegls(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 c = getPixel(ivec2(pos.x - 1, block_origin.y - 1), size);
    vec4 a = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 b = getPixel(ivec2(block_origin.x - 1, pos.y), size);

    vec4 result;
    for (int ch = 0; ch < 3; ch++) {
        if (c[ch] >= max(a[ch], b[ch])) {
            result[ch] = min(a[ch], b[ch]);
        } else if (c[ch] <= min(a[ch], b[ch])) {
            result[ch] = max(a[ch], b[ch]);
        } else {
            result[ch] = a[ch] + b[ch] - c[ch];
        }
    }
    result.a = 1.0;
    return result;
}

// Predictor 13: Second-order Difference
vec4 pred_diff(ivec2 pos, ivec2 block_origin, ivec2 size) {
    vec4 x1 = getPixel(ivec2(pos.x, block_origin.y - 1), size);
    vec4 x2 = getPixel(ivec2(pos.x, block_origin.y - 2), size);
    vec4 y1 = getPixel(ivec2(block_origin.x - 1, pos.y), size);
    vec4 y2 = getPixel(ivec2(block_origin.x - 2, pos.y), size);

    return clamp((2.0 * y2 - y1 + 2.0 * x2 - x1) * 0.5, 0.0, 1.0);
}

// Get prediction for a pixel using specified predictor
vec4 getPrediction(ivec2 pos, ivec2 block_origin, ivec2 size, int pred) {
    switch (pred) {
        case 0:  return pred_none(pos, block_origin, size);
        case 1:  return pred_corner(pos, block_origin, size);
        case 2:  return pred_h(pos, block_origin, size);
        case 3:  return pred_v(pos, block_origin, size);
        case 4:  return pred_dc(pos, block_origin, size);
        case 5:  return pred_dcmedian(pos, block_origin, size);
        case 6:  return pred_median(pos, block_origin, size);
        case 7:  return pred_avg(pos, block_origin, size);
        case 8:  return pred_truemotion(pos, block_origin, size);
        case 9:  return pred_paeth(pos, block_origin, size);
        case 10: return pred_ldiag(pos, block_origin, size);
        case 11: return pred_hv(pos, block_origin, size);
        case 12: return pred_jpegls(pos, block_origin, size);
        case 13: return pred_diff(pos, block_origin, size);
        default: return pred_paeth(pos, block_origin, size);
    }
}

// Calculate SAD for a predictor over block edge pixels only
// (Using edge pixels as proxy for full block to save computation)
float calculateBlockSAD(ivec2 block_origin, ivec2 size, int pred) {
    float sad = 0.0;
    int count = 0;

    // Sample along the edges of the block
    for (int i = 0; i < block_size; i++) {
        // Top edge
        ivec2 pos_top = block_origin + ivec2(i, 0);
        if (pos_top.x < size.x && pos_top.y < size.y) {
            vec4 actual = getPixel(pos_top, size);
            vec4 predicted = getPrediction(pos_top, block_origin, size, pred);
            sad += abs(actual.r - predicted.r) + abs(actual.g - predicted.g) + abs(actual.b - predicted.b);
            count++;
        }

        // Left edge
        ivec2 pos_left = block_origin + ivec2(0, i);
        if (pos_left.x < size.x && pos_left.y < size.y) {
            vec4 actual = getPixel(pos_left, size);
            vec4 predicted = getPrediction(pos_left, block_origin, size, pred);
            sad += abs(actual.r - predicted.r) + abs(actual.g - predicted.g) + abs(actual.b - predicted.b);
            count++;
        }
    }

    return count > 0 ? sad / float(count) : 0.0;
}

// Find best (minimum SAD) predictor
int findBestPredictor(ivec2 block_origin, ivec2 size) {
    float best_sad = 1e10;
    int best_pred = 9;  // Default to Paeth

    for (int p = 0; p < NUM_PREDICTORS; p++) {
        float sad = calculateBlockSAD(block_origin, size, p);
        if (sad < best_sad) {
            best_sad = sad;
            best_pred = p;
        }
    }

    return best_pred;
}

// Find worst (maximum SAD) predictor - for glitch art!
int findWorstPredictor(ivec2 block_origin, ivec2 size) {
    float worst_sad = -1.0;
    int worst_pred = 0;  // Default to None

    for (int p = 0; p < NUM_PREDICTORS; p++) {
        float sad = calculateBlockSAD(block_origin, size, p);
        if (sad > worst_sad) {
            worst_sad = sad;
            worst_pred = p;
        }
    }

    return worst_pred;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // Calculate which block this pixel belongs to
    ivec2 block_id = pos / block_size;
    ivec2 block_origin = block_id * block_size;

    // Determine which predictor to use
    int actual_predictor;

    if (predictor_mode == 14) {
        // SAD mode - find best predictor for this block
        actual_predictor = findBestPredictor(block_origin, size);
    } else if (predictor_mode == 15) {
        // BSAD mode - find WORST predictor for this block (glitch art!)
        actual_predictor = findWorstPredictor(block_origin, size);
    } else if (predictor_mode == 16) {
        // Random mode - random predictor per block
        float r = random(uvec3(block_id.x, block_id.y, seed));
        actual_predictor = int(r * float(NUM_PREDICTORS)) % NUM_PREDICTORS;
    } else {
        // Specific predictor mode (0-13)
        actual_predictor = predictor_mode;
    }

    // Get prediction
    vec4 prediction = getPrediction(pos, block_origin, size, actual_predictor);

    imageStore(output_prediction, pos, prediction);
}
