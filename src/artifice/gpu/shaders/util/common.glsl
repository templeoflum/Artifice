// Common utility functions for Artifice compute shaders
// Include this in other shaders with: #include "util/common.glsl"
// Note: GLSL doesn't have real includes, so this is for reference/copy-paste

// ============================================================================
// Constants
// ============================================================================

#define PI 3.14159265358979323846
#define TAU 6.28318530717958647692
#define E 2.71828182845904523536

// ============================================================================
// Type aliases for clarity
// ============================================================================

// vec4 color: RGBA in [0, 1]
// vec3 rgb: RGB in [0, 1]
// float gray: Grayscale in [0, 1]

// ============================================================================
// Color utilities
// ============================================================================

// Luminance weights (Rec. 709)
const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

// Alternative luminance weights (Rec. 601)
const vec3 LUMA_WEIGHTS_601 = vec3(0.299, 0.587, 0.114);

float luminance(vec3 color) {
    return dot(color, LUMA_WEIGHTS);
}

float luminance601(vec3 color) {
    return dot(color, LUMA_WEIGHTS_601);
}

// ============================================================================
// Math utilities
// ============================================================================

float remap(float value, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 saturate(vec3 x) {
    return clamp(x, 0.0, 1.0);
}

vec4 saturate(vec4 x) {
    return clamp(x, 0.0, 1.0);
}

// Smooth step with configurable edges
float smootherstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// ============================================================================
// Bit manipulation (for corruption effects)
// ============================================================================

// Convert float [0,1] to uint [0, 255] per channel
uvec4 floatToBytes(vec4 color) {
    return uvec4(clamp(color * 255.0, 0.0, 255.0));
}

// Convert uint [0, 255] per channel back to float [0,1]
vec4 bytesToFloat(uvec4 bytes) {
    return vec4(bytes) / 255.0;
}

// Pack RGBA into single uint (little endian)
uint packRGBA(vec4 color) {
    uvec4 bytes = floatToBytes(color);
    return bytes.r | (bytes.g << 8) | (bytes.b << 16) | (bytes.a << 24);
}

// Unpack single uint to RGBA
vec4 unpackRGBA(uint packed) {
    uvec4 bytes;
    bytes.r = packed & 0xFFu;
    bytes.g = (packed >> 8) & 0xFFu;
    bytes.b = (packed >> 16) & 0xFFu;
    bytes.a = (packed >> 24) & 0xFFu;
    return bytesToFloat(bytes);
}

// ============================================================================
// Coordinate utilities
// ============================================================================

// Get normalized UV from pixel coordinates
vec2 pixelToUV(ivec2 pixel, ivec2 size) {
    return (vec2(pixel) + 0.5) / vec2(size);
}

// Get pixel coordinates from UV
ivec2 uvToPixel(vec2 uv, ivec2 size) {
    return ivec2(uv * vec2(size));
}

// Check if pixel is within bounds
bool inBounds(ivec2 pixel, ivec2 size) {
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < size.x && pixel.y < size.y;
}

// ============================================================================
// Random number generation (for noise/corruption)
// ============================================================================

// Hash function for pseudo-random numbers
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint hash2(uvec2 v) {
    return hash(v.x ^ hash(v.y));
}

uint hash3(uvec3 v) {
    return hash(v.x ^ hash(v.y) ^ hash(v.z));
}

// Random float in [0, 1] from uint seed
float randomFloat(uint seed) {
    return float(hash(seed)) / float(0xFFFFFFFFu);
}

// Random float in [0, 1] from 2D coordinates
float random2D(ivec2 coord, uint seed) {
    return randomFloat(hash2(uvec2(coord)) ^ seed);
}

// ============================================================================
// Image sampling helpers
// ============================================================================

// Safe image load with bounds checking (returns black if out of bounds)
vec4 safeImageLoad(readonly image2D img, ivec2 coord) {
    ivec2 size = imageSize(img);
    if (inBounds(coord, size)) {
        return imageLoad(img, coord);
    }
    return vec4(0.0);
}

// Image load with edge clamping
vec4 clampedImageLoad(readonly image2D img, ivec2 coord) {
    ivec2 size = imageSize(img);
    coord = clamp(coord, ivec2(0), size - ivec2(1));
    return imageLoad(img, coord);
}

// Image load with wrapping
vec4 wrappedImageLoad(readonly image2D img, ivec2 coord) {
    ivec2 size = imageSize(img);
    coord = ((coord % size) + size) % size;  // Proper modulo for negatives
    return imageLoad(img, coord);
}
