#version 430

// Color Space Conversion compute shader
// Converts between various color spaces

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/output images
layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

// Parameters
uniform int from_space;  // Source color space
uniform int to_space;    // Target color space

// Color space IDs
#define CS_RGB    0
#define CS_HSV    1
#define CS_HSL    2
#define CS_LAB    3
#define CS_XYZ    4
#define CS_YCbCr  5
#define CS_YUV    6
#define CS_LUV    7

// ============================================================================
// RGB <-> HSV
// ============================================================================

vec3 rgb_to_hsv(vec3 rgb) {
    float M = max(max(rgb.r, rgb.g), rgb.b);
    float m = min(min(rgb.r, rgb.g), rgb.b);
    float C = M - m;

    float h = 0.0;
    if (C > 0.0001) {
        if (M == rgb.r) {
            h = mod((rgb.g - rgb.b) / C, 6.0);
        } else if (M == rgb.g) {
            h = (rgb.b - rgb.r) / C + 2.0;
        } else {
            h = (rgb.r - rgb.g) / C + 4.0;
        }
        h /= 6.0;
    }

    float s = (M > 0.0001) ? C / M : 0.0;
    float v = M;

    return vec3(h, s, v);
}

vec3 hsv_to_rgb(vec3 hsv) {
    float h = hsv.x * 6.0;
    float s = hsv.y;
    float v = hsv.z;

    float C = v * s;
    float X = C * (1.0 - abs(mod(h, 2.0) - 1.0));
    float m = v - C;

    vec3 rgb;
    if (h < 1.0) rgb = vec3(C, X, 0.0);
    else if (h < 2.0) rgb = vec3(X, C, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, C, X);
    else if (h < 4.0) rgb = vec3(0.0, X, C);
    else if (h < 5.0) rgb = vec3(X, 0.0, C);
    else rgb = vec3(C, 0.0, X);

    return rgb + m;
}

// ============================================================================
// RGB <-> HSL
// ============================================================================

vec3 rgb_to_hsl(vec3 rgb) {
    float M = max(max(rgb.r, rgb.g), rgb.b);
    float m = min(min(rgb.r, rgb.g), rgb.b);
    float C = M - m;
    float L = (M + m) / 2.0;

    float h = 0.0;
    if (C > 0.0001) {
        if (M == rgb.r) {
            h = mod((rgb.g - rgb.b) / C, 6.0);
        } else if (M == rgb.g) {
            h = (rgb.b - rgb.r) / C + 2.0;
        } else {
            h = (rgb.r - rgb.g) / C + 4.0;
        }
        h /= 6.0;
    }

    float s = (L > 0.0001 && L < 0.9999) ? C / (1.0 - abs(2.0 * L - 1.0)) : 0.0;

    return vec3(h, s, L);
}

vec3 hsl_to_rgb(vec3 hsl) {
    float h = hsl.x * 6.0;
    float s = hsl.y;
    float L = hsl.z;

    float C = (1.0 - abs(2.0 * L - 1.0)) * s;
    float X = C * (1.0 - abs(mod(h, 2.0) - 1.0));
    float m = L - C / 2.0;

    vec3 rgb;
    if (h < 1.0) rgb = vec3(C, X, 0.0);
    else if (h < 2.0) rgb = vec3(X, C, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, C, X);
    else if (h < 4.0) rgb = vec3(0.0, X, C);
    else if (h < 5.0) rgb = vec3(X, 0.0, C);
    else rgb = vec3(C, 0.0, X);

    return rgb + m;
}

// ============================================================================
// RGB <-> XYZ (sRGB with D65 illuminant)
// ============================================================================

// Linearize sRGB
float srgb_to_linear(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

// Apply sRGB gamma
float linear_to_srgb(float c) {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0/2.4) - 0.055;
}

vec3 rgb_to_xyz(vec3 rgb) {
    // Linearize
    vec3 linear = vec3(
        srgb_to_linear(rgb.r),
        srgb_to_linear(rgb.g),
        srgb_to_linear(rgb.b)
    );

    // RGB to XYZ matrix (sRGB, D65)
    mat3 M = mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    );

    return M * linear;
}

vec3 xyz_to_rgb(vec3 xyz) {
    // XYZ to RGB matrix (sRGB, D65)
    mat3 M = mat3(
         3.2404542, -1.5371385, -0.4985314,
        -0.9692660,  1.8760108,  0.0415560,
         0.0556434, -0.2040259,  1.0572252
    );

    vec3 linear = M * xyz;

    // Apply gamma
    return vec3(
        linear_to_srgb(linear.r),
        linear_to_srgb(linear.g),
        linear_to_srgb(linear.b)
    );
}

// ============================================================================
// RGB <-> LAB (CIE L*a*b*)
// ============================================================================

// D65 reference white
const vec3 D65 = vec3(0.95047, 1.0, 1.08883);

float lab_f(float t) {
    const float delta = 6.0/29.0;
    return t > delta*delta*delta
        ? pow(t, 1.0/3.0)
        : t / (3.0*delta*delta) + 4.0/29.0;
}

float lab_f_inv(float t) {
    const float delta = 6.0/29.0;
    return t > delta
        ? t*t*t
        : 3.0*delta*delta * (t - 4.0/29.0);
}

vec3 rgb_to_lab(vec3 rgb) {
    vec3 xyz = rgb_to_xyz(rgb);

    float fx = lab_f(xyz.x / D65.x);
    float fy = lab_f(xyz.y / D65.y);
    float fz = lab_f(xyz.z / D65.z);

    float L = 116.0 * fy - 16.0;
    float a = 500.0 * (fx - fy);
    float b = 200.0 * (fy - fz);

    // Normalize to [0,1] range for storage
    return vec3(L / 100.0, (a + 128.0) / 255.0, (b + 128.0) / 255.0);
}

vec3 lab_to_rgb(vec3 lab) {
    // Denormalize from [0,1]
    float L = lab.x * 100.0;
    float a = lab.y * 255.0 - 128.0;
    float b = lab.z * 255.0 - 128.0;

    float fy = (L + 16.0) / 116.0;
    float fx = a / 500.0 + fy;
    float fz = fy - b / 200.0;

    vec3 xyz = vec3(
        D65.x * lab_f_inv(fx),
        D65.y * lab_f_inv(fy),
        D65.z * lab_f_inv(fz)
    );

    return xyz_to_rgb(xyz);
}

// ============================================================================
// RGB <-> YCbCr (BT.601)
// ============================================================================

vec3 rgb_to_ycbcr(vec3 rgb) {
    float Y  =  0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float Cb = -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b + 0.5;
    float Cr =  0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b + 0.5;
    return vec3(Y, Cb, Cr);
}

vec3 ycbcr_to_rgb(vec3 ycbcr) {
    float Y  = ycbcr.x;
    float Cb = ycbcr.y - 0.5;
    float Cr = ycbcr.z - 0.5;

    float r = Y + 1.402 * Cr;
    float g = Y - 0.344 * Cb - 0.714 * Cr;
    float b = Y + 1.772 * Cb;

    return clamp(vec3(r, g, b), 0.0, 1.0);
}

// ============================================================================
// RGB <-> YUV
// ============================================================================

vec3 rgb_to_yuv(vec3 rgb) {
    float Y =  0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float U = -0.147 * rgb.r - 0.289 * rgb.g + 0.436 * rgb.b + 0.5;
    float V =  0.615 * rgb.r - 0.515 * rgb.g - 0.100 * rgb.b + 0.5;
    return vec3(Y, U, V);
}

vec3 yuv_to_rgb(vec3 yuv) {
    float Y = yuv.x;
    float U = yuv.y - 0.5;
    float V = yuv.z - 0.5;

    float r = Y + 1.140 * V;
    float g = Y - 0.395 * U - 0.581 * V;
    float b = Y + 2.032 * U;

    return clamp(vec3(r, g, b), 0.0, 1.0);
}

// ============================================================================
// Conversion dispatcher
// ============================================================================

vec3 to_rgb(vec3 color, int space) {
    switch (space) {
        case CS_RGB:   return color;
        case CS_HSV:   return hsv_to_rgb(color);
        case CS_HSL:   return hsl_to_rgb(color);
        case CS_LAB:   return lab_to_rgb(color);
        case CS_XYZ:   return xyz_to_rgb(color);
        case CS_YCbCr: return ycbcr_to_rgb(color);
        case CS_YUV:   return yuv_to_rgb(color);
        default:       return color;
    }
}

vec3 from_rgb(vec3 rgb, int space) {
    switch (space) {
        case CS_RGB:   return rgb;
        case CS_HSV:   return rgb_to_hsv(rgb);
        case CS_HSL:   return rgb_to_hsl(rgb);
        case CS_LAB:   return rgb_to_lab(rgb);
        case CS_XYZ:   return rgb_to_xyz(rgb);
        case CS_YCbCr: return rgb_to_ycbcr(rgb);
        case CS_YUV:   return rgb_to_yuv(rgb);
        default:       return rgb;
    }
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    // Bounds check
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    // Load pixel
    vec4 color = imageLoad(input_image, pixel);

    // Convert: source -> RGB -> target
    vec3 rgb = to_rgb(color.rgb, from_space);
    vec3 result = from_rgb(rgb, to_space);

    imageStore(output_image, pixel, vec4(result, color.a));
}
