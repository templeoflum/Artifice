"""
Color space conversion functions.

Ported from GLIC's colorspaces.pde - all 16 color spaces.
All functions operate on float32 arrays with values in [0, 1] range for RGB.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Constants from GLIC
D65X = 0.950456
D65Y = 1.0
D65Z = 1.088754
CIE_EPSILON = 216.0 / 24389.0
CIE_K = 24389.0 / 27.0
CIE_K2_EPSILON = CIE_K * CIE_EPSILON
D65FX_4 = 4.0 * D65X / (D65X + 15.0 * D65Y + 3.0 * D65Z)
D65FY_9 = 9.0 * D65Y / (D65X + 15.0 * D65Y + 3.0 * D65Z)
RANGE_X = 100.0 * (0.4124 + 0.3576 + 0.1805)
RANGE_Y = 100.0
RANGE_Z = 100.0 * (0.0193 + 0.1192 + 0.9505)
M_EPSILON = 1.0e-10
ONE_THIRD = 1.0 / 3.0
ONE_116 = 1.0 / 116.0
CORR_RATIO = 1.0 / 2.4

# YUV constants
U_MAX = 0.436
V_MAX = 0.615


def _clip(arr: NDArray) -> NDArray:
    """Clip array to [0, 1] range."""
    return np.clip(arr, 0.0, 1.0)


# =============================================================================
# RGB <-> Greyscale
# =============================================================================

def rgb_to_grey(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to Greyscale (luminance)."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.stack([luma, luma, luma], axis=0)


def grey_to_rgb(grey: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert Greyscale back to RGB (identity for grey)."""
    return grey.copy()


# =============================================================================
# RGB <-> CMY (simple inversion)
# =============================================================================

def rgb_to_cmy(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to CMY."""
    return 1.0 - rgb


def cmy_to_rgb(cmy: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert CMY to RGB."""
    return 1.0 - cmy


# =============================================================================
# RGB <-> HSB/HSV
# =============================================================================

def rgb_to_hsb(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to HSB (Hue, Saturation, Brightness)."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    _min = np.minimum(np.minimum(r, g), b)
    _max = np.maximum(np.maximum(r, g), b)
    delta = _max - _min

    # Brightness
    brightness = _max

    # Saturation
    saturation = np.where(_max > 0, delta / (_max + M_EPSILON), 0.0)

    # Hue
    hue = np.zeros_like(r)

    # Where delta > 0
    mask = delta > 0

    # R is max
    r_max = mask & (r == _max)
    hue = np.where(r_max, (g - b) / (delta + M_EPSILON), hue)

    # G is max
    g_max = mask & (g == _max)
    hue = np.where(g_max, 2.0 + (b - r) / (delta + M_EPSILON), hue)

    # B is max
    b_max = mask & (b == _max)
    hue = np.where(b_max, 4.0 + (r - g) / (delta + M_EPSILON), hue)

    hue = hue / 6.0
    hue = np.where(hue < 0, hue + 1.0, hue)

    return np.stack([hue, saturation, brightness], axis=0).astype(np.float32)


def hsb_to_rgb(hsb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert HSB to RGB."""
    h, s, b = hsb[0], hsb[1], hsb[2]

    # Achromatic case
    achromatic = s == 0

    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)

    p = b * (1.0 - s)
    q = b * (1.0 - s * f)
    t = b * (1.0 - s * (1.0 - f))

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    blue = np.zeros_like(h)

    # i == 0
    mask = i == 0
    r = np.where(mask, b, r)
    g = np.where(mask, t, g)
    blue = np.where(mask, p, blue)

    # i == 1
    mask = i == 1
    r = np.where(mask, q, r)
    g = np.where(mask, b, g)
    blue = np.where(mask, p, blue)

    # i == 2
    mask = i == 2
    r = np.where(mask, p, r)
    g = np.where(mask, b, g)
    blue = np.where(mask, t, blue)

    # i == 3
    mask = i == 3
    r = np.where(mask, p, r)
    g = np.where(mask, q, g)
    blue = np.where(mask, b, blue)

    # i == 4
    mask = i == 4
    r = np.where(mask, t, r)
    g = np.where(mask, p, g)
    blue = np.where(mask, b, blue)

    # i == 5
    mask = i == 5
    r = np.where(mask, b, r)
    g = np.where(mask, p, g)
    blue = np.where(mask, q, blue)

    # Achromatic override
    r = np.where(achromatic, b, r)
    g = np.where(achromatic, b, g)
    blue = np.where(achromatic, b, blue)

    return _clip(np.stack([r, g, blue], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> HWB (Hue, Whiteness, Blackness)
# =============================================================================

def rgb_to_hwb(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to HWB."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    w = np.minimum(np.minimum(r, g), b)  # Whiteness
    v = np.maximum(np.maximum(r, g), b)  # Value (inverse of blackness)

    # Hue calculation
    delta = v - w
    hue = np.zeros_like(r)

    # Non-grey pixels
    mask = delta > 0

    # R is min
    r_min = mask & (r == w)
    f = np.where(r_min, g - b, 0.0)
    p = np.where(r_min, 3.0, 0.0)

    # G is min
    g_min = mask & (g == w)
    f = np.where(g_min, b - r, f)
    p = np.where(g_min, 5.0, p)

    # B is min
    b_min = mask & (b == w)
    f = np.where(b_min, r - g, f)
    p = np.where(b_min, 1.0, p)

    hue = np.where(mask, (p - f / (delta + M_EPSILON)) / 6.0, 1.0)  # 1.0 for grey

    blackness = 1.0 - v

    return np.stack([hue, w, blackness], axis=0).astype(np.float32)


def hwb_to_rgb(hwb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert HWB to RGB."""
    h, w, b = hwb[0], hwb[1], hwb[2]

    v = 1.0 - b  # Value from blackness

    # Grey case (h == 1.0 in our encoding)
    grey = h >= 0.999

    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    f = np.where((i & 1) != 0, 1.0 - f, f)
    n = w + f * (v - w)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    blue = np.zeros_like(h)

    # i == 0
    mask = i == 0
    r = np.where(mask, v, r)
    g = np.where(mask, n, g)
    blue = np.where(mask, w, blue)

    # i == 1
    mask = i == 1
    r = np.where(mask, n, r)
    g = np.where(mask, v, g)
    blue = np.where(mask, w, blue)

    # i == 2
    mask = i == 2
    r = np.where(mask, w, r)
    g = np.where(mask, v, g)
    blue = np.where(mask, n, blue)

    # i == 3
    mask = i == 3
    r = np.where(mask, w, r)
    g = np.where(mask, n, g)
    blue = np.where(mask, v, blue)

    # i == 4
    mask = i == 4
    r = np.where(mask, n, r)
    g = np.where(mask, w, g)
    blue = np.where(mask, v, blue)

    # i == 5
    mask = i == 5
    r = np.where(mask, v, r)
    g = np.where(mask, w, g)
    blue = np.where(mask, n, blue)

    # Grey override
    r = np.where(grey, v, r)
    g = np.where(grey, v, g)
    blue = np.where(grey, v, blue)

    return _clip(np.stack([r, g, blue], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> YUV
# =============================================================================

def rgb_to_yuv(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to YUV."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (-0.14713 * r - 0.28886 * g + 0.436 * b)
    v = (0.615 * r - 0.51499 * g - 0.10001 * b)

    # Map U and V to [0, 1] range
    u = (u + U_MAX) / (2 * U_MAX)
    v = (v + V_MAX) / (2 * V_MAX)

    return np.stack([y, u, v], axis=0).astype(np.float32)


def yuv_to_rgb(yuv: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert YUV to RGB."""
    y, u, v = yuv[0], yuv[1], yuv[2]

    # Unmap U and V from [0, 1]
    u = u * (2 * U_MAX) - U_MAX
    v = v * (2 * V_MAX) - V_MAX

    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> YCbCr
# =============================================================================

def rgb_to_ycbcr(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to YCbCr."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    y = 0.298839 * r + 0.586811 * g + 0.114350 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    return np.stack([y, cb, cr], axis=0).astype(np.float32)


def ycbcr_to_rgb(ycbcr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert YCbCr to RGB."""
    y, cb, cr = ycbcr[0], ycbcr[1], ycbcr[2]

    cb = cb - 0.5
    cr = cr - 0.5

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> YPbPr
# =============================================================================

def rgb_to_ypbpr(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to YPbPr."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    pb = b - y
    pr = r - y

    # Map to [0, 1] with wrap-around handling
    pb = np.mod(pb + 1.0, 1.0)
    pr = np.mod(pr + 1.0, 1.0)

    return np.stack([y, pb, pr], axis=0).astype(np.float32)


def ypbpr_to_rgb(ypbpr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert YPbPr to RGB."""
    y, pb, pr = ypbpr[0], ypbpr[1], ypbpr[2]

    # Unmap with wrap-around
    b = pb + y
    r = pr + y
    b = np.where(b > 1.0, b - 1.0, b)
    r = np.where(r > 1.0, r - 1.0, r)

    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> YDbDr
# =============================================================================

def rgb_to_ydbdr(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to YDbDr."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    db = 0.5 + (-0.450 * r - 0.883 * g + 1.333 * b) / 2.666
    dr = 0.5 + (-1.333 * r + 1.116 * g + 0.217 * b) / 2.666

    return np.stack([y, db, dr], axis=0).astype(np.float32)


def ydbdr_to_rgb(ydbdr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert YDbDr to RGB."""
    y, db, dr = ydbdr[0], ydbdr[1], ydbdr[2]

    db = (db - 0.5) * 2.666
    dr = (dr - 0.5) * 2.666

    r = y + 9.2303716147657e-05 * db - 0.52591263066186533 * dr
    g = y - 0.12913289889050927 * db + 0.26789932820759876 * dr
    b = y + 0.66467905997895482 * db - 7.9202543533108e-05 * dr

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> R-GGB-G (Green difference)
# =============================================================================

def rgb_to_rggbg(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to R-G, G, B-G."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    rg = np.mod(r - g + 1.0, 1.0)
    bg = np.mod(b - g + 1.0, 1.0)

    return np.stack([rg, g, bg], axis=0).astype(np.float32)


def rggbg_to_rgb(rggbg: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert R-G, G, B-G to RGB."""
    rg, g, bg = rggbg[0], rggbg[1], rggbg[2]

    r = rg + g
    b = bg + g
    r = np.where(r > 1.0, r - 1.0, r)
    b = np.where(b > 1.0, b - 1.0, b)

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> OHTA
# =============================================================================

def rgb_to_ohta(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to OHTA color space."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    i1 = 0.33333 * r + 0.33334 * g + 0.33333 * b
    i2 = 0.5 * (r - b) + 0.5  # Map to [0, 1]
    i3 = -0.25 * r + 0.5 * g - 0.25 * b + 0.5  # Map to [0, 1]

    return np.stack([i1, i2, i3], axis=0).astype(np.float32)


def ohta_to_rgb(ohta: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert OHTA to RGB."""
    i1, i2, i3 = ohta[0], ohta[1], ohta[2]

    # Unmap from [0, 1]
    i2 = (i2 - 0.5) * 2.0  # Back to [-1, 1] * 0.5
    i3 = (i3 - 0.5) * 2.0

    r = i1 + 1.0 * i2 - 0.66668 * i3
    g = i1 + 1.33333 * i3
    b = i1 - 1.0 * i2 - 0.66668 * i3

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> XYZ (CIE 1931, D65 illuminant)
# =============================================================================

def _srgb_to_linear(c: NDArray) -> NDArray:
    """Convert sRGB to linear RGB (gamma correction)."""
    return np.where(c > 0.04045, np.power((c + 0.055) / 1.055, 2.4), c / 12.92)


def _linear_to_srgb(c: NDArray) -> NDArray:
    """Convert linear RGB to sRGB (inverse gamma)."""
    return np.where(c > 0.0031308, 1.055 * np.power(c, CORR_RATIO) - 0.055, 12.92 * c)


def rgb_to_xyz(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to XYZ."""
    r = _srgb_to_linear(rgb[0])
    g = _srgb_to_linear(rgb[1])
    b = _srgb_to_linear(rgb[2])

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # Normalize to [0, 1] range
    x = x / RANGE_X * 100.0
    y = y / RANGE_Y * 100.0
    z = z / RANGE_Z * 100.0

    return np.stack([x, y, z], axis=0).astype(np.float32)


def xyz_to_rgb(xyz: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert XYZ to RGB."""
    # Denormalize from [0, 1]
    x = xyz[0] * RANGE_X / 100.0
    y = xyz[1] * RANGE_Y / 100.0
    z = xyz[2] * RANGE_Z / 100.0

    r = _linear_to_srgb(x * 3.2406 + y * -1.5372 + z * -0.4986)
    g = _linear_to_srgb(x * -0.9689 + y * 1.8758 + z * 0.0415)
    b = _linear_to_srgb(x * 0.0557 + y * -0.2040 + z * 1.0570)

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> LAB (CIE L*a*b*)
# =============================================================================

def rgb_to_lab(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to LAB."""
    # First convert to XYZ (internal, not normalized to [0,1])
    r = _srgb_to_linear(rgb[0])
    g = _srgb_to_linear(rgb[1])
    b = _srgb_to_linear(rgb[2])

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / D65X
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / D65Y
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / D65Z

    # Apply f function
    fx = np.where(x > CIE_EPSILON, np.power(x, ONE_THIRD), (CIE_K * x + 16.0) * ONE_116)
    fy = np.where(y > CIE_EPSILON, np.power(y, ONE_THIRD), (CIE_K * y + 16.0) * ONE_116)
    fz = np.where(z > CIE_EPSILON, np.power(z, ONE_THIRD), (CIE_K * z + 16.0) * ONE_116)

    L = (116.0 * fy - 16.0) / 100.0  # [0, 1]
    a = (fx - fy) + 0.5  # Map to [0, 1]
    b_out = (fy - fz) + 0.5  # Map to [0, 1]

    return np.stack([L, a, b_out], axis=0).astype(np.float32)


def lab_to_rgb(lab: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert LAB to RGB."""
    L = lab[0] * 100.0
    a = lab[1] - 0.5
    b = lab[2] - 0.5

    fy = (L + 16.0) * ONE_116
    fx = fy + a
    fz = fy - b

    # Inverse f function
    fx3 = fx * fx * fx
    x = np.where(fx3 > CIE_EPSILON, fx3, (116.0 * fx - 16.0) / CIE_K)

    fy3 = fy * fy * fy
    y = np.where(fy3 > CIE_EPSILON, fy3, L / CIE_K)

    fz3 = fz * fz * fz
    z = np.where(fz3 > CIE_EPSILON, fz3, (116.0 * fz - 16.0) / CIE_K)

    # Scale by reference white
    x = x * D65X
    y = y * D65Y
    z = z * D65Z

    # XYZ to RGB
    r = _linear_to_srgb(x * 3.2406 + y * -1.5372 + z * -0.4986)
    g = _linear_to_srgb(x * -0.9689 + y * 1.8758 + z * 0.0415)
    b_out = _linear_to_srgb(x * 0.0557 + y * -0.2040 + z * 1.0570)

    return _clip(np.stack([r, g, b_out], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> LUV (CIE L*u*v*)
# =============================================================================

def _perceptible_reciprocal(x: NDArray) -> NDArray:
    """Safe reciprocal with epsilon handling."""
    sgn = np.sign(x)
    abs_x = np.abs(x)
    return np.where(abs_x >= M_EPSILON, 1.0 / x, sgn / M_EPSILON)


def rgb_to_luv(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to LUV."""
    # First convert to XYZ
    r = _srgb_to_linear(rgb[0])
    g = _srgb_to_linear(rgb[1])
    b = _srgb_to_linear(rgb[2])

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # Compute L
    L = np.where(y > CIE_EPSILON, 116.0 * np.power(y, ONE_THIRD) - 16.0, CIE_K * y)

    # Compute u' and v'
    alpha = _perceptible_reciprocal(x + 15.0 * y + 3.0 * z)
    L13 = 13.0 * L

    u = L13 * (4.0 * alpha * x - D65FX_4)
    v = L13 * (9.0 * alpha * y - D65FY_9)

    # Normalize to [0, 1]
    L = L / 100.0
    u = (u + 134.0) / 354.0
    v = (v + 140.0) / 262.0

    return np.stack([L, u, v], axis=0).astype(np.float32)


def luv_to_rgb(luv: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert LUV to RGB."""
    L = luv[0] * 100.0
    u = luv[1] * 354.0 - 134.0
    v = luv[2] * 262.0 - 140.0

    # Compute Y
    Y = np.where(L > CIE_K2_EPSILON, np.power((L + 16.0) * ONE_116, 3.0), L / CIE_K)

    L13 = 13.0 * L + M_EPSILON
    L52 = 52.0 * L
    Y5 = 5.0 * Y

    L13u = L52 / (u + L13 * D65FX_4 + M_EPSILON)
    X = (Y * ((39.0 * L / (v + L13 * D65FY_9 + M_EPSILON)) - 5.0) + Y5) / (((L13u - 1.0) / 3.0) + ONE_THIRD + M_EPSILON)
    Z = X * ((L13u - 1.0) / 3.0) - Y5

    # XYZ to RGB
    r = _linear_to_srgb(X * 3.2406 + Y * -1.5372 + Z * -0.4986)
    g = _linear_to_srgb(X * -0.9689 + Y * 1.8758 + Z * 0.0415)
    b = _linear_to_srgb(X * 0.0557 + Y * -0.2040 + Z * 1.0570)

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> HCL (Hue, Chroma, Luminance)
# =============================================================================

def rgb_to_hcl(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to HCL."""
    r, g, b = rgb[0], rgb[1], rgb[2]

    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    chroma = max_val - min_val

    # Hue
    hue = np.zeros_like(r)
    mask = chroma > 0

    r_max = mask & (r == max_val)
    hue = np.where(r_max, np.mod((g - b) / (chroma + M_EPSILON) + 6.0, 6.0), hue)

    g_max = mask & (g == max_val)
    hue = np.where(g_max, (b - r) / (chroma + M_EPSILON) + 2.0, hue)

    b_max = mask & (b == max_val)
    hue = np.where(b_max, (r - g) / (chroma + M_EPSILON) + 4.0, hue)

    hue = hue / 6.0

    # Luminance (same as YCbCr luma)
    luma = 0.298839 * r + 0.586811 * g + 0.114350 * b

    return np.stack([hue, chroma, luma], axis=0).astype(np.float32)


def hcl_to_rgb(hcl: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert HCL to RGB."""
    h, c, l = hcl[0], hcl[1], hcl[2]

    h6 = h * 6.0
    x = c * (1.0 - np.abs(np.mod(h6, 2.0) - 1.0))

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Sector 0
    mask = (h6 >= 0) & (h6 < 1)
    r = np.where(mask, c, r)
    g = np.where(mask, x, g)

    # Sector 1
    mask = (h6 >= 1) & (h6 < 2)
    r = np.where(mask, x, r)
    g = np.where(mask, c, g)

    # Sector 2
    mask = (h6 >= 2) & (h6 < 3)
    g = np.where(mask, c, g)
    b = np.where(mask, x, b)

    # Sector 3
    mask = (h6 >= 3) & (h6 < 4)
    g = np.where(mask, x, g)
    b = np.where(mask, c, b)

    # Sector 4
    mask = (h6 >= 4) & (h6 < 5)
    r = np.where(mask, x, r)
    b = np.where(mask, c, b)

    # Sector 5
    mask = (h6 >= 5) & (h6 < 6)
    r = np.where(mask, c, r)
    b = np.where(mask, x, b)

    # Add match value
    m = l - (0.298839 * r + 0.586811 * g + 0.114350 * b)
    r = r + m
    g = g + m
    b = b + m

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# RGB <-> YXY (CIE xyY)
# =============================================================================

def rgb_to_yxy(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to Yxy (CIE xyY)."""
    # First convert to XYZ
    r = _srgb_to_linear(rgb[0])
    g = _srgb_to_linear(rgb[1])
    b = _srgb_to_linear(rgb[2])

    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505

    total = X + Y + Z + M_EPSILON
    x = np.where(X > 0, X / total, 0.0)
    y = np.where(Y > 0, Y / total, 0.0)

    # Normalize Y to [0, 1]
    Y_norm = Y / RANGE_Y * 100.0

    return np.stack([Y_norm, x, y], axis=0).astype(np.float32)


def yxy_to_rgb(yxy: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert Yxy to RGB."""
    Y = yxy[0] * RANGE_Y / 100.0
    x = yxy[1]
    y = yxy[2]

    div_y = Y / (y + M_EPSILON)

    X = x * div_y
    Z = (1.0 - x - y) * div_y

    # XYZ to RGB
    r = _linear_to_srgb(X * 3.2406 + Y * -1.5372 + Z * -0.4986)
    g = _linear_to_srgb(X * -0.9689 + Y * 1.8758 + Z * 0.0415)
    b = _linear_to_srgb(X * 0.0557 + Y * -0.2040 + Z * 1.0570)

    return _clip(np.stack([r, g, b], axis=0)).astype(np.float32)


# =============================================================================
# Conversion dispatch
# =============================================================================

# All conversion functions: (to_rgb, from_rgb)
COLORSPACE_CONVERTERS = {
    "RGB": (lambda x: x, lambda x: x),
    "GREY": (grey_to_rgb, rgb_to_grey),
    "GRAYSCALE": (grey_to_rgb, rgb_to_grey),
    "CMY": (cmy_to_rgb, rgb_to_cmy),
    "HSB": (hsb_to_rgb, rgb_to_hsb),
    "HSV": (hsb_to_rgb, rgb_to_hsb),
    "HWB": (hwb_to_rgb, rgb_to_hwb),
    "YUV": (yuv_to_rgb, rgb_to_yuv),
    "YCbCr": (ycbcr_to_rgb, rgb_to_ycbcr),
    "YCBCR": (ycbcr_to_rgb, rgb_to_ycbcr),
    "YPbPr": (ypbpr_to_rgb, rgb_to_ypbpr),
    "YPBPR": (ypbpr_to_rgb, rgb_to_ypbpr),
    "YDbDr": (ydbdr_to_rgb, rgb_to_ydbdr),
    "YDBDR": (ydbdr_to_rgb, rgb_to_ydbdr),
    "R-GGB-G": (rggbg_to_rgb, rgb_to_rggbg),
    "RGGBG": (rggbg_to_rgb, rgb_to_rggbg),
    "OHTA": (ohta_to_rgb, rgb_to_ohta),
    "XYZ": (xyz_to_rgb, rgb_to_xyz),
    "LAB": (lab_to_rgb, rgb_to_lab),
    "LUV": (luv_to_rgb, rgb_to_luv),
    "HCL": (hcl_to_rgb, rgb_to_hcl),
    "YXY": (yxy_to_rgb, rgb_to_yxy),
}


def convert_colorspace(
    data: NDArray[np.float32],
    from_space: str,
    to_space: str,
) -> NDArray[np.float32]:
    """
    Convert image data between color spaces.

    Args:
        data: Image data in CHW format, float32 [0, 1]
        from_space: Source color space name
        to_space: Target color space name

    Returns:
        Converted image data
    """
    from_space = from_space.upper()
    to_space = to_space.upper()

    if from_space == to_space:
        return data.copy()

    # Get converters
    if from_space not in COLORSPACE_CONVERTERS:
        raise ValueError(f"Unknown source colorspace: {from_space}")
    if to_space not in COLORSPACE_CONVERTERS:
        raise ValueError(f"Unknown target colorspace: {to_space}")

    to_rgb_func, _ = COLORSPACE_CONVERTERS[from_space]
    _, from_rgb_func = COLORSPACE_CONVERTERS[to_space]

    # Convert via RGB
    rgb = to_rgb_func(data)
    result = from_rgb_func(rgb)

    return result


def list_colorspaces() -> list[str]:
    """Get list of supported color spaces."""
    # Return unique names (not aliases)
    unique = {"RGB", "GREY", "CMY", "HSB", "HWB", "YUV", "YCbCr", "YPbPr",
              "YDbDr", "R-GGB-G", "OHTA", "XYZ", "LAB", "LUV", "HCL", "YXY"}
    return sorted(unique)
