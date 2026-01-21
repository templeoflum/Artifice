"""
GLIC prediction algorithms.

All 16 predictors from GLIC plus meta-predictors (SAD, BSAD, RANDOM).
Each predictor generates a predicted block from neighboring pixels.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from artifice.core.data_types import ImageBuffer, Segment, SegmentList


class PredictorType(IntEnum):
    """Predictor type identifiers matching GLIC."""

    SAD = -1  # Best predictor (min SAD)
    BSAD = -2  # Worst predictor (max SAD) - for glitch effects
    RANDOM = -3  # Random predictor

    NONE = 0  # No prediction (zeros)
    CORNER = 1  # Top-left corner pixel
    H = 2  # Horizontal (left edge)
    V = 3  # Vertical (top edge)
    DC = 4  # DC mean of edges
    DCMEDIAN = 5  # Median of DC and edges
    MEDIAN = 6  # Median of corner and edges
    AVG = 7  # Average of top and left
    TRUEMOTION = 8  # TrueMotion video codec style
    PAETH = 9  # PNG Paeth filter
    LDIAG = 10  # Linear diagonal
    HV = 11  # Position-based H or V
    JPEGLS = 12  # JPEG-LS predictor
    DIFF = 13  # Second-order difference
    REF = 14  # Reference region (requires search)
    ANGLE = 15  # Angle-based reference


# Human-readable names
PREDICTOR_NAMES = {
    PredictorType.SAD: "SAD (Best)",
    PredictorType.BSAD: "BSAD (Worst)",
    PredictorType.RANDOM: "Random",
    PredictorType.NONE: "None",
    PredictorType.CORNER: "Corner",
    PredictorType.H: "Horizontal",
    PredictorType.V: "Vertical",
    PredictorType.DC: "DC Mean",
    PredictorType.DCMEDIAN: "DC Median",
    PredictorType.MEDIAN: "Median",
    PredictorType.AVG: "Average",
    PredictorType.TRUEMOTION: "TrueMotion",
    PredictorType.PAETH: "Paeth",
    PredictorType.LDIAG: "Linear Diagonal",
    PredictorType.HV: "H/V Position",
    PredictorType.JPEGLS: "JPEG-LS",
    PredictorType.DIFF: "Difference",
    PredictorType.REF: "Reference",
    PredictorType.ANGLE: "Angle",
}


def _get_pixel(data: NDArray, x: int, y: int, border: float = 0.0) -> float:
    """Safe pixel access with border handling."""
    h, w = data.shape
    if 0 <= y < h and 0 <= x < w:
        return float(data[y, x])
    return border


def _median3(a: float, b: float, c: float) -> float:
    """Median of three values."""
    return max(min(a, b), min(max(a, b), c))


# =============================================================================
# Individual Predictors
# =============================================================================

def pred_none(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """No prediction - returns zeros."""
    return np.zeros((seg.size, seg.size), dtype=np.float32)


def pred_corner(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Predict from top-left corner pixel."""
    val = _get_pixel(data, seg.x - 1, seg.y - 1, border)
    return np.full((seg.size, seg.size), val, dtype=np.float32)


def pred_h(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Horizontal prediction - use left edge."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for y in range(seg.size):
        val = _get_pixel(data, seg.x - 1, seg.y + y, border)
        result[y, :] = val
    return result


def pred_v(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Vertical prediction - use top edge."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for x in range(seg.size):
        val = _get_pixel(data, seg.x + x, seg.y - 1, border)
        result[:, x] = val
    return result


def pred_dc(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """DC mean prediction - average of edges."""
    total = 0.0
    count = 0

    # Left edge
    for y in range(seg.size):
        total += _get_pixel(data, seg.x - 1, seg.y + y, border)
        count += 1

    # Top edge
    for x in range(seg.size):
        total += _get_pixel(data, seg.x + x, seg.y - 1, border)
        count += 1

    # Corner
    total += _get_pixel(data, seg.x - 1, seg.y - 1, border)
    count += 1

    dc_val = total / count if count > 0 else border
    return np.full((seg.size, seg.size), dc_val, dtype=np.float32)


def pred_dcmedian(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """DC median prediction - median of DC and edge pixels."""
    # First calculate DC
    dc_val = float(pred_dc(data, seg, border)[0, 0])

    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for x in range(seg.size):
        v1 = _get_pixel(data, seg.x + x, seg.y - 1, border)
        for y in range(seg.size):
            v2 = _get_pixel(data, seg.x - 1, seg.y + y, border)
            result[y, x] = _median3(dc_val, v1, v2)

    return result


def pred_median(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Median prediction - median of corner and edge pixels."""
    corner = _get_pixel(data, seg.x - 1, seg.y - 1, border)

    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for x in range(seg.size):
        v1 = _get_pixel(data, seg.x + x, seg.y - 1, border)
        for y in range(seg.size):
            v2 = _get_pixel(data, seg.x - 1, seg.y + y, border)
            result[y, x] = _median3(corner, v1, v2)

    return result


def pred_avg(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Average prediction - average of top and left."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for x in range(seg.size):
        v1 = _get_pixel(data, seg.x + x, seg.y - 1, border)
        for y in range(seg.size):
            v2 = _get_pixel(data, seg.x - 1, seg.y + y, border)
            result[y, x] = (v1 + v2) / 2.0

    return result


def pred_truemotion(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """TrueMotion prediction - v1 + v2 - corner."""
    corner = _get_pixel(data, seg.x - 1, seg.y - 1, border)

    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for x in range(seg.size):
        v1 = _get_pixel(data, seg.x + x, seg.y - 1, border)
        for y in range(seg.size):
            v2 = _get_pixel(data, seg.x - 1, seg.y + y, border)
            result[y, x] = np.clip(v1 + v2 - corner, 0.0, 1.0)

    return result


def pred_paeth(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Paeth prediction (PNG filter)."""
    corner = _get_pixel(data, seg.x - 1, seg.y - 1, border)

    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    for x in range(seg.size):
        v1 = _get_pixel(data, seg.x + x, seg.y - 1, border)  # top
        for y in range(seg.size):
            v2 = _get_pixel(data, seg.x - 1, seg.y + y, border)  # left

            pp = v1 + v2 - corner
            pa = abs(pp - v2)  # Distance to left
            pb = abs(pp - v1)  # Distance to top
            pc = abs(pp - corner)  # Distance to corner

            if pa <= pb and pa <= pc:
                v = v2
            elif pb <= pc:
                v = v1
            else:
                v = corner

            result[y, x] = np.clip(v, 0.0, 1.0)

    return result


def pred_ldiag(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Linear diagonal interpolation."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)
    size = seg.size

    for x in range(size):
        for y in range(size):
            ss = x + y
            xx_idx = min(ss + 1, size - 1)
            yy_idx = min(ss, size - 1)

            xx = _get_pixel(data, seg.x + xx_idx, seg.y - 1, border)
            yy = _get_pixel(data, seg.x - 1, seg.y + yy_idx, border)

            weight_total = x + y + 2
            result[y, x] = ((x + 1) * xx + (y + 1) * yy) / weight_total

    return result


def pred_hv(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """H/V prediction based on position."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)

    for x in range(seg.size):
        v_top = _get_pixel(data, seg.x + x, seg.y - 1, border)
        for y in range(seg.size):
            v_left = _get_pixel(data, seg.x - 1, seg.y + y, border)

            if x > y:
                result[y, x] = v_top
            elif y > x:
                result[y, x] = v_left
            else:
                result[y, x] = (v_top + v_left) / 2.0

    return result


def pred_jpegls(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """JPEG-LS predictor."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)

    for x in range(seg.size):
        c = _get_pixel(data, seg.x + x - 1, seg.y - 1, border)  # top-left
        a = _get_pixel(data, seg.x + x, seg.y - 1, border)  # top

        for y in range(seg.size):
            b = _get_pixel(data, seg.x - 1, seg.y + y, border)  # left

            if c >= max(a, b):
                v = min(a, b)
            elif c <= min(a, b):
                v = max(a, b)
            else:
                v = a + b - c

            result[y, x] = v

    return result


def pred_diff(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Second-order difference prediction."""
    result = np.zeros((seg.size, seg.size), dtype=np.float32)

    for x in range(seg.size):
        x1 = _get_pixel(data, seg.x + x, seg.y - 1, border)
        x2 = _get_pixel(data, seg.x + x, seg.y - 2, border)

        for y in range(seg.size):
            y1 = _get_pixel(data, seg.x - 1, seg.y + y, border)
            y2 = _get_pixel(data, seg.x - 2, seg.y + y, border)

            v = (2 * y2 - y1 + 2 * x2 - x1) / 2.0
            result[y, x] = np.clip(v, 0.0, 1.0)

    return result


def pred_ref(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
    num_candidates: int = 45,
) -> NDArray[np.float32]:
    """Reference region prediction - find best matching region."""
    h, w = data.shape
    size = seg.size

    best_sad = float("inf")
    best_result = None
    best_ref = (0, 0)

    rng = np.random.default_rng()

    for _ in range(num_candidates):
        # Generate random reference position (before current segment)
        ref_x = rng.integers(-size, seg.x + 1)

        if ref_x < seg.x - size:
            ref_y = rng.integers(-size, seg.y + 1)
        else:
            ref_y = rng.integers(-size, seg.y - size + 1)

        # Get reference region
        result = np.zeros((size, size), dtype=np.float32)
        for x in range(size):
            for y in range(size):
                result[y, x] = _get_pixel(data, ref_x + x, ref_y + y, border)

        # Calculate SAD
        actual = np.zeros((size, size), dtype=np.float32)
        for x in range(size):
            for y in range(size):
                actual[y, x] = _get_pixel(data, seg.x + x, seg.y + y, border)

        sad = np.sum(np.abs(actual - result))

        if sad < best_sad:
            best_sad = sad
            best_result = result
            best_ref = (ref_x, ref_y)

    if best_result is None:
        return np.full((size, size), border, dtype=np.float32)

    # Store reference in segment
    seg.ref_x = best_ref[0]
    seg.ref_y = best_ref[1]

    return best_result


def pred_angle(
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """Angle-based prediction - find best angle."""
    size = seg.size
    step_a = 1.0 / min(16, size)

    best_sad = float("inf")
    best_result = None
    best_angle = 0.0
    best_mode = 0

    for mode in range(3):
        a = 0.0
        while a < 1.0:
            result = np.zeros((size, size), dtype=np.float32)

            for x in range(size):
                for y in range(size):
                    ref_x, ref_y = _get_angle_ref(mode, x, y, a, size)
                    ref_x = min(ref_x, size - 1)
                    result[y, x] = _get_pixel(
                        data, int(ref_x) + seg.x, int(ref_y) + seg.y, border
                    )

            # Calculate SAD
            actual = np.zeros((size, size), dtype=np.float32)
            for px in range(size):
                for py in range(size):
                    actual[py, px] = _get_pixel(data, seg.x + px, seg.y + py, border)

            sad = np.sum(np.abs(actual - result))

            if sad < best_sad:
                best_sad = sad
                best_result = result
                best_angle = a
                best_mode = mode

            a += step_a

    if best_result is None:
        return np.full((size, size), border, dtype=np.float32)

    # Store in segment
    seg.angle = best_angle
    seg.ref_type = best_mode

    return best_result


def _get_angle_ref(mode: int, x: int, y: int, a: float, w: int) -> tuple[float, float]:
    """Get reference coordinates for angle prediction."""
    if mode == 0:
        v = (w - y - 1) + x * a
        xx = (v - w) / (a + 1e-10)
        yy = w - 1 - a - v
    elif mode == 1:
        v = (w - x - 1) + y * a
        yy = (v - w) / (a + 1e-10)
        xx = w - 1 - a - v
    else:  # mode == 2
        v = x + y * a
        yy = -1.0
        xx = v + a

    if xx > yy:
        return (round(xx), -1)
    else:
        return (-1, round(yy))


# =============================================================================
# Predictor Registry
# =============================================================================

PREDICTORS: dict[int, Callable] = {
    PredictorType.NONE: pred_none,
    PredictorType.CORNER: pred_corner,
    PredictorType.H: pred_h,
    PredictorType.V: pred_v,
    PredictorType.DC: pred_dc,
    PredictorType.DCMEDIAN: pred_dcmedian,
    PredictorType.MEDIAN: pred_median,
    PredictorType.AVG: pred_avg,
    PredictorType.TRUEMOTION: pred_truemotion,
    PredictorType.PAETH: pred_paeth,
    PredictorType.LDIAG: pred_ldiag,
    PredictorType.HV: pred_hv,
    PredictorType.JPEGLS: pred_jpegls,
    PredictorType.DIFF: pred_diff,
    PredictorType.REF: pred_ref,
    PredictorType.ANGLE: pred_angle,
}

MAX_PRED = 16  # Number of basic predictors

# List of core (non-meta) predictors for testing
CORE_PREDICTORS = [
    PredictorType.NONE,
    PredictorType.CORNER,
    PredictorType.H,
    PredictorType.V,
    PredictorType.DC,
    PredictorType.DCMEDIAN,
    PredictorType.MEDIAN,
    PredictorType.AVG,
    PredictorType.TRUEMOTION,
    PredictorType.PAETH,
    PredictorType.LDIAG,
    PredictorType.HV,
    PredictorType.JPEGLS,
    PredictorType.DIFF,
    PredictorType.REF,
    PredictorType.ANGLE,
]


def _calc_sad(
    predicted: NDArray[np.float32],
    data: NDArray[np.float32],
    seg: Segment,
    border: float = 0.0,
) -> float:
    """Calculate Sum of Absolute Differences."""
    total = 0.0
    for x in range(seg.size):
        for y in range(seg.size):
            actual = _get_pixel(data, seg.x + x, seg.y + y, border)
            total += abs(actual - predicted[y, x])
    return total


def predict_segment(
    data: NDArray[np.float32],
    seg: Segment,
    predictor: int | PredictorType,
    border: float = 0.0,
) -> NDArray[np.float32]:
    """
    Generate prediction for a segment.

    Args:
        data: 2D array (H, W) - single channel
        seg: Segment to predict
        predictor: Predictor type
        border: Border value for out-of-bounds access

    Returns:
        2D array (seg.size, seg.size) of predicted values
    """
    predictor = int(predictor)

    # Meta-predictors
    if predictor == PredictorType.RANDOM:
        predictor = np.random.randint(0, MAX_PRED)
        seg.pred_type = predictor
        return predict_segment(data, seg, predictor, border)

    elif predictor == PredictorType.SAD:
        # Find best predictor (minimum SAD)
        best_sad = float("inf")
        best_result = None
        best_type = 0

        for p in range(MAX_PRED):
            result = predict_segment(data, seg, p, border)
            sad = _calc_sad(result, data, seg, border)
            if sad < best_sad:
                best_sad = sad
                best_result = result
                best_type = p

        seg.pred_type = best_type
        return best_result if best_result is not None else pred_none(data, seg, border)

    elif predictor == PredictorType.BSAD:
        # Find worst predictor (maximum SAD) - for glitch effects
        worst_sad = float("-inf")
        worst_result = None
        worst_type = 0

        for p in range(MAX_PRED):
            result = predict_segment(data, seg, p, border)
            sad = _calc_sad(result, data, seg, border)
            if sad > worst_sad:
                worst_sad = sad
                worst_result = result
                worst_type = p

        seg.pred_type = worst_type
        return worst_result if worst_result is not None else pred_none(data, seg, border)

    # Standard predictors
    if predictor in PREDICTORS:
        seg.pred_type = predictor
        return PREDICTORS[predictor](data, seg, border)

    # Unknown predictor
    seg.pred_type = PredictorType.NONE
    return pred_none(data, seg, border)


def predict_image(
    buffer: ImageBuffer,
    segments: SegmentList,
    predictor: int | PredictorType,
    channel: int = 0,
) -> ImageBuffer:
    """
    Generate predictions for all segments in an image.

    Args:
        buffer: Input image
        segments: Segment list
        predictor: Predictor type to use
        channel: Which channel to predict

    Returns:
        ImageBuffer containing predictions
    """
    data = buffer.data[channel]
    h, w = data.shape
    border = buffer.border_value[channel] if buffer.border_value else 0.0

    result = np.zeros((h, w), dtype=np.float32)

    for seg in segments:
        pred = predict_segment(data, seg, predictor, border)

        # Place prediction in result
        x_end = min(seg.x + seg.size, w)
        y_end = min(seg.y + seg.size, h)
        pred_h = y_end - seg.y
        pred_w = x_end - seg.x

        result[seg.y:y_end, seg.x:x_end] = pred[:pred_h, :pred_w]

    # Create output buffer with just this channel
    out_data = np.zeros_like(buffer.data)
    out_data[channel] = result

    return ImageBuffer(
        data=out_data,
        colorspace=buffer.colorspace,
        border_value=buffer.border_value,
        metadata={"predictor": predictor, "channel": channel},
    )
