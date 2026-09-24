import numpy as np

NODATA = 0
LIT = 1
SELF_SHADOWED = 2


def saturated(image, pileup_fraction=0.002, min_count=8):
    image = np.asarray(image)
    finite = image[np.isfinite(image)] if np.issubdtype(image.dtype, np.floating) else image.ravel()
    if finite.size == 0:
        return np.zeros(image.shape, dtype=bool)
    peak = finite.max()
    count = np.count_nonzero(finite == peak)
    if count < min_count or count / finite.size <= pileup_fraction:
        return np.zeros(image.shape, dtype=bool)
    return image >= peak


def estimate_valid(image, nodata=None, shadow_threshold=None, saturation_threshold=None,
                   pileup_fraction=0.002):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        full = float(np.iinfo(image.dtype).max)
        low = 0.02 * full if shadow_threshold is None else shadow_threshold
        high = full - 1.0 if saturation_threshold is None else saturation_threshold
        finite = np.ones(image.shape, dtype=bool)
    else:
        low = 0.02 if shadow_threshold is None else shadow_threshold
        high = np.inf if saturation_threshold is None else saturation_threshold
        finite = np.isfinite(image)

    valid = finite & (image > low) & (image < high)
    if saturation_threshold is None and pileup_fraction is not None:
        valid &= ~saturated(image, pileup_fraction)
    if nodata is not None:
        valid &= image != nodata
    return valid
