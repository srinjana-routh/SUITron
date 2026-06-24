"""
io.py
Utilities for loading solar images (FITS, JPG, PNG) and saving segmentation masks.
"""

import os
import cv2
import numpy as np

# FITS support via astropy (optional — only needed for .fits files)
try:
    from astropy.io import fits as astropy_fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: str, hdu_index: int = 1):
    """
    Load a solar image from disk and return a 3-channel uint8 RGB array
    suitable for Detectron2, plus the original FITS header if available.

    Parameters
    ----------
    path : str
        Path to a .fits/.fit, .jpg, .jpeg, or .png file.
    hdu_index : int
        Which HDU to read from a FITS file. Default 1 (most GONG files).
        Use 0 for simple single-extension FITS.

    Returns
    -------
    rgb_image : np.ndarray  shape (H, W, 3), dtype uint8
    header    : astropy FITS Header or None
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext in (".fits", ".fit", ".fts"):
        return _load_fits(path, hdu_index)
    elif ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        return _load_standard(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Supported: .fits, .fit, .jpg, .jpeg, .png, .tif, .tiff"
        )


def _load_fits(path: str, hdu_index: int):
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required to read FITS files. "
            "Install it with:  pip install astropy"
        )
    with astropy_fits.open(path) as hdul:
        # Try requested HDU; fall back to 0 if index is out of range
        try:
            image_data = hdul[hdu_index].data
            header = hdul[hdu_index].header
        except IndexError:
            image_data = hdul[0].data
            header = hdul[0].header

        if image_data is None:
            raise ValueError(
                f"HDU {hdu_index} in '{path}' has no data. "
                "Try hdu_index=0."
            )

        image_data = image_data.astype(np.float32)

        # Handle 3-D cubes (e.g. SDO/AIA multi-frame) — take first frame
        if image_data.ndim == 3:
            image_data = image_data[0]

    rgb = _normalise_to_rgb(image_data)
    return rgb, header


def _load_standard(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # If already colour, just return; if grayscale after conversion still 3ch
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    return rgb, None


def _normalise_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a float array → 8-bit grayscale → 3-channel RGB."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        arr_norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_norm = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    return cv2.cvtColor(arr_norm, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results(results, output_dir: str):
    """
    Save per-class and combined segmentation masks to *output_dir*.

    Creates:
      output_dir/individual_class_masks/<stem>_<class>_result.png  (.fits)
      output_dir/combined_masks/<stem>_result.png  (.fits)

    Parameters
    ----------
    results : SUITronResults
        The object returned by SUITronPredictor.predict().
    output_dir : str
        Root directory for outputs (created if it doesn't exist).
    """
    os.makedirs(output_dir, exist_ok=True)

    indiv_dir   = os.path.join(output_dir, "individual_class_masks")
    combined_dir = os.path.join(output_dir, "combined_masks")
    os.makedirs(indiv_dir,   exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    stem = results.stem

    # ---- per-class masks ----
    for class_name, mask_np in results.class_masks.items():
        _save_mask_png(mask_np, indiv_dir, f"{stem}_{class_name}_result.png")
        if ASTROPY_AVAILABLE and results.header is not None:
            _save_mask_fits(
                mask_np, results.header,
                indiv_dir, f"{stem}_{class_name}_result.fits"
            )

    # ---- combined mask ----
    combined_np = results.combined_mask
    _save_mask_png(combined_np, combined_dir, f"{stem}_result.png")
    if ASTROPY_AVAILABLE and results.header is not None:
        _save_mask_fits(
            combined_np, results.header,
            combined_dir, f"{stem}_result.fits"
        )


def _save_mask_png(mask_np: np.ndarray, directory: str, filename: str):
    path = os.path.join(directory, filename)
    cv2.imwrite(path, mask_np.astype(np.uint8))


def _save_mask_fits(mask_np: np.ndarray, header, directory: str, filename: str):
    path = os.path.join(directory, filename)
    hdu = astropy_fits.PrimaryHDU(data=mask_np.astype(np.uint8), header=header)
    hdu.writeto(path, overwrite=True)
