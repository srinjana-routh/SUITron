"""
SUITron/predictor.py
Core predictor class that wraps Detectron2 for solar feature segmentation.
"""

import os
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from .io import load_image, save_results
from .visualise import show_results


# Default class names for the shipped model_final.pth
# (filament = dark elongated structures; plage = bright chromospheric regions)
DEFAULT_CLASS_NAMES = ["Left", "Right", "Unidentifiable"]


class SUITronResults:
    """
    Container for all outputs produced by one predictor call.

    Attributes
    ----------
    original_image : np.ndarray  (H, W, 3) uint8 RGB
    overlay_image  : np.ndarray  (H, W, 3) uint8 RGB  — Detectron2 viz
    class_masks    : dict  {class_name: np.ndarray (H, W) uint8}
                     Each unique instance gets its own integer label (1, 2, …).
    combined_mask  : np.ndarray (H, W) int32
                     All instances combined; each gets a unique integer label.
    num_instances  : int
    class_names    : list[str]
    scores         : list[float]
    header         : astropy FITS Header or None
    stem           : str   base filename (no extension)
    score_threshold: float
    """

    def __init__(
        self,
        original_image,
        overlay_image,
        class_masks,
        combined_mask,
        scores,
        class_names,
        header,
        stem,
        score_threshold,
    ):
        self.original_image  = original_image
        self.overlay_image   = overlay_image
        self.class_masks     = class_masks
        self.combined_mask   = combined_mask
        self.num_instances   = int(combined_mask.max())
        self.class_names     = class_names
        self.scores          = scores
        self.header          = header
        self.stem            = stem
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    def show(self, figsize=(16, 5)):
        """Display results inline (Jupyter / matplotlib)."""
        show_results(self, figsize=figsize)

    def save(self, output_dir: str):
        """Save PNG and (if header present) FITS masks to *output_dir*."""
        save_results(self, output_dir)
        print(f"✓ Saved results to '{output_dir}'")

    def __repr__(self):
        counts = {k: int((v > 0).sum()) for k, v in self.class_masks.items()}
        return (
            f"SUITronResults(stem='{self.stem}', "
            f"instances={self.num_instances}, "
            f"pixel_counts={counts})"
        )


# ---------------------------------------------------------------------------

class SUITronPredictor:
    """
    Load a trained SUITron model and run inference on solar images.

    Parameters
    ----------
    model_path : str
        Path to model_final.pth (or any Detectron2 checkpoint).
    score_threshold : float
        Minimum confidence score to keep a detection. Default 0.4.
    device : str
        "cuda" (GPU) or "cpu". Defaults to "cuda" if available.
    class_names : list[str]
        Ordered list of class names the model was trained on.
        Default: ["filament", "plage"]
    num_classes : int
        Must match the number of classes the model was trained on.
        Default: 2

    Example
    -------
    >>> import SUITronPredictor
    >>> p = SUITronPredictor("model_final.pth")
    >>> results = p.predict("20150910_gong.fits")
    >>> results.show()
    >>> results.save("output/")
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.4,
        device: str = None,
        class_names: list = None,
        num_classes: int = 3,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'\n"
                "Download model_final.pth from the GitHub Releases page."
            )

        self.model_path      = model_path
        self.score_threshold = score_threshold
        self.class_names     = class_names or DEFAULT_CLASS_NAMES
        self.num_classes     = num_classes

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Build Detectron2 config & predictor
        self._cfg = self._build_cfg()
        self._predictor = DefaultPredictor(self._cfg)

        print(
            f"✓ SUITronPredictor loaded\n"
            f"  model   : {model_path}\n"
            f"  device  : {self.device}\n"
            f"  classes : {self.class_names}\n"
            f"  threshold: {self.score_threshold}"
        )

    # ------------------------------------------------------------------
    def _build_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS               = self.model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        cfg.MODEL.DEVICE                = self.device
        # Suppress Detectron2's verbose logger for cleaner notebook output
        cfg.freeze()
        return cfg

    # ------------------------------------------------------------------
    def predict(self, image_path: str, hdu_index: int = 1) -> SUITronResults:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path : str
            Path to .fits, .jpg, or .png file.
        hdu_index : int
            HDU index for FITS files (default 1; use 0 for primary HDU).

        Returns
        -------
        SUITronResults
        """
        rgb_image, header = load_image(image_path, hdu_index=hdu_index)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        # Remove double extension for .fits.gz etc.
        stem = os.path.splitext(stem)[0] if stem.endswith(".fits") else stem
        return self._run(rgb_image, header=header, stem=stem)

    def predict_array(self, array: np.ndarray, stem: str = "image") -> SUITronResults:
        """
        Run inference on a raw NumPy array.

        Parameters
        ----------
        array : np.ndarray
            2-D float/int array (any range) or 3-D (H, W, 3) uint8 RGB.
        stem : str
            Base name used for output filenames.

        Returns
        -------
        SUITronResults
        """
        import cv2
        if array.ndim == 2:
            # Normalise and convert to 3-channel
            lo, hi = np.nanmin(array), np.nanmax(array)
            norm = ((array - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            rgb  = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 3:
            rgb = array.astype(np.uint8)
        else:
            raise ValueError(
                "array must be shape (H, W) or (H, W, 3). "
                f"Got shape {array.shape}."
            )
        return self._run(rgb, header=None, stem=stem)

    # ------------------------------------------------------------------
    def _run(self, rgb_image: np.ndarray, header, stem: str) -> SUITronResults:
        """Internal: run Detectron2, build masks, return SUITronResults."""
        import cv2

        # Detectron2 expects BGR
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        outputs   = self._predictor(bgr_image)
        instances = outputs["instances"].to("cpu")

        H, W = rgb_image.shape[:2]

        # ---- Build per-class and combined masks ----
        class_masks   = {name: np.zeros((H, W), dtype=np.uint8)
                         for name in self.class_names}
        combined_mask = np.zeros((H, W), dtype=np.int32)

        pred_classes = instances.pred_classes.numpy()
        pred_masks   = instances.pred_masks.numpy()   # (N, H, W) bool
        scores       = instances.scores.numpy().tolist()

        for i, (cls_idx, mask_bool) in enumerate(zip(pred_classes, pred_masks)):
            cls_name = (
                self.class_names[cls_idx]
                if cls_idx < len(self.class_names)
                else f"class_{cls_idx}"
            )
            instance_id = i + 1  # 1-indexed unique label
            class_masks[cls_name][mask_bool] = instance_id
            combined_mask[mask_bool]         = instance_id

        # ---- Detectron2 overlay visualisation ----
        v   = Visualizer(rgb_image, metadata=None)
        out = v.draw_instance_predictions(instances)
        overlay_image = out.get_image()   # (H, W, 3) RGB uint8

        return SUITronResults(
            original_image  = rgb_image,
            overlay_image   = overlay_image,
            class_masks     = class_masks,
            combined_mask   = combined_mask,
            scores          = scores,
            class_names     = self.class_names,
            header          = header,
            stem            = stem,
            score_threshold = self.score_threshold,
        )
