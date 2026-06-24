# 🌞 SUITron — Solar Feature Segmentation with Detectron2
This is a deep learning model, primarily designed for filament extraction from full-disk observations from the Solar Ultraviolet Imaging Telescope (SUIT) aboard Aditya-L1. It uses a Mask R-CNN model trained on GONG Hα observations and can run on other passband images as well (IRIS Mg II, Kodaikanal H-alpha).

## What It Does

Given a solar image (FITS or JPG/PNG), **SUITron**:

1. Normalises and prepares the image for the model
2. Runs Mask R-CNN instance segmentation
3. Returns per-class binary masks for **filaments** (left-oriented, right-oriented and unidentifiable orientation)

---

## Quickstart (5 minutes)

### 1. Install dependencies

```bash
pip install torch torchvision
pip install pyyaml==5.1

# Install Detectron2 (from source — required)
git clone https://github.com/facebookresearch/detectron2
pip install -e detectron2/
```

> **GPU recommended.** If you're on CPU only, set `cfg.MODEL.DEVICE = "cpu".

### 2. Download the model

Download `model_final.pth` and place it anywhere accessible:

```
suitron/
├── model_final.pth   ← put it here, or pass the path explicitly
```

### 3. Run on your images

```python
from suitron import SUITronPredictor

predictor = SUITronPredictor("path/to/model_final.pth")
results = predictor.predict("path/to/your_solar_image.fits")

results.show()                  # visualise detections
results.save("output/")         # save masks as PNG + FITS
```

## Usage

### Python API

```python
import SUITronPredictor

# Load predictor
predictor = SUITronPredictor(
    model_path="model_final.pth",
    score_threshold=0.4,   # detection confidence threshold (0–1)
    device="cuda",         # "cuda" or "cpu"
)

# --- From a FITS file ---
results = predictor.predict("20150910_gong.fits", hdu_index=1)

# --- From a JPEG/PNG ---
results = predictor.predict("20150910_gong.jpg")

# Inspect results
print(f"Found {results.num_instances} features")
print(f"Classes: {results.class_names}")       # e.g. ['Left', 'Right', 'Unidentifiable']
print(f"Scores: {results.scores}")             # detection confidences

# Visualise
results.show()

# Save
results.save("output/", fits_header=results.header)  # preserves FITS header if available
```

### Batch processing

```python
import os
import SUITronPredictor

predictor = SUITronPredictor("model_final.pth")

input_dir  = "images/"
output_dir = "results/"

for fname in os.listdir(input_dir):
    results = predictor.predict(os.path.join(input_dir, fname))
    results.save(output_dir)
    print(f"✓ {fname} — {results.num_instances} features detected")
```

### Score threshold tuning

- `0.4` (default) — good balance of recall and precision (tested on GONG Hα; below example)
- `0.5–0.6` — fewer, higher-confidence detections
- `0.2–0.3` — more detections, may include faint features; increases false positives

- <img width="1119" height="276" alt="image" src="https://github.com/user-attachments/assets/2db109ae-f8a0-4722-bbe8-34c34ce467b6" />



## Citation

If you use SUITron in your research, please cite:

```bibtex
@software{suitron,
  author  = {Routh, S.},
  title   = {SUITron},
  year    = {2026},
  url     = {https://github.com/srinjana-routh/SUITron}
}
