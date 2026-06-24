"""
suitron/visualise.py
Plotting helpers for SUITron results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_results(results, figsize=(16, 5)):
    """
    Display the original image, Detectron2 overlay, and per-class masks
    side by side.

    Parameters
    ----------
    results : SUITronResults
    figsize : tuple
    """
    n_classes = len(results.class_masks)
    ncols = 2 + n_classes   # original | overlay | class masks...

    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    # 1. Original image
    axes[0].imshow(results.original_image, cmap="gray")
    axes[0].set_title("Input Image", fontsize=11)
    axes[0].axis("off")

    # 2. Detectron2 overlay
    axes[1].imshow(results.overlay_image)
    axes[1].set_title(
        f"Detections  (n={results.num_instances}, "
        f"thresh={results.score_threshold:.2f})",
        fontsize=11,
    )
    axes[1].axis("off")

    # 3. Per-class masks
    class_colors = plt.cm.Set1.colors
    for idx, (class_name, mask_np) in enumerate(results.class_masks.items()):
        ax = axes[2 + idx]
        ax.imshow(mask_np, cmap="nipy_spectral", interpolation="nearest")
        n_feat = int(mask_np.max())
        ax.set_title(f"{class_name.capitalize()} Mask\n({n_feat} instances)", fontsize=11)
        ax.axis("off")

    plt.suptitle(results.stem, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def show_class_overlay(original_image, class_masks, alpha=0.5, figsize=(8, 8)):
    """
    Overlay all class masks on the original image with semi-transparent colour.
    """
    palette = {
        "filament": (0.2, 0.4, 1.0),   # blue
        "plage":    (1.0, 0.5, 0.1),   # orange
    }
    default_colors = plt.cm.tab10.colors

    display = np.array(original_image, dtype=float) / 255.0
    if display.ndim == 2:
        display = np.stack([display] * 3, axis=-1)

    legend_patches = []
    for i, (class_name, mask_np) in enumerate(class_masks.items()):
        color = palette.get(class_name, default_colors[i % len(default_colors)])
        binary = (mask_np > 0).astype(float)
        for c in range(3):
            display[:, :, c] = np.where(
                binary > 0,
                (1 - alpha) * display[:, :, c] + alpha * color[c],
                display[:, :, c],
            )
        legend_patches.append(
            mpatches.Patch(color=color, label=class_name.capitalize())
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.clip(display, 0, 1))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)
    ax.axis("off")
    ax.set_title("Class Overlay", fontsize=12)
    plt.tight_layout()
    plt.show()
