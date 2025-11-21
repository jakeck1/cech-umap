#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.data import (
    load_mnist,
    load_fashion_mnist,
    load_pbmc68k_reduced,
    load_paul15,
    load_usps,
    load_coil20,
    load_20ng_tfidf,
    load_cifar10_resnet18_feats,
)

EMB_DIR = "../embeddings"
PLOT_DIR = "../plots"  
os.makedirs(PLOT_DIR, exist_ok=True)

DATASETS = {
    "MNIST": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "pbmc68k": load_pbmc68k_reduced,
    "hematopoesis": load_paul15,
    "USPS": load_usps,
    "coil20": load_coil20,
    "20news": load_20ng_tfidf,
    "cifar10": load_cifar10_resnet18_feats,
}

NEIGHBORS = [2, 3, 5, 8, 13, 21,33,55]
METHODS = ["UMAP", "CUMAP"]  # row order

# --------------------------------------------------------
# helpers
# --------------------------------------------------------
def _pick_cmap(num_classes: int):
    """
    Choose a colormap with enough distinct categories.
    """
    import matplotlib.cm as cm

    if num_classes <= 10:
        return cm.get_cmap("tab10", num_classes)
    elif num_classes <= 20:
        return cm.get_cmap("tab20", num_classes)
    else:
        # fallback for many classes; categorical spacing from a continuous map
        return cm.get_cmap("gist_ncar", num_classes)

def _labels_to_colors(y):
    """
    Map arbitrary labels in y (ints/strings) to colors using a suitable palette.
    """
    y_np = np.asarray(y)
    # handle torch tensors
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()

    # ensure 1D
    y_np = y_np.reshape(-1)

    # build an index per unique label
    uniq = np.unique(y_np)
    idx_map = {lab: i for i, lab in enumerate(uniq)}
    idx = np.vectorize(idx_map.get)(y_np)

    cmap = _pick_cmap(len(uniq))
    colors = cmap(idx)  # RGBA
    return colors

def _point_size(n):
    """
    Reasonable scatter size depending on N.
    """
    if n > 100_000:
        return 0.5
    if n > 50_000:
        return 0.8
    if n > 10_000:
        return 1.0
    if n > 5_000:
        return 2.0
    if n > 2_000:
        return 3.0
    return 4.0


    

if __name__ == "__main__":
    for key, loader in DATASETS.items():
        print(f"[{key}] loading labels to set colors …")
        # Only need labels for color; features aren't needed here
        X, y = loader()
        N = len(y)
        colors = _labels_to_colors(y)
        s = _point_size(N)

        n_rows = len(METHODS)
        n_cols = len(NEIGHBORS)

        # bigger figure for many columns
        # width per column ~ 2.6, height per row ~ 2.6
        fig_w = max(8, 2.6 * n_cols)
        fig_h = max(4.5, 2.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            constrained_layout=True
        )

        # keep track if at least one panel rendered; else skip saving
        rendered_any = False

        for r, method in enumerate(METHODS):
            for c, k in enumerate(NEIGHBORS):
                ax = axes[r, c]
                emb = np.load(os.path.join(EMB_DIR,f'{method}__{key}_{k}_{0}.npy'))
                if emb is None:
                    ax.axis("off")
                    ax.set_title(f"{method} | k={k}\n(missing)", fontsize=9)
                    continue

                if emb.shape[0] != N or emb.shape[1] < 2:
                    ax.axis("off")
                    ax.set_title(f"{method} | k={k}\n(shape mismatch: {emb.shape})", fontsize=9)
                    continue

                rendered_any = True
                # Use first two dims (assumes 2D)
                ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=s, linewidths=0, alpha=0.9)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title(f"{method} | k={k}", fontsize=10)

        if not rendered_any:
            plt.close(fig)
            print(f"[{key}] skipped (no embeddings found for any k).")
            continue

        out_png = os.path.join(PLOT_DIR, f"{key}_grid.png")
        out_pdf = os.path.join(PLOT_DIR, f"{key}_grid.pdf")

        fig.suptitle(f"{key}", fontsize=14, y=1.02)
        # tight layout already applied; still leave a bit of padding in savefig
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[{key}] saved: {out_png} and {out_pdf}")
