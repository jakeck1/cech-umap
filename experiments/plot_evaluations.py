# plot_evals.py
# Publication-ready plotting for evaluation runs saved by run_evals.py

import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

IN_DIR = '../merged_results'
OUT_DIR = '../plots'
os.makedirs(OUT_DIR, exist_ok=True)

USE_TEX = False         
PROFILE = "neurips"     
MAKE_MULTIPANEL = False  

# Colorblind-safe categorical palette (Okabe–Ito)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]

METHOD_DISPLAY = {
    "umap": "UMAP",
    "cumap": r"ČUMAP",  # ČUMAP rendered with a check accent
}

METRIC_DISPLAY = {
    "trustworthiness": "Trustworthiness",
    "pca_procrustes": "PCA–metric",
    "Wass_H0": r"Wasserstein $H_0$",
    "Wass_H1": r"Wasserstein $H_1$",
}

def set_pub_style(profile: str = "neurips", use_tex: bool = False):
    # base typography
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif" if not use_tex else "serif",
        "font.size": 10 if profile in {"neurips", "icml"} else 9,
        "axes.titlesize": 11 if profile in {"neurips", "icml"} else 10,
        "axes.labelsize": 10 if profile in {"neurips", "icml"} else 9,
        "legend.fontsize": 12 if profile in {"neurips", "icml"} else 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
    })
    # figure sizes (inches) roughly matching common column widths
    if profile in {"neurips", "icml"}:
        mpl.rcParams["figure.figsize"] = (3.5, 2.6)  # single-column
    elif profile == "nature":
        mpl.rcParams["figure.figsize"] = (3.4, 2.5)
    else:  # science / default
        mpl.rcParams["figure.figsize"] = (3.42, 2.5)

def _load_npz(path):
    z = np.load(path, allow_pickle=True)
    data = z["data"]                      # shape [M,R,K,S]
    methods = list(z["methods"])
    metrics = list(z["metrics"])
    neighbors = list(z["neighbors"])
    seeds = list(z["seeds"])
    return data, methods, metrics, neighbors, seeds

def _mean_std_over_seeds(arr):  # arr shape [K, S]
    mu = np.nanmean(arr, axis=1)
    sd = np.nanstd(arr, axis=1, ddof=1)
    return mu, sd

def _pretty_metric(metric_key: str) -> str:
    return METRIC_DISPLAY.get(metric_key, metric_key.replace("_", " "))

def _pretty_method(method_key: str) -> str:
    return METHOD_DISPLAY.get(method_key, method_key)

def _safe_color(i):
    return PALETTE[i % len(PALETTE)]

def plot_per_metric_figures(path_npz: str, out_dir: str,create_big_plots = True):
    ds_name = os.path.splitext(os.path.basename(path_npz))[0]
    data, methods, metrics, neighbors, seeds = _load_npz(path_npz)
    M, R, K, S = data.shape
    assert K == len(neighbors)
    bigfig,bigax = plt.subplots(nrows=len(data),ncols = len(metrics))
    for ri, metric in enumerate(metrics):
        fig, ax = plt.subplots()

        # lines + shaded std
        for mi, method in enumerate(methods):
            vals = data[mi, ri]  # [K, S]
            mu, sd = _mean_std_over_seeds(vals)
            c = _safe_color(mi)
            ax.plot(
                neighbors, mu,
                label=_pretty_method(method),
                linewidth=1.8, marker="o", markersize=4,
                color=c,
            )
            ax.fill_between(neighbors, mu - sd, mu + sd, alpha=0.15, linewidth=0, color=c)

        ax.set_xlabel("Number of neighbors")
        ax.set_ylabel(_pretty_metric(metric),fontsize=16)
        ax.set_title(ds_name,fontsize=16)
        ax.grid(True, axis="both")
        ax.legend(frameon=False, ncol=1)
        # Tight numeric formatting
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        fig_path_png = os.path.join(out_dir, f"{ds_name}_{metric}.png")
        fig_path_pdf = os.path.join(out_dir, f"{ds_name}_{metric}.pdf")
        fig.savefig(fig_path_png, bbox_inches="tight")
        fig.savefig(fig_path_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path_png} and {fig_path_pdf}")

def plot_multipanel(path_npz: str, out_dir: str):
    """Optional: compact multi-panel (rows=metrics, shared legend)."""
    ds_name = os.path.splitext(os.path.basename(path_npz))[0]
    data, methods, metrics, neighbors, seeds = _load_npz(path_npz)
    M, R, K, S = data.shape

    # multipanel figure sizing
    cols = 2
    rows = int(np.ceil(R / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 2.6*rows), squeeze=False)

    for ri, metric in enumerate(metrics):
        r, c = divmod(ri, cols)
        ax = axes[r, c]
        for mi, method in enumerate(methods):
            vals = data[mi, ri]  # [K,S]
            mu, sd = _mean_std_over_seeds(vals)
            color = _safe_color(mi)
            ax.plot(neighbors, mu, label=_pretty_method(method),
                    linewidth=1.6, marker="o", markersize=3.5, color=color)
            ax.fill_between(neighbors, mu - sd, mu + sd, alpha=0.15, linewidth=0, color=color)
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel(_pretty_metric(metric))
        ax.grid(True, alpha=0.35)
        ax.set_title(_pretty_metric(metric), pad=2)

    # hide any unused axes
    for i in range(R, rows*cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    # Shared legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(ds_name, y=1.04, fontsize=11)
    fig.tight_layout()

    fig_path_png = os.path.join(out_dir, f"{ds_name}_multipanel.png")
    fig_path_pdf = os.path.join(out_dir, f"{ds_name}_multipanel.pdf")
    fig.savefig(fig_path_png, bbox_inches="tight")
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path_png} and {fig_path_pdf}")

 
   

if __name__ == "__main__":
    set_pub_style(profile=PROFILE, use_tex=USE_TEX)

    files = [f for f in os.listdir(IN_DIR) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {IN_DIR}. Run merge_evaluations.py first.")
    for fname in sorted(files):
        npz_path = os.path.join(IN_DIR, fname)
        plot_per_metric_figures(npz_path, OUT_DIR)
        if MAKE_MULTIPANEL:
            plot_multipanel(npz_path, OUT_DIR)
