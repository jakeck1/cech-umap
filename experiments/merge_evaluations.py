#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from typing import Dict, Tuple

import numpy as np


def parse_name(filepath: str) -> Tuple[str, int, int] | None:
    """
    Extract (dataset, neighbors, seed) from a filename of the form:
        <anything>/<DATASET>__<k>__<seed>.json
    e.g. 'results/MNIST__5__3.json' -> ('MNIST', 5, 3)
    """
    m = re.search(r"([^/\\]+)__(\d+)__(\d+)\.json$", filepath)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def merge_partials(
    partial_dir: str,
    out_dir: str,
    methods: list[str],
    metrics: list[str],
    neighbors: list[int],
    seeds: list[int],
) -> None:
    """
    Merge individual JSON evaluation files into one .npz per dataset with shape:
        (n_methods, n_metrics, n_neighbors, n_seeds)
    """
    files = glob.glob(os.path.join(partial_dir, "*.json"))
    by_ds: Dict[str, Dict[Tuple[int, int], dict]] = {}

    for path in files:
        parsed = parse_name(path)
        if not parsed:
            print(f"Skipping unparsed file: {path}")
            continue

        ds, k, s = parsed
        with open(path) as f:
            rep = json.load(f)

        by_ds.setdefault(ds, {})[(k, s)] = rep

    os.makedirs(out_dir, exist_ok=True)

    for ds, repmap in by_ds.items():
        print(f"[MERGE] dataset={ds}")
        M, R = len(methods), len(metrics)
        K, S = len(neighbors), len(seeds)

        data = np.full((M, R, K, S), np.nan, dtype=np.float64)

        for ki, k in enumerate(neighbors):
            for si, s in enumerate(seeds):
                rep = repmap.get((k, s))
                if rep is None:
                    continue

                for mi, meth in enumerate(methods):
                    for ri, met in enumerate(metrics):
                        # JSON should look like: rep["umap"]["trustworthiness"], etc.
                        data[mi, ri, ki, si] = rep.get(meth, {}).get(met, np.nan)

        base = os.path.join(out_dir, ds)
        npz_path = f"{base}.npz"
        np.savez_compressed(
            npz_path,
            data=data,
            methods=np.array(methods, dtype=object),
            metrics=np.array(metrics, dtype=object),
            neighbors=np.array(neighbors, dtype=int),
            seeds=np.array(seeds, dtype=int),
        )
        print(f"[MERGE] wrote {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-(dataset,k,seed) JSON evaluation files into .npz."
    )
    parser.add_argument(
        "--partial-dir",
        type=str,
        default="../results",
        help="Directory containing individual JSON result files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../merged_results",
        help="Directory where merged .npz files will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["umap", "cumap"],  # adjust if your JSON keys differ
        help="Method names (keys in the JSON files).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["trustworthiness", "pca_procrustes", "Wass_H0", "Wass_H1"],
        help="Metric names (keys under each method in the JSON files).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        nargs="*",
        default=[2, 3, 5, 8, 13, 21, 34, 55],
        help="Neighbor values (k) to expect.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to expect.",
    )
    args = parser.parse_args()

    merge_partials(
        partial_dir=args.partial_dir,
        out_dir=args.out_dir,
        methods=args.methods,
        metrics=args.metrics,
        neighbors=args.neighbors,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
