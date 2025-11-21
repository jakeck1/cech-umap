#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import umap

from cech_umap.cumap import CechUMAP
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

loader_dict = {
    "MNIST": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "pbmc68k": load_pbmc68k_reduced,
    "hematopoesis": load_paul15,
    "USPS": load_usps,
    "coil20": load_coil20,
    "20news": load_20ng_tfidf,
    "cifar10": load_cifar10_resnet18_feats,
}

DEFAULT_NEIGHBORS = [2, 3, 5, 8, 13, 21, 34, 55]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_single(dataset: str, neigh: int, seed: int, out_dir: str) -> None:
    """Compute UMAP and CechUMAP embeddings for one (dataset, neighbors, seed)."""
    get_data = loader_dict[dataset]
    X, y = get_data()

    umap_model = umap.UMAP(n_neighbors=neigh, random_state=seed)
    umap_emb = umap_model.fit_transform(X)

    X_t = torch.tensor(X).float()
    cumap_model = CechUMAP(random_state=seed)
    cumap_emb = cumap_model.fit_transform(X_t)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"UMAP__{dataset}_{neigh}_{seed}.npy"), umap_emb)
    np.save(os.path.join(out_dir, f"CUMAP__{dataset}_{neigh}_{seed}.npy"), cumap_emb)


def main():
    parser = argparse.ArgumentParser(
        description="Create UMAP and CechUMAP embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=loader_dict.keys(),
        help="If omitted, run on all datasets.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        nargs="*",
        help="List of neighbor values. If omitted, use paper defaults.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        help="List of seeds. If omitted, use paper defaults.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../embeddings",
        help="Output directory for embeddings.",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(loader_dict.keys())
    neighbors = args.neighbors if args.neighbors else DEFAULT_NEIGHBORS
    seeds = args.seed if args.seed else DEFAULT_SEEDS

    for d in datasets:
        for k in neighbors:
            for s in seeds:
                print(f"[CREATE_EMBEDDINGS] dataset={d} k={k} seed={s}")
                run_single(d, k, s, args.out_dir)


if __name__ == "__main__":
    main()
