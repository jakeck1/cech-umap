#!/usr/bin/env python3
import argparse
import os
import json

import numpy as np

from evaluation.evaluation import evaluate_embeddings
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


def run_single(dataset: str, neigh: int, seed: int, embedding_dir: str, out_dir: str):
    """Load embeddings and compute evaluation metrics for one triple."""
    umap_path = os.path.join(embedding_dir, f"UMAP__{dataset}_{neigh}_{seed}.npy")
    cumap_path = os.path.join(embedding_dir, f"CUMAP__{dataset}_{neigh}_{seed}.npy")

    umap_emb = np.load(umap_path)
    cumap_emb = np.load(cumap_path)

    results = evaluate_embeddings(
        {"umap": umap_emb, "cumap": cumap_emb},
        k_trust=neigh,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}__{neigh}__{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[EVAL] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embeddings and save JSON metrics."
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
        "--embedding-dir",
        type=str,
        default="../embeddings",
        help="Directory where embeddings were saved.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../results",
        help="Directory where JSON evaluation files will be written.",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(loader_dict.keys())
    neighbors = args.neighbors if args.neighbors else DEFAULT_NEIGHBORS
    seeds = args.seed if args.seed else DEFAULT_SEEDS

    for d in datasets:
        for k in neighbors:
            for s in seeds:
                print(f"[EVAL] dataset={d} k={k} seed={s}")
                run_single(d, k, s, args.embedding_dir, args.out_dir)


if __name__ == "__main__":
    main()
