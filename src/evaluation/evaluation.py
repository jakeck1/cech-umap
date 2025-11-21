# eval_helper.py
# Minimal evaluation helpers: trustworthiness, PCA-Procrustes, Wasserstein topo distances.

from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from sklearn.manifold import trustworthiness
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise_distances
from scipy.spatial import procrustes
from scipy import sparse as _sp

# Optional topology imports
from ripser import ripser
from persim import wasserstein
_TOPO_OK = True


def pca_procrustes_global(X, Z, random_state: int = 0) -> float:
    """
    Align low-dim embedding Z to d-dim PCA/SVD of X and return 1 - disparity \in (0,1].
    Uses centered PCA for dense X; TruncatedSVD for sparse X.
    """
    Z = np.asarray(Z, dtype=np.float32)
    d = Z.shape[1]
    if _sp.issparse(X):
        P = TruncatedSVD(n_components=d, random_state=random_state).fit_transform(X)
    else:
        P = PCA(n_components=d, random_state=random_state).fit_transform(X)

    P = P - P.mean(0, keepdims=True)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, disparity = procrustes(P, Zc)
    return float(1.0 - disparity)

def topo_wasserstein(
    X,
    Z,
    n_samples: int = 800,
    maxdim: int = 2,
    metric_high: str = "euclidean",
    random_state: int = 0,
    R: int = 30,   # number of repeated subsamples
) -> Dict[str, float]:

    if not _TOPO_OK:
        return {}

    rs = np.random.RandomState(random_state)
    n = X.shape[0]
    m = min(n_samples, n)

    out_vals = {f"Wass_H{k}": [] for k in range(maxdim + 1)}

    for _ in range(R):
        idx = rs.choice(n, size=m, replace=False)
        XA = X[idx]
        ZA = Z[idx]

        DX = pairwise_distances(XA, XA, metric=metric_high)
        DZ = pairwise_distances(ZA, ZA, metric="euclidean")

        # --- Normalization by median pairwise distance ---
        def _normalize(D):
            iu = np.triu_indices_from(D, k=1)
            s = np.median(D[iu])
            s = max(s, 1e-12)
            return D / s

        DX = _normalize(DX)
        DZ = _normalize(DZ)

        RX = ripser(DX, maxdim=maxdim, distance_matrix=True)
        RZ = ripser(DZ, maxdim=maxdim, distance_matrix=True)

        for k in range(maxdim + 1):
            DXk = RX["dgms"][k]
            DZk = RZ["dgms"][k]

            # --- Remove infinite H0 bar (essential class) ---
            if np.isinf(DXk[:, 1]).any():
                DXk = DXk[~np.isinf(DXk[:, 1])]
            if np.isinf(DZk[:, 1]).any():
                DZk = DZk[~np.isinf(DZk[:, 1])]

            w = float(wasserstein(DXk, DZk, matching=False))
            out_vals[f"Wass_H{k}"].append(w)

    # return mean over repeated subsamples
    return {k: float(np.mean(v)) for k, v in out_vals.items()}



def evaluate_embeddings(
    X,
    embeddings: Dict[str, np.ndarray],
    *,
    k_trust: int = 15,
    metric_high: str = "euclidean",
    topo_samples: int = 800,
    random_state: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Returns for each method:
      - trustworthiness
      - pca_procrustes
      - Wass_H0, Wass_H1,
    """
    report: Dict[str, Dict[str, float]] = {}
    for name, Z in embeddings.items():
        Z = np.asarray(Z, dtype=np.float32)
        # Trustworthiness (local)
        tw = float(trustworthiness(X, Z, n_neighbors=k_trust, metric=metric_high))
        # Global PCA-Procrustes
        pproc = pca_procrustes_global(X, Z, random_state=random_state)
        # Topology
        topo = topo_wasserstein(
            X, Z,
            n_samples=topo_samples,
            maxdim=1,
            metric_high=metric_high,
            random_state=random_state,
        )
        out = {"trustworthiness": tw, "pca_procrustes": pproc}
        out.update(topo)
        report[name] = out
    return report
