from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Literal
import numpy as np
import torch
import random 
import math

def find_ab_params(min_dist: float, spread: float) -> Tuple[float, float]:
    def curve(x, a, b): return 1.0 / (1.0 + a * np.power(x, 2.0 * b))
    xv = np.linspace(0, spread * 3.0, 300).astype(np.float64)
    yv = np.zeros_like(xv)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    (a, b), _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0), maxfev=10000)
    return float(max(a, 1e-6)), float(max(b, 1e-6))
   

def phi_kernel(dist: torch.Tensor,
                kind: Literal["exp","student"],
                tau: float = 1.0) -> torch.Tensor:
    if kind == "exp":
        return torch.exp(-dist / max(tau, 1e-9))
    elif kind == "student":
        return 1.0/(1.0 + dist**2)
    else:
        raise NotImplementedError
    


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pca_torch(X: torch.Tensor, n_components: int = 50) -> torch.Tensor:
    # Center in-place-friendly way (no extra alloc for mean-broadcast)
    X = X - X.mean(dim=0, keepdim=True)
    n, d = X.shape
    if d <= n:
        # SVD on (n x d), thin
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        comps = Vh[:n_components].T
        Z = X @ comps
    else:
        # Eigh on XX^T
        C = (X @ X.T) / max(d - 1, 1)
        evals, U = torch.linalg.eigh(C)
        idx = torch.argsort(evals, descending=True)[:n_components]
        U = U[:, idx]
        S = torch.sqrt(torch.clamp(evals[idx], min=1e-9)) * math.sqrt(max(d - 1, 1))
        Z = U * S
    return Z


def auto_triplet_microbatch(
    K: int, D: int,                      # neighbors and ambient dim of X_knn
    target_util_gb: float = 8.0,         # how much free VRAM to aim to use
    bf16: bool = True
) -> int:
    bpe = 2 if (bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 4
    C = 3 * K
    approx_bytes_per_row = C * D * bpe * 6
    B = int((target_util_gb * (1024**3)) // max(approx_bytes_per_row, 1))
    return max(2048, min(65536, B))
