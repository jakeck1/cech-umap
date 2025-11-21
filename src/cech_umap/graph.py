import torch

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Literal
import scipy.sparse as sp

# knn_backends.py
from typing import Tuple, Optional, Literal, Dict, Any
import math
import torch

@torch.no_grad()
def _batched_knn_exact(
    X: torch.Tensor,
    k: int,
    row_bs: int = 8192,
    col_bs: int = 8192,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact kNN with tiling. Returns (knn_idx, knn_dist) of shape (N, k).
    - metric="euclidean": ℓ2 distances via torch.cdist (p=2).
    - metric="cosine": 1 - cosine_similarity, computed via dot-products on normalized X.
    """
    device, dtype = X.device, X.dtype
    N = X.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    k_eff = min(k + 1, N)  # include potential self then drop

    if metric == "cosine":
        # Normalize rows to unit norm (avoid division-by-zero with eps)
        eps = torch.finfo(X.dtype).eps
        Xn = X / (X.norm(dim=1, keepdim=True).clamp_min(eps))
    elif metric == "euclidean":
        Xn = X
    else:
        raise ValueError(f"Unsupported metric for exact backend: {metric}")

    best_dist = torch.full((N, k_eff), float("inf"), device=device, dtype=dtype)
    best_idx = torch.full((N, k_eff), -1, device=device, dtype=torch.long)
    row_ids_full = torch.arange(N, device=device)

    def _merge_block(dist_block: torch.Tensor, col_indices: torch.Tensor, row_range: slice):
        # in-place top-k merge without extra temporaries beyond concatenation
        r, c = dist_block.shape
        D_old = best_dist[row_range]
        I_old = best_idx[row_range]

        D_cat = torch.empty((r, D_old.shape[1] + c), device=device, dtype=dtype)
        I_cat = torch.empty((r, I_old.shape[1] + c), device=device, dtype=torch.long)
        D_cat[:, : D_old.shape[1]] = D_old
        D_cat[:, D_old.shape[1] :] = dist_block
        I_cat[:, : I_old.shape[1]] = I_old
        I_cat[:, I_old.shape[1] :] = col_indices.expand(r, c)

        td, ti = torch.topk(D_cat, k=k_eff, dim=1, largest=False)
        best_dist[row_range] = td
        best_idx[row_range] = torch.gather(I_cat, 1, ti)

    for r0 in range(0, N, row_bs):
        r1 = min(r0 + row_bs, N)
        A = Xn[r0:r1]
        for c0 in range(0, N, col_bs):
            c1 = min(c0 + col_bs, N)
            B = Xn[c0:c1]
            if metric == "euclidean":
                D = torch.cdist(A, B, p=2)
            else:  # cosine distance = 1 - cosine_similarity
                # A,B are row-normalized -> cosine similarity = A @ B^T
                S = A @ B.transpose(0, 1)
                D = 1.0 - S

            # mask diagonal when tiles overlap
            if not (c1 <= r0 or r1 <= c0):
                rows_abs = row_ids_full[r0:r1]
                cols_abs = row_ids_full[c0:c1]
                self_mask = rows_abs.view(-1, 1) == cols_abs.view(1, -1)
                D = D.masked_fill(self_mask, float("inf"))

            _merge_block(D, row_ids_full[c0:c1], slice(r0, r1))

    # paranoia: ensure self gone, cut to k
    is_self = (best_idx == row_ids_full.view(-1, 1))
    if is_self.any():
        best_dist = best_dist.masked_fill(is_self, float("inf"))
        td, ti = torch.topk(best_dist, k=k_eff, dim=1, largest=False)
        best_idx = torch.gather(best_idx, 1, ti)
        best_dist = td

    return best_idx[:, :k], best_dist[:, :k]

def _ensure_numpy(x: torch.Tensor):
    if x.device.type != "cpu":
        x = x.to("cpu")
    return x.detach().contiguous().numpy()

@torch.no_grad()
def _pynndescent_knn(
    X: torch.Tensor,
    k: int,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    pynndescent_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build kNN with PyNNDescent (approximate, CPU). Returns indices/distances as torch tensors
    on X.device/X.dtype.
    """
    try:
        from pynndescent import NNDescent  # type: ignore
    except Exception as e:
        raise ImportError("PyNNDescent not installed. pip install pynndescent") from e

    if k <= 0:
        raise ValueError("k must be positive")

    # Map metric names
    if metric == "euclidean":
        nn_metric = "euclidean"
    elif metric == "cosine":
        # PyNNDescent supports "cosine" directly (distance = 1 - cos_sim).
        nn_metric = "cosine"
    else:
        raise ValueError(f"Unsupported metric for PyNNDescent: {metric}")

    npX = _ensure_numpy(X)
    # Build the index and query points against themselves
    kwargs = dict(metric=nn_metric, n_neighbors=min(k + 1, max(2, k + 1)))
    if pynndescent_kwargs:
        kwargs.update(pynndescent_kwargs)

    index = NNDescent(npX, **kwargs)
    # Query all points; returns (indices, distances)
    I, D = index.query(npX, k=min(k + 1, npX.shape[0]))

    # Remove self (0 distance or self index)
    # Find and drop the self-column if present
    # Heuristic: self appears exactly once per row; pick column where index == row id
    import numpy as np
    rows = np.arange(npX.shape[0])[:, None]
    self_col = (I == rows).argmax(axis=1)  # position of self per row (ties unlikely)
    # Build masks to drop self, then take first k
    mask = np.ones_like(I, dtype=bool)
    mask[np.arange(I.shape[0]), self_col] = False
    I = I[mask].reshape(npX.shape[0], -1)[:, :k]
    D = D[mask].reshape(npX.shape[0], -1)[:, :k]

    # Back to torch on original device/dtype
    device, dtype = X.device, X.dtype
    knn_idx = torch.from_numpy(I).to(device=device, dtype=torch.long)
    knn_dist = torch.from_numpy(D).to(device=device, dtype=dtype)
    return knn_idx, knn_dist


@torch.no_grad()
def _faiss_knn(
    X: torch.Tensor,
    k: int,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    use_gpu: bool = True,
    index_kind: Literal["flat", "ivfpq", "hnsw"] = "ivfpq",
    ivf_nlist: int = 4096,
    ivf_nprobe: int = 64,
    pq_m: Optional[int] = None,         # number of subquantizers for PQ; default heuristic if None
    hnsw_m: int = 32,
    hnsw_efSearch: int = 128,
    hnsw_efConstruction: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build kNN with FAISS. Supports:
      - index_kind="flat"  : exact (L2/IP)
      - index_kind="ivfpq" : IVF-PQ (approximate)
      - index_kind="hnsw"  : HNSW (approximate, CPU-only in standard FAISS)
    Distances returned:
      - metric="euclidean": L2 distances
      - metric="cosine": 1 - cosine_similarity  (implemented via inner product on normalized vectors)
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise ImportError("FAISS not installed. pip install faiss-gpu (or faiss-cpu)") from e

    if k <= 0:
        raise ValueError("k must be positive")

    # Prepare data on CPU float32 for FAISS
    Xcpu = X.detach().to("cpu", dtype=torch.float32)
    N, d = Xcpu.shape
    ivf_nlist = min(ivf_nlist,int(X.shape[0]//2))
    # Metric handling
    use_ip = False
    if metric == "euclidean":
        faiss_metric = faiss.METRIC_L2
    elif metric == "cosine":
        # Normalize rows; then use inner product as similarity.
        # We'll later map similarity s -> distance 1 - s.
        eps = torch.finfo(Xcpu.dtype).eps
        Xcpu = Xcpu / (Xcpu.norm(dim=1, keepdim=True).clamp_min(eps))
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        use_ip = True
    else:
        raise ValueError(f"Unsupported metric for FAISS: {metric}")

    # Build the index
    if index_kind == "flat":
        index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)
        if use_gpu:
            # move to all GPUs if available
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)  # GPU 0 by default
        index.add(Xcpu.numpy())
        sim_or_negdist, I = index.search(Xcpu.numpy(), k + 1)  # +1 to allow self
    elif index_kind == "ivfpq":
        # IVF coarse quantizer
        coarse = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)
        if pq_m is None:
            # heuristic: subquantizers ~ sqrt(d)
            pq_m = max(8, int(round(math.sqrt(d))))
        index = faiss.IndexIVFPQ(coarse, d, ivf_nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
        # Training
        index.train(Xcpu.numpy())
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)  # GPU 0
        index.nprobe = ivf_nprobe
        index.add(Xcpu.numpy())
        sim_or_negdist, I = index.search(Xcpu.numpy(), k + 1)
    elif index_kind == "hnsw":
        # HNSW is typically CPU in FAISS builds
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
        index.hnsw.efSearch = hnsw_efSearch
        index.hnsw.efConstruction = hnsw_efConstruction
        index.add(Xcpu.numpy())
        sim_or_negdist, I = index.search(Xcpu.numpy(), k + 1)
    else:
        raise ValueError(f"Unknown FAISS index_kind: {index_kind}")

    # Convert returned scores to distances
    import numpy as np
    if metric == "euclidean":
        D = sim_or_negdist  # FAISS returns squared L2 for L2 indices; IndexFlatL2 returns L2^2
        # Convert to L2 (not squared) for consistency with your exact backend.
        D = np.maximum(D, 0.0)
        D = np.sqrt(D, dtype=D.dtype, where=np.ones_like(D, dtype=bool))
    else:
        # Inner product similarity s in [-1,1] (after normalization).
        # Cosine distance = 1 - s
        D = 1.0 - sim_or_negdist

    # Drop self; self is at position 0 for exact flat/IP; for ANN it should still include self with best score
    rows = np.arange(N)[:, None]
    self_col = (I == rows).argmax(axis=1)
    mask = np.ones_like(I, dtype=bool)
    mask[np.arange(N), self_col] = False
    I = I[mask].reshape(N, -1)[:, :k]
    D = D[mask].reshape(N, -1)[:, :k]

    device, dtype = X.device, X.dtype
    knn_idx = torch.from_numpy(I).to(device=device, dtype=torch.long)
    knn_dist = torch.from_numpy(D).to(device=device, dtype=dtype)
    return knn_idx, knn_dist


@torch.no_grad()
def compute_knn(
    X: torch.Tensor,
    k: int,
    nn_backend: Literal["exact", "pynndescent", "faiss"] = "exact",
    metric: Literal["euclidean", "cosine"] = "euclidean",
    # exact backend params
    row_bs: int = 8192,
    col_bs: int = 8192,
    # pynndescent params
    pynndescent_kwargs: Optional[Dict[str, Any]] = None,
    # faiss params
    faiss_use_gpu: bool = True,
    faiss_index_kind: Literal["flat", "ivfpq", "hnsw"] = "ivfpq",
    faiss_ivf_nlist: int = 4096,
    faiss_ivf_nprobe: int = 64,
    faiss_pq_m: Optional[int] = None,
    faiss_hnsw_m: int = 32,
    faiss_hnsw_efSearch: int = 128,
    faiss_hnsw_efConstruction: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified kNN API.
    Parameters
    ----------
    X : (N,d) torch.Tensor
        Input data. Output tensors will be on X.device and use X.dtype.
    k : int
        Number of neighbors to return (excluding self).
    nn_backend : {"exact","pynndescent","faiss"}
        Neighbor search backend.
    metric : {"euclidean","cosine"}
        Distance metric. For cosine we use 1 - cosine_similarity.
    Returns
    -------
    knn_idx : LongTensor (N,k)
    knn_dist : Tensor    (N,k)
    """
    if nn_backend == "exact":
        return _batched_knn_exact(X, k, row_bs=row_bs, col_bs=col_bs, metric=metric)
    elif nn_backend == "pynndescent":
        return _pynndescent_knn(X, k, metric=metric, pynndescent_kwargs=pynndescent_kwargs)
    elif nn_backend == "faiss":
        return _faiss_knn(
            X, k, metric=metric,
            use_gpu=faiss_use_gpu,
            index_kind=faiss_index_kind,
            ivf_nlist=faiss_ivf_nlist,
            ivf_nprobe=faiss_ivf_nprobe,
            pq_m=faiss_pq_m,
            hnsw_m=faiss_hnsw_m,
            hnsw_efSearch=faiss_hnsw_efSearch,
            hnsw_efConstruction=faiss_hnsw_efConstruction,
        )
    else:
        raise ValueError(f"Unknown nn_backend: {nn_backend}")

@torch.no_grad()
def batched_knn(
    X: torch.Tensor,
    k: int,
    row_bs: int = 8192,
    col_bs: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact kNN with tiling. Returns (knn_idx, knn_dist) of shape (N, k).
    """
    device, dtype = X.device, X.dtype
    N = X.shape[0]
    k_eff = min(k + 1, N)

    best_dist = torch.full((N, k_eff), float("inf"), device=device, dtype=dtype)
    best_idx = torch.full((N, k_eff), -1, device=device, dtype=torch.long)

    row_ids_full = torch.arange(N, device=device)

    def _merge_block(dist_block: torch.Tensor, col_indices: torch.Tensor, row_range: slice):
        # top-k merge without extra temporaries
        r, c = dist_block.shape
        D_old = best_dist[row_range]
        I_old = best_idx[row_range]
        # concatenate in one go
        D_cat = torch.empty((r, D_old.shape[1] + c), device=device, dtype=dtype)
        I_cat = torch.empty((r, I_old.shape[1] + c), device=device, dtype=torch.long)
        D_cat[:, : D_old.shape[1]] = D_old
        D_cat[:, D_old.shape[1] :] = dist_block
        I_cat[:, : I_old.shape[1]] = I_old
        I_cat[:, I_old.shape[1] :] = col_indices.expand(r, c)
        td, ti = torch.topk(D_cat, k=k_eff, dim=1, largest=False)
        best_dist[row_range] = td
        best_idx[row_range] = torch.gather(I_cat, 1, ti)

    for r0 in range(0, N, row_bs):
        r1 = min(r0 + row_bs, N)
        A = X[r0:r1]
        for c0 in range(0, N, col_bs):
            c1 = min(c0 + col_bs, N)
            B = X[c0:c1]
            D = torch.cdist(A, B, p=2)
            # mask diagonal when tiles overlap
            if not (c1 <= r0 or r1 <= c0):
                rows_abs = row_ids_full[r0:r1]
                cols_abs = row_ids_full[c0:c1]
                self_mask = rows_abs.view(-1, 1) == cols_abs.view(1, -1)
                D = D.masked_fill(self_mask, float("inf"))
            _merge_block(D, row_ids_full[c0:c1], slice(r0, r1))

    # paranoia: ensure self gone, cut to k
    is_self = (best_idx == row_ids_full.view(-1, 1))
    if is_self.any():
        best_dist = best_dist.masked_fill(is_self, float("inf"))
        td, ti = torch.topk(best_dist, k=k_eff, dim=1, largest=False)
        best_idx = torch.gather(best_idx, 1, ti)
        best_dist = td

    return best_idx[:, :k], best_dist[:, :k]

@torch.no_grad()
def smooth_knn_dist(
    knn_dists: torch.Tensor,
    k: int,
    local_connectivity: float = 1.0,
    n_iter: int = 64,
    bandwidth: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized smooth_knn_dist (UMAP-style).
    """
    device = knn_dists.device
    dtype = knn_dists.dtype
    N, K = knn_dists.shape
    target = math.log2(max(k, 2)) * bandwidth

    sorted_d = knn_dists.sort(dim=1).values
    nonzero_mask = sorted_d > 0
    nnz_counts = nonzero_mask.sum(dim=1)

    rhos = torch.zeros(N, dtype=dtype, device=device)
    lc_floor = int(math.floor(local_connectivity))
    lc_interp = float(local_connectivity - lc_floor)

    if lc_floor > 0:
        # base index == lc_floor-th nonzero
        idx = torch.clamp(nnz_counts - (nnz_counts - lc_floor).clamp(min=0), min=lc_floor) - 1
        base = sorted_d.gather(1, idx.view(-1, 1)).squeeze(1)
        rhos = base
        if lc_interp > 1e-5:
            idx2 = torch.clamp(idx + 1, max=K - 1)
            base2 = sorted_d.gather(1, idx2.view(-1, 1)).squeeze(1)
            rhos = rhos + lc_interp * (base2 - base)
    else:
        sd = sorted_d.clone()
        sd[~nonzero_mask] = float("inf")
        first_vals = sd.min(dim=1).values
        rhos = torch.where(torch.isfinite(first_vals), lc_interp * first_vals, torch.zeros_like(first_vals))

    lo = torch.zeros(N, dtype=dtype, device=device)
    hi = torch.full((N,), float("inf"), dtype=dtype, device=device)
    mid = torch.ones(N, dtype=dtype, device=device)

    d = knn_dists
    tiny = torch.finfo(dtype).tiny
    for _ in range(n_iter):
        denom = mid.clamp_min(tiny).unsqueeze(1)
        diff = d - rhos.unsqueeze(1)
        term = torch.where(diff > 0, torch.exp(-diff / denom), torch.ones_like(diff))
        psum = term.sum(dim=1)

        hi_mask = psum > target
        lo_mask = ~hi_mask
        hi = torch.where(hi_mask, mid, hi)
        lo = torch.where(lo_mask, mid, lo)
        mid = torch.where(torch.isinf(hi), mid * 2.0, (lo + hi) / 2.0)

        if (torch.abs(psum - target) < 1e-5).all():
            break

    mean_dist = knn_dists.mean(dim=1)
    sigmas = torch.maximum(mid, 1e-3 * mean_dist)
    return sigmas, rhos


@torch.no_grad()
def compute_membership_strengths(knn_idx, knn_dist, sigmas=None, rhos=None):
    N, K = knn_idx.shape
    rows = torch.arange(N, device=knn_idx.device).repeat_interleave(K)
    cols = knn_idx.reshape(-1)
    dists = knn_dist.reshape(-1)
    not_self = cols != rows

    if rhos is not None:
        rho = rhos.repeat_interleave(K)

        diff = dists - rho
    else:
        diff = dists

    if sigmas is not None:
        sig = sigmas.repeat_interleave(K)

        vals = torch.where((diff <= 0) | (sig <= 0), torch.ones_like(diff), torch.exp(-(diff / sig)))
    else:
        vals = torch.where((diff <= 0) | (sig <= 0), torch.ones_like(diff), torch.exp(-(diff)))

    valid = (cols >= 0) & not_self
    return rows[valid], cols[valid], vals[valid]

def _sanitize_indices(rows: torch.Tensor, cols: torch.Tensor, n: int):
    # ensure types and contiguity
    rows = rows.to(dtype=torch.long).contiguous()
    cols = cols.to(dtype=torch.long).contiguous()
    # drop invalid pairs early (device-side mask, cheap)
    valid = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
    if not bool(valid.all()):
        rows = rows[valid]
        cols = cols[valid]
    return rows, cols
def fuzzy_union_torch(rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, n: int):
    assert rows.device == cols.device == vals.device
    device = rows.device
    rows = rows.to(torch.long).contiguous()
    cols = cols.to(torch.long).contiguous()
    vals = vals.contiguous()

    # sanitize (drop -1 / OOB)
    valid = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
    if not bool(valid.all()):
        rows = rows[valid]; cols = cols[valid]; vals = vals[valid]

    # A (coalesced, sorted lexicographically)
    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n), device=device).coalesce()
    i = A.indices(); v = A.values()

    # U_add = A + A^T
    iT = torch.stack([i[1], i[0]])
    idx_sum = torch.cat([i, iT], dim=1)
    val_sum = torch.cat([v, v], dim=0)
    U_add = torch.sparse_coo_tensor(idx_sum, val_sum, (n, n), device=device).coalesce()
    iu, vu = U_add.indices(), U_add.values()  # sorted

    # Intersect A with A^T safely (no OOB indexing on GPU)
    key_A  = i[0] * n + i[1]
    key_AT = iT[0] * n + iT[1]
    sA  = torch.argsort(key_A);  key_A_s  = key_A[sA];  v_A_s  = v[sA]
    sAT = torch.argsort(key_AT); key_AT_s = key_AT[sAT]; v_AT_s = v[sAT]

    pos_in_AT = torch.searchsorted(key_AT_s, key_A_s)
    valid_pos = pos_in_AT < key_AT_s.numel()

    eq = torch.zeros_like(valid_pos, dtype=torch.bool)
    if bool(valid_pos.any()):
        eq[valid_pos] = (key_AT_s[pos_in_AT[valid_pos]] == key_A_s[valid_pos])
    match = valid_pos & eq

    if bool(match.any()):
        key_U = iu[0] * n + iu[1]  # sorted
        pos_in_U = torch.searchsorted(key_U, key_A_s[match])

        validU = pos_in_U < key_U.numel()
        eqU = torch.zeros_like(validU, dtype=torch.bool)
        if bool(validU.any()):
            eqU[validU] = (key_U[pos_in_U[validU]] == key_A_s[match][validU])
        take = validU & eqU

        if bool(take.any()):
            prod = v_A_s[match][take] * v_AT_s[pos_in_AT[match][take]]
            vu = vu.clone()
            vu[pos_in_U[take]] -= prod

    off = (iu[0] != iu[1])
    iu = iu[:, off]; vu = vu[off]
    nz = (vu != 0)
    return iu[0, nz], iu[1, nz], vu[nz]




def fuzzy_union(rows, cols, vals, n: int, binarize=False):
    if rows.is_cuda:   # keep it on device
        r, c, v = fuzzy_union_torch(rows, cols, vals, n)
        return r, c, (v > 0).to(v.dtype) if binarize else v
    coo = sp.coo_matrix((vals.detach().cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())), shape=(n, n))
    coo.sum_duplicates()
    coo.eliminate_zeros()
    U = (coo + coo.T) - coo.multiply(coo.T)
    if binarize:
        U = (U > 0).astype(float)
    U = U.tocoo()
    device = vals.device
    r = torch.from_numpy(U.row).to(device=device, dtype=torch.long)
    c = torch.from_numpy(U.col).to(device=device, dtype=torch.long)
    v = torch.from_numpy(U.data).to(device=device, dtype=vals.dtype)
    return r, c, v
