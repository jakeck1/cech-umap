

import torch

def triangle_smallest_enclosing_ball_radius(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1e-9,
    softmax: bool = True,
    softmax_beta: float = 1.0,
) -> torch.Tensor:
    sides = torch.stack([a, b, c], dim=-1)
    sides_sorted, _ = sides.sort(dim=-1)
    x, y, z = sides_sorted[..., 0], sides_sorted[..., 1], sides_sorted[..., 2]
    obtuse = z * z >= x * x + y * y - eps
    s = (x + y + z) / 2.0
    area = torch.sqrt(torch.clamp(s * (s - x) * (s - y) * (s - z), min=eps))
    R_circ = (x * y * z) / (4.0 * area)

    if softmax:
        smoothed_max = torch.logsumexp(softmax_beta * sides_sorted, dim=-1) / softmax_beta
        return torch.where(obtuse, smoothed_max / 2.0, R_circ)
    else:
        return torch.where(obtuse, z / 2.0, R_circ)




@torch.no_grad()
def r_hd_discrete_fast(
    X_all: torch.Tensor,                    # [N, D]  (== X_knn)
    ti: torch.Tensor, tj: torch.Tensor, tk: torch.Tensor,  # [B]
    knn_idx: torch.Tensor,                  # [N, K]
    beta: float = 1.0,
    use_bf16: bool = True
) -> torch.Tensor:                           # returns r_hd: [B]
    """
    Smooth min_y max{ d(i,y), d(j,y), d(k,y) } over y in N(i)∪N(j) (∪N(k) if include_k..).
    Single pass, fused via batched matmul; no candidate chunk loop.
    """
    device = X_all.device
    D = X_all.shape[1]
    # anchors
    Xi, Xj, Xk = X_all[ti], X_all[tj], X_all[tk]           # [B, D]
    A = torch.stack([Xi, Xj, Xk], dim=1)                   # [B, 3, D]
    A2 = (A * A).sum(dim=-1, keepdim=True)                 # [B, 3, 1]

    # candidates: union of kNNs
    cand = torch.cat([knn_idx[ti], knn_idx[tj], knn_idx[tk]], dim=1)   # [B, 3K]
    B, C = cand.shape  # C is small (<= ~96)

    # gather candidate coords once
    Y = X_all.index_select(0, cand.view(-1)).view(B, C, D)  # [B, C, D]
    Y2 = (Y * Y).sum(dim=-1).unsqueeze(1)                   # [B, 1, C]

    # distances via ||a - y||^2 = ||a||^2 + ||y||^2 - 2 a·y
    if use_bf16 and X_all.is_cuda and torch.cuda.is_bf16_supported():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            S = torch.einsum("bmd,bnd->bmn", A, Y)          # [B, 3, C]
            dist = torch.sqrt(torch.clamp(A2 + Y2 - 2 * S, min=1e-12))  # [B, 3, C]
            max_over_anchors = torch.logsumexp(beta * dist, dim=1) / beta  # [B, C]
            r = -torch.logsumexp(-beta * max_over_anchors, dim=1) / beta   # [B]
            return r.to(X_all.dtype)
    else:
        S = torch.einsum("bmd,bnd->bmn", A, Y)
        dist = torch.sqrt(torch.clamp(A2 + Y2 - 2 * S, min=1e-12))
        max_over_anchors = torch.logsumexp(beta * dist, dim=1) / beta
        return -torch.logsumexp(-beta * max_over_anchors, dim=1) / beta

