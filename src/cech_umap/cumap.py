import math
import os
import sys
import time
import random
from dataclasses import dataclass, field,replace
from typing import Optional, Tuple, Callable, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

# ---- CUDA / matmul fast paths ----
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

_COMPILE_AVAILABLE = hasattr(torch, "compile")

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from cech_umap.utils import phi_kernel,pca_torch,auto_triplet_microbatch
from cech_umap.graph import compute_knn
from cech_umap.config import CUMAPConfig
from cech_umap.cech import triangle_smallest_enclosing_ball_radius, r_hd_discrete_fast


class CechUMAP(nn.Module):
    def __init__(
        self,
        #key hyperparameters
        n_neighbors: int = 10,
        n_components: int = 2,
        init: Literal["pca", "spectral", "random"] = "pca",
        high_dim_cech: str = "euclidean",
        low_dim_cech: str = "euclidean",
        use_softmax: bool = True,
        softmax_beta: float = 10.0,
        softmax_beta_discrete: float = 10.0,
        negative_rate_triplets: int = 1,
        exclude_neighbors_in_negative_triplets: bool = False,
        phi: Literal["exp", "student"] = "student",
        exp_tau: float = 1.0, #hyperparameter for exp dist.

        
        triplets_per_edge: int = 1,
        proportion_uniform_triplets: float = 0.5,
        
        #additional hyperparameters
        pca_components: Optional[int] = None,
        gamma: float = 1.0,
        use_triplet_weight_in_ce_loss: bool = False,

        ##nearestneighbor-computation
        nn_backend: Literal["exact", "pynndescent", "faiss"] = "pynndescent",
        faiss_index_kind: str = "ivfpq",
        

        #optimization
        n_epochs: int = 200,
        optimizer: Literal["sparse_adam", "adagrad", "sgd"] = "sparse_adam",
        adagrad_lr_decay: float = 0.0,
        learning_rate: float = 1.0,
        device: Optional[str] = None,
        anneal_learning_rate: bool = False,

        #batching
        knn_batch_rows: int = 2048,
        triplet_microbatch: int = 0,


        

        #logging
        epochs_print: int = 10,
        verbose: bool = True,
        
        #CUDA tools
        use_compile: bool = False,
        use_amp_bfloat16: bool = True,
        #this is only for spectral init
        binarize_for_spectral_init: bool = True, 
        #random state
        random_state: int = 42,

        

    ):
        """
        Initialize CechUMAP.

        Parameters
        ----------
        n_neighbors : int, optional (default=10)
            number of nearest neighbors to consider for each point
        n_components : int, optional (default=2)
            number of output dimensions
        init : Literal["pca", "spectral", "random"], optional (default="pca")
            Method to initialize the embedding
        high_dim_cech : str, optional (default="euclidean")
            whether to use embedding distance ('euclidean') or intrinsic discrte distance ('discrete') to compute triplet weights in high dim
        low_dim_cech : str, optional (default="euclidean")
            whether to use embedding distance ('euclidean') or intrinsic discrte distance ('discrete') to compute triplet weights in low dim
        use_softmax : bool, optional (default=True)
            Whether to use softmax instead of hard max for computing triangle weights
        softmax_beta : float, optional (default=10.0)
            Beta of the softmax function in euclidean case
        softmax_beta_discrete : float, optional (default=10.0)
            Beta of the softmax function for discrete case
        negative_rate_triplets : int, optional (default=1)
            Number of negative triples per positive triples
        exclude_neighbors_in_negative_triplets : bool, optional (default=False)
            Whether to exclude neighbors
            in the negative samples
        phi : Literal["exp", "student"], optional (default="student")
            Distribution function for CE objective
        exp_tau : float, optional (default=1.0)
            Hyperparameter for the exp distribution (only used if phi="exp")
        triplets_per_edge : int, optional (default=1)
            Number of triplets to sample per edge
        proportion_uniform_triplets : float, optional (default=0.5)
            Proportion of uniform triplets versus semi-local samples (semi-local samples have two nearest neighbors and one non-neighbor, 
            experiment with this hyperparameter for different results)

        pca_components : Optional[int], optional (default=None)
            Number of PCA components to use

        gamma : float, optional (default=1.0)
            Hyperparameter to exaggerate negative triplet part of loss (i.e. loss is ce_loss (positive ) + gamma* ce_loss(negative))

        use_triplet_weight_in_ce_loss: bool  (default False)
            Controls whether in the negative triplet part of the CE loss, weights in the high dimensional space are used or not 
            (latter is mimicking UMAP and is default.) (This may produce smoother, less disconnected embeddings with appropriate settings)

        nn_backend : Literal["exact", "pynndescent", "faiss"], optional (default="pynndescent")
            Method to use for nearest neighbor computation

        faiss_index_kind : str, optional (default="ivfpq")
            Index kind to use for Faiss (only used if nn_backend="faiss")

        n_epochs : int, optional (default=200)
            Number of epochs to run

        optimizer : Literal["sparse_adam", "adagrad", "sgd"], optional (default="sparse_adam")
            Method to use for optimization

        adagrad_lr_decay : float, optional (default=0.0)
            Learning rate decay for Adagrad

        learning_rate : float, optional (default=1.0)
            Initial learning rate

        device : Optional[str], optional (default=None)
            Device to use for computation

        anneal_learning_rate : bool, optional (default=False)
            Whether to manually anneal learning rate similar to UMAP

        use_compile : bool, optional (default=False)
            Whether to use torch.compile

        use_amp_bfloat16 : bool, optional (default=True)
            Whether to use AMP with bfloat16

        binarize_for_spectral_init : bool, optional (default=True)
            Whether to binarize for spectral init (only used if init="spectral")

        random_state : int, optional (default=42)
            Random state to use

        verbose : bool, optional (default=True)
            Whether to print verbose output

        epochs_print : int, optional (default=10)
            Number of epochs to print (only used if verbose=True)

        knn_batch_rows : int, optional (default=2048)
            Batch size to use for nearest neighbor computation

        triplet_microbatch : int, optional (default=0)
            Batch size to use for triplet computation, default 0 is automatic microbatching
       """
        super().__init__()

        self.cfg = CUMAPConfig(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            binarize_for_spectral_init=binarize_for_spectral_init,
            init=init,
            pca_components=pca_components,
            knn_batch_rows=knn_batch_rows,
            epochs_print=epochs_print,
            random_state=random_state,
            device=device,
            optimizer=optimizer,
            adagrad_lr_decay=adagrad_lr_decay,
            high_dim_cech=high_dim_cech,
            low_dim_cech=low_dim_cech,
            use_softmax=use_softmax,
            softmax_beta=softmax_beta,
            softmax_beta_discrete=softmax_beta_discrete,
            use_compile=use_compile,
            verbose=verbose,
            nn_backend=nn_backend,
            faiss_index_kind=faiss_index_kind,
            negative_rate_triplets=negative_rate_triplets,
            exclude_neighbors_in_negative_triplets=exclude_neighbors_in_negative_triplets,
            gamma=gamma,
            phi=phi,
            exp_tau=exp_tau,
            use_amp_bfloat16=use_amp_bfloat16,
            triplet_microbatch=triplet_microbatch,
            triplets_per_edge=triplets_per_edge,
            proportion_uniform_triplets=proportion_uniform_triplets,
            anneal_learning_rate = anneal_learning_rate,
            use_triplet_weight_in_ce_loss= use_triplet_weight_in_ce_loss
        )



        self.device_ = torch.device(self.cfg.device) if self.cfg.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_: Optional[torch.Tensor] = None
        self.use_bf16=False
        self._r_hd_discrete = r_hd_discrete_fast
        if self.cfg.use_compile and _COMPILE_AVAILABLE:
            try:
                self._r_hd_discrete = torch.compile(r_hd_discrete_fast, fullgraph=False, mode="max-autotune")
            except Exception:
                self._r_hd_discrete = r_hd_discrete_fast

    # ---------- spectral init ----------
    def _spectral_init(self, n: int, head: torch.Tensor, tail: torch.Tensor, weight: torch.Tensor, n_components: int) -> torch.Tensor:
        """
        Spectral initialization for kNN graph.
        If the graph is empty, returns a random tensor.
        Otherwise, computes the k smallest eigenvectors of the symmetric normalized adjacency matrix Laplacian.
        The embedding is then centered and scaled similarly to PCA initialization.
        """
        dev = self.device_
        dtype = torch.float32
        if head.numel() == 0:
            return torch.randn(n, n_components, device=dev, dtype=dtype) * 1e-3

        # Build symmetric normalized adjacency
        idx = torch.stack([head, tail], dim=0)
        val = weight.to(dtype)
        W = torch.sparse_coo_tensor(idx, val, size=(n, n), device=dev, dtype=dtype).coalesce()
        idxT = torch.stack([W.indices()[1], W.indices()[0]], dim=0)
        Wsym = torch.sparse_coo_tensor(torch.cat([W.indices(), idxT], dim=1),
                                       torch.cat([W.values(), W.values()], dim=0),
                                       size=(n, n), device=dev, dtype=dtype).coalesce()
        deg = torch.sparse.sum(Wsym, dim=1).to_dense().clamp_min(1e-12)
        dinv_sqrt = deg.rsqrt()
        i = Wsym.indices()
        v = Wsym.values() * dinv_sqrt[i[0]] * dinv_sqrt[i[1]]

        # Try SciPy eigsh on CPU
        A = sp.coo_matrix((v.detach().cpu().numpy(), (i[0].cpu().numpy(), i[1].cpu().numpy())), shape=(n, n)).tocsr()
        L = (sp.eye(n, format="csr") - A).tocsr()
        k = min(n_components + 1, max(2, n))
        evals, evecs = eigsh(L, k=k, which="SM", tol=1e-5)
        Z = torch.from_numpy(evecs[:, 1 : n_components + 1]).to(dev, dtype)
        
        # center + scale ~ like PCA init
        Z = Z - Z.mean(dim=0, keepdim=True)
        std = Z.std(dim=0, keepdim=True).clamp_min(1e-6)
        Z = 10.0 * (Z / std)
        return Z

    def _prepare_init(self, X: torch.Tensor, n: int, head=None, tail=None, weight=None) -> torch.Tensor:
        if self.cfg.init == "random":
            Z = torch.empty((n, self.cfg.n_components), device=self.device_, dtype=torch.float32)
            Z.uniform_(-10.0, 10.0)
            return Z
        if self.cfg.init == "spectral" and head is not None:
            return self._spectral_init(n, head, tail, weight, self.cfg.n_components)
        # PCA
        Z = pca_torch(X, n_components=self.cfg.n_components)
        mn = Z.min(dim=0, keepdim=True).values
        mx = Z.max(dim=0, keepdim=True).values
        Z = 10.0 * (Z - mn) / torch.clamp(mx - mn, min=1e-8)
        Z = (Z + torch.randn_like(Z) * 1e-4).to(torch.float32)
        return Z


    def _triplet_ce_loss(
        self,
        emb: nn.Embedding,
        trip_i: torch.Tensor,
        trip_j: torch.Tensor,
        trip_k: torch.Tensor,
        w_trip: torch.Tensor,
        
        neg_i: Optional[torch.Tensor],
        neg_j: Optional[torch.Tensor],
        neg_k: Optional[torch.Tensor],
        knn_idx: torch.Tensor,
        gamma_trip: float,
        phi_kind: str,
        exp_tau: float,
        eps: float = 1e-6,
        w_neg_trip = None

        
    ) -> torch.Tensor:

        """
        Compute the triplet CE loss, given positive triplets, negative triplets and weights of positive triplets.

        Parameters
        ----------
        emb : nn.Embedding
            The embedding layer.
        trip_i, trip_j, trip_k : torch.Tensor
            positive triplet indices
        w_trip : torch.Tensor
            weights of positive triplets.
        neg_i, neg_j, neg_k : Optional[torch.Tensor]
            negative triplet indices
        knn_idx : torch.Tensor
            The kNN indices of positive triplets
        gamma_trip : float
            The weighting factor of the negative triplets.
        phi_kind : str
            type of distribution function phi
        exp_tau : float
            The temperature parameter for phi = exp
        eps : float, optional
            min value for phi, defaults to 1e-6.
        
        Returns
        -------
        loss : torch.Tensor
            The computed triplet CE loss.
    """
        Yi, Yj, Yk = emb(trip_i), emb(trip_j), emb(trip_k)

        if self.cfg.low_dim_cech == "euclidean":
            dij = (Yi - Yj).norm(dim=1)
            dik = (Yi - Yk).norm(dim=1)
            djk = (Yj - Yk).norm(dim=1)
            R = triangle_smallest_enclosing_ball_radius(
                dij,
                dik,
                djk,
                softmax=self.cfg.use_softmax,
                softmax_beta=self.cfg.softmax_beta,
            )

        elif self.cfg.low_dim_cech == "discrete":
            # Candidate centers: union of kNNs of the three anchors (duplicates are fine)
            cand = torch.cat([knn_idx[trip_i], knn_idx[trip_j], knn_idx[trip_k]], dim=1)  # [B, 3K]
            cand_Y = emb(cand)  # [B, 3K, d_low]

            # Distances from each anchor to every candidate
            Di = (Yi.unsqueeze(1) - cand_Y).norm(dim=-1)  # [B, 3K]
            Dj = (Yj.unsqueeze(1) - cand_Y).norm(dim=-1)  # [B, 3K]
            Dk = (Yk.unsqueeze(1) - cand_Y).norm(dim=-1)  # [B, 3K]

            D = torch.stack([Di, Dj, Dk], dim=1)  # [B, 3, 3K]
            beta = float(self.cfg.softmax_beta_discrete)
            max_over_anchors = torch.logsumexp(beta * D, dim=1) / beta         # [B, 3K]
            R = -torch.logsumexp(-beta * max_over_anchors, dim=1) / beta       # [B]
        else:
            raise ValueError(f"Unknown cfg.low_dim_cech: {self.cfg.low_dim_cech}")

        r_hat = 2.0 * R
        q = phi_kernel(r_hat, phi_kind, tau=exp_tau)
          
        N = w_trip.shape[0]
        pos_loss = (w_trip * (-torch.log(torch.clamp(q, min=eps)))).sum()

        if neg_i is not None and neg_i.numel() > 0:
            Yin, Yjn, Ykn = emb(neg_i), emb(neg_j), emb(neg_k)
            dij_n = (Yin - Yjn).norm(dim=1)
            dik_n = (Yin - Ykn).norm(dim=1)
            djk_n = (Yjn - Ykn).norm(dim=1)
            Rn = triangle_smallest_enclosing_ball_radius(
                dij_n,
                dik_n,
                djk_n,
                softmax=self.cfg.use_softmax,
                softmax_beta=self.cfg.softmax_beta,
            )
            rhat_n = 2.0 * Rn
            qn = phi_kernel(rhat_n, phi_kind, tau=exp_tau)
            if self.cfg.use_triplet_weight_in_ce_loss:
                neg_loss = gamma_trip * (w_neg_trip*(-torch.log(torch.clamp(1.0 - qn, min=eps)))).sum()

            else:

                neg_loss = gamma_trip * (-torch.log(torch.clamp(1.0 - qn, min=eps))).sum()
        else:
            neg_loss = torch.tensor(0.0, device=self.device_, dtype=torch.float32)
        return (pos_loss + neg_loss) / max(N, 1)

    @torch.no_grad()
    def _build_graph(self, X: torch.Tensor):
        """
        Build the graph given input data X.

        Parameters
        ----------
        X : (N,D) torch.Tensor
            Input data.

        Returns
        -------
        head : (NK) torch.Tensor
            Head of each edge.
        tail : (NK) torch.Tensor
            Tail of each edge.
        weight : (NK) torch.Tensor
            Weight of each edge.
        knn_idx : (N,K) torch.Tensor
            Indices of the K nearest neighbors for each point.
        X_knn : (N,D) torch.Tensor
            Data used for computing nearest neighbors.
        knn_dist : (N,K) torch.Tensor
            Distances of the K nearest neighbors for each point.

        """
        cfg = self.cfg
        N, D = X.shape
        X_knn = X
        if cfg.pca_components and cfg.pca_components < D:
            X_knn = pca_torch(X, n_components=cfg.pca_components)

        
        knn_idx, knn_dist = compute_knn(
            X_knn,
            cfg.n_neighbors,
            nn_backend=cfg.nn_backend,
            faiss_index_kind=cfg.faiss_index_kind,
            row_bs=cfg.knn_batch_rows,
            col_bs=cfg.knn_batch_rows,
        )
        
        N, K = knn_idx.shape
        rows = torch.arange(N, device=knn_idx.device).repeat_interleave(K)
        cols = knn_idx.reshape(-1)

        if cfg.binarize_for_spectral_init:
            weight = torch.ones_like(knn_dist.reshape(-1))
        else:
            weight = torch.exp(-knn_dist.reshape(-1)/knn_dist.max())



        head, tail = rows, cols

        return head, tail, weight, knn_idx, X_knn, knn_dist

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
    
        """
        Fit the CUMAP model and transform input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        emb : np.ndarray
            Transformed data.
        """
        cfg = self.cfg
        dev = self.device_
        

        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        
        N, D = X.shape

        


        head, tail, weight, knn_idx, X_knn, knn_dist = self._build_graph(X)
        if self.cfg.verbose:
            print("Initialized the knn-graph.")
        # sanity: KNN indices should be in [0, N-1]
        if not bool((knn_idx >= 0).all() and (knn_idx < N).all()):
            bad = (~((knn_idx >= 0) & (knn_idx < N)))
            num_bad = int(bad.sum().item())
            first_bad = tuple(torch.nonzero(bad, as_tuple=True)[0][:10].tolist())
            raise RuntimeError(f"KNN produced {num_bad} out-of-range indices; examples {first_bad}")

        max_d = knn_dist.max()
        
        # init (allow spectral using the already-built graph)
        init = self._prepare_init(X_knn, N, head=head, tail=tail, weight=weight)
        emb = nn.Embedding(
            num_embeddings=N,
            embedding_dim=cfg.n_components,
            sparse=True,
            _weight=init.clone(),
        ).to(dev)
        if self.cfg.verbose:
            print(f"Constructed initial {self.cfg.init} embedding.")

        # optimizer
        opt_name = cfg.optimizer.lower()
        if opt_name == "adagrad":
            opt = torch.optim.Adagrad(emb.parameters(), lr=cfg.learning_rate, lr_decay=cfg.adagrad_lr_decay)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(emb.parameters(), lr=cfg.learning_rate)
        else:
            opt = torch.optim.SparseAdam(emb.parameters(), lr=cfg.learning_rate)

        # schedule
        n_epochs = cfg.n_epochs
        self.arange_cache = None

        if weight.numel() == 0:
            self.embedding_ = emb.weight.detach().clone()
            return self.embedding_.cpu().numpy()

        
        rng = torch.Generator(device=dev)
        rng.manual_seed(cfg.random_state)
        # triplet knobs
        gamma_trip = float(cfg.gamma)
        k_per_edge_trip = int(getattr(cfg, "triplets_per_edge"))
        self.use_bf16 = bool(getattr(cfg, "use_amp_bfloat16", True))

        # pre-allocate a small tmp buffer reused for due-mask fallback
        lr0 = float(cfg.learning_rate)
        print_every = max(1, int(cfg.epochs_print))
        if self.cfg.verbose:
                    print("Optimizing embedding.")

        for epoch in range(1, n_epochs + 1):
            # linear LR decay (α is applied as a scalar to the loss to keep SparseAdam happy)
            
            if cfg.anneal_learning_rate:
                alpha = lr0 * (1.0 - (epoch - 1) / max(1, n_epochs))
            else:
                alpha = 1.0
            i = head
            j = tail
            
            B = i.shape[0]
            if B == 0:
                continue



            opt.zero_grad(set_to_none=True)

            total_loss = 0.0

            # ---- triplet loss driven by EPS, single loop & fast radius ----
            ti_all = i.repeat_interleave(k_per_edge_trip)
            tj_all = j.repeat_interleave(k_per_edge_trip)
            B_all = ti_all.numel()

            K = int(knn_idx.shape[1])
            Dhd = int(X_knn.shape[1])
            triplet_microbatch = int(getattr(cfg, "triplet_microbatch", 0)) or auto_triplet_microbatch(
                K, Dhd, target_util_gb=8.0, bf16=self.use_bf16
            )

            total_ce_trip = 0.0

            for b0 in range(0, B_all, triplet_microbatch):
                b1 = min(b0 + triplet_microbatch, B_all)
                ti_s = ti_all[b0:b1]
                tj_s = tj_all[b0:b1]
                B_s = ti_s.shape[0]

                # --- sample positive k from N(i) ∪ N(j) \ {i,j}
                with torch.no_grad():
                    tk_s = self.sample_pos_triplet_vertices(ti_s,tj_s,knn_idx,B_s,N,dev,rng)
                    # --- high-d radius (single pass / microbatch)
                   
                    r_hd_s = self.cech_weights(X_knn,knn_idx,ti_s,tj_s,tk_s)
                    
                    # weight that decays with r_hd
                    w_trip_s = torch.exp(-2.0 * r_hd_s / max(max_d, torch.tensor(1e-9, device=dev)))

                    ti_n,tj_n,tk_n = self.sample_neg_triplets( ti_s, tj_s, knn_idx, dev, B_s, N, rng)
                
                    r_hd_neg = self.cech_weights(X_knn,knn_idx,ti_n,tj_n,tk_n)
                    
                    # weight that decays with r_hd
                    w_neg_trip = torch.exp(-2.0 * r_hd_neg / max(max_d, torch.tensor(1e-9, device=dev)))

                ce_trip_s = self._triplet_ce_loss(
                    emb,
                    ti_s,
                    tj_s,
                    tk_s,
                    w_trip_s,
                    ti_n,
                    tj_n,
                    tk_n,
                    knn_idx,
                    gamma_trip=gamma_trip,
                    phi_kind=cfg.phi,
                    exp_tau=cfg.exp_tau,
                    w_neg_trip=w_neg_trip
                )
                total_ce_trip = total_ce_trip + ce_trip_s

            total_loss = total_loss + total_ce_trip

            
            # backward + step (scale loss by α)
            (alpha *total_loss).backward()
            #alpha*total_loss.backward()

            if not emb.sparse:
                torch.nn.utils.clip_grad_norm_(emb.parameters(), max_norm=5.0)

            opt.step()

            if self.cfg.verbose:
                if (epoch % print_every) == 0 or epoch == 1 or epoch == n_epochs:
                    with torch.no_grad():
                        Yi = emb(i)
                        Yj = emb(j)
                        md = (Yi - Yj).norm(dim=1).mean().item()
                        tloss = float(total_loss.item())
                    print(f"[epoch {epoch:4d}/{n_epochs}] || loss={tloss:.4f}")

        self.embedding_ = emb.weight.detach().clone()
        return self.embedding_.cpu().numpy()
    


    @torch.no_grad()
    def cech_weights(self,X_knn,knn_idx,ti_s,tj_s,tk_s):
        """
        Compute weights of triplets according to their radius in the cech complex.

        Parameters
        ----------
        X_knn : Tensor (N,D)
            Input data.
        knn_idx : Tensor (N,K)
            Indices of k-nearest neighbors for each data point.
        ti_s : Tensor (B)
            first indices for triplets.
        tj_s : Tensor (B)
            second indices for triplets.
        tk_s : Tensor (B)
            third indices for triplets.

        Returns
        -------
        r_hd_s : Tensor (B)
            The weights for each triplet.
        """
        if self.cfg.high_dim_cech == 'discrete':
            beta_hd =float(getattr(self.cfg, "softmax_beta_discrete",1.0))
            r_hd_s = self._r_hd_discrete(
                X_knn,
                ti_s,
                tj_s,
                tk_s,
                knn_idx,
                beta=beta_hd,
                use_bf16=self.use_bf16,
            )
            return r_hd_s
        elif self.cfg.high_dim_cech == 'euclidean':
            Xi, Xj, Xk = X_knn[ti_s], X_knn[tj_s], X_knn[tk_s]
            dij_hd = (Xi - Xj).norm(dim=1)
            dik_hd = (Xi - Xk).norm(dim=1)
            djk_hd = (Xj - Xk).norm(dim=1)
            use_soft = bool(getattr(self.cfg, "use_softmax", True))
            softmax_beta = float(getattr(self.cfg, "softmax_beta",1.0))
            R_hd = triangle_smallest_enclosing_ball_radius(
                dij_hd, dik_hd, djk_hd, softmax=use_soft, softmax_beta=softmax_beta
            )
            r_hd_s = 2.0 * R_hd

            return r_hd_s
        else: raise ValueError('Not a valid option for high_dim_cech, valid options are "discrete","euclidean" ')

    @torch.no_grad()
    def sample_pos_triplet_vertices(self,ti_s,tj_s,knn_idx,B_s,N,dev,rng):
        cand = torch.cat([knn_idx[ti_s], knn_idx[tj_s]], dim=1)  # [B_s, 2K]
        if self.arange_cache is None or self.arange_cache.numel() != B_s:
            self.arange_cache = torch.arange(B_s, device=dev)
        valid = (cand != ti_s.unsqueeze(1)) & (cand != tj_s.unsqueeze(1))
        cand = torch.where(valid, cand, torch.full_like(cand, -1))

        # Gumbel-max sampling w/ masked -inf
        logits = torch.zeros_like(cand, dtype=torch.float32, device=dev)
        logits[cand == -1] = float("-inf")
        g = -torch.log(-torch.log(torch.rand_like(logits)))
        col = (logits + g).argmax(dim=1)
        tk_s = cand[self.arange_cache, col]

        # fallback if a row was all invalid
        empty = (tk_s == -1)
        if empty.any():
            ke = torch.randint(0, N, (int(empty.sum()),), device=dev, generator=rng)
            bad = (ke == ti_s[empty]) | (ke == tj_s[empty])
            if bad.any():
                ke[bad] = (ke[bad] + 1) % N
            tk_s[empty] = ke

        return tk_s

        
        
    @torch.no_grad()
    def sample_neg_triplets(self, ti_s, tj_s, knn_idx, dev, B_s, N, rng):
       
        """
        Sample negative triplets for a given batch of positive triplets.

        Parameters
        ----------
        ti_s : Tensor (B_s)
            Indices of the first vertex of the positive triplets.
        tj_s : Tensor (B_s)
            Indices of the second vertex of the positive triplets.
        knn_idx : Tensor (B_s, K)
            Indices of the k-NN neighbors of the positive triplets.
        dev : torch.device
            The device to use for the sampling.
        B_s : int
            The batch size of the positive triplets.
        N : int
            The number of vertices in the graph.
        rng : torch.Generator
            A random number generator to use for the sampling.

        Returns
        -------
        ti_n, tj_n, tk_n : Tensor (total_neg)
            The indices of the negative triplets.
        """
        
        cfg = self.cfg
        neg_rate_trip = cfg.negative_rate_triplets
        if neg_rate_trip <= 0:
            return None, None, None

        total_neg = int(B_s * neg_rate_trip)
        if total_neg == 0:
            return None, None, None

        # clamp + choose rounding rule for better long-run proportionality
        prop = float(getattr(cfg, "proportion_uniform_triplets", 0.0))
        prop = max(0.0, min(1.0, prop))
        n_uni = int(round(prop * total_neg))
        n_knn = total_neg - n_uni

        exclude_neighbors = bool(getattr(cfg, "exclude_neighbors_in_negative_triplets", False))

        out_ti, out_tj, out_tk = [], [], []

        # ---------------------------- helper: quick repair ----------------------------
        def repair_third_vertex(tk, ti, tj, cand_pos=None):
            
            bad = (tk == ti) | (tk == tj)
            if cand_pos is not None:
                bad |= (tk.unsqueeze(1) == cand_pos).any(dim=1)

            # two random repair passes
            for _ in range(2):
                if not bad.any():
                    break
                repl = torch.randint(0, N, (int(bad.sum()),), device=dev, generator=rng)
                tk[bad] = repl
                # recompute bad only for locations we repaired
                new_bad = (tk[bad] == ti[bad]) | (tk[bad] == tj[bad])
                if cand_pos is not None:
                    new_bad |= (tk[bad].unsqueeze(1) == cand_pos[bad]).any(dim=1)
                tmp = torch.zeros_like(bad)
                tmp[bad] = new_bad
                bad = tmp

            if bad.any():
                tk[bad] = (tk[bad] + 1) % N

            return tk

        # ---------------------------- part A: uniform-edge negatives ----------------------------
        if n_uni > 0:
            # pick anchors from current positive sources to match your previous distributional choice
            base_idx = torch.randint(0, B_s, (n_uni,), device=dev, generator=rng)
            ti_u = ti_s[base_idx]

            # sample the negative endpoint j uniformly, excluding self and KNN( ti_u )
            tj_u = torch.randint(0, N, (n_uni,), device=dev, generator=rng)
            neigh_u = knn_idx[ti_u]  # [n_uni, K]
            bad_edge = (tj_u == ti_u) | (tj_u.unsqueeze(1) == neigh_u).any(dim=1)
            if bad_edge.any():
                # one random resample for bad positions
                j2 = torch.randint(0, N, (int(bad_edge.sum()),), device=dev, generator=rng)
                tj_u[bad_edge] = j2
                # final quick pass: if still colliding, bump by 1 mod N
                bad_edge2 = (tj_u[bad_edge] == ti_u[bad_edge]) | (tj_u[bad_edge].unsqueeze(1) == neigh_u[bad_edge]).any(dim=1)
                if bad_edge2.any():
                    idx2 = torch.nonzero(bad_edge, as_tuple=False).squeeze(1)[bad_edge2]
                    tj_u[idx2] = (tj_u[idx2] + 1) % N

            # third vertex: uniform with repairs
            tk_u = torch.randint(0, N, (n_uni,), device=dev, generator=rng)
            cand_pos_u = None
            if exclude_neighbors:
                cand_pos_u = torch.cat([knn_idx[ti_u], knn_idx[tj_u]], dim=1)  # [n_uni, 2K]
            tk_u = repair_third_vertex(tk_u, ti_u, tj_u, cand_pos_u)

            out_ti.append(ti_u); out_tj.append(tj_u); out_tk.append(tk_u)

        # ---------------------------- part B: knn-excluded negatives ----------------------------
        if n_knn > 0:
            base_idx = torch.randint(0, B_s, (n_knn,), device=dev, generator=rng)
            ti_k = ti_s[base_idx]
            tj_k = tj_s[base_idx]
            cand_rep = None
            if exclude_neighbors:

                cand_rep = torch.cat([knn_idx[ti_k], knn_idx[tj_k]], dim=1)  # [n_knn, 2K]
            tk_k = torch.randint(0, N, (n_knn,), device=dev, generator=rng)
            tk_k = repair_third_vertex(tk_k, ti_k, tj_k, cand_rep)

            out_ti.append(ti_k); out_tj.append(tj_k); out_tk.append(tk_k)

        # ---------------------------- merge + shuffle ----------------------------
        if len(out_ti) == 1:
            ti_n, tj_n, tk_n = out_ti[0], out_tj[0], out_tk[0]
        else:
            ti_n = torch.cat(out_ti, dim=0)
            tj_n = torch.cat(out_tj, dim=0)
            tk_n = torch.cat(out_tk, dim=0)
            perm = torch.randperm(ti_n.numel(), device=dev, generator=rng)
            ti_n = ti_n[perm]; tj_n = tj_n[perm]; tk_n = tk_n[perm]

        return ti_n, tj_n, tk_n

                    