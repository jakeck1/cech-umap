from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Literal



@dataclass
class CUMAPConfig:
    n_neighbors: int = 10
    n_components: int = 2
    

    n_epochs: int =200

    learning_rate: float = 1.0
    binarize_for_spectral_init: bool = True


    init: str = "pca"  # or "spectral" or "random"
    pca_components: int = None
    knn_batch_rows: int = 2048
    epochs_print: int = 10
    random_state: int = 42

    device: Optional[str] = None
    optimizer: str = "sparse_adam"   # or 'adagrad' or 'sgd'
    adagrad_lr_decay: float = 0.0

    high_dim_cech: str = "euclidean"
    low_dim_cech: str = "euclidean"

    use_softmax: bool = True
    softmax_beta: float = 10.0
    softmax_beta_discrete: float = 10.0

    use_compile: bool = False  # optional torch.compile on hot kernels
    verbose: bool = True

    nn_backend:  Literal["exact", "pynndescent", "faiss"] = 'pynndescent'
    faiss_index_kind: str = 'ivfpq'

    ## Triplet stuff
    negative_rate_triplets: int = 1
    exclude_neighbors_in_negative_triplets: bool = False
    gamma: float = 1.0
    phi: Literal[ "exp","student"] = "student"
    exp_tau: float = 1.0

    use_triplet_weight_in_ce_loss: bool = False

    use_amp_bfloat16: bool = True
    triplet_microbatch: int = 0  # auto
    triplets_per_edge: int = 1
    proportion_uniform_triplets: float = 0.5

    anneal_learning_rate: bool = False

    


