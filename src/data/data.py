import torch, torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.datasets import fetch_openml
import numpy as np
import torch

import os
import urllib.request
import zipfile
import numpy as np
from PIL import Image

import os, io, re, zipfile, urllib.request
from glob import glob
from sklearn.utils import check_random_state
# ==== Common imports ====

from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.datasets import (
    fetch_openml,
    fetch_olivetti_faces,
    load_digits,
    fetch_rcv1,
)
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans

COIL20_URLS = [
    # Processed (cropped to 128x128), current CAVE path
    "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
    # Unprocessed (background present) – still fine; we’ll resize to 128x128
    "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip",
]

def _download(url: str, dst_zip: str):
    os.makedirs(os.path.dirname(dst_zip), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst_zip, "wb") as f:
        f.write(r.read())

def _extract(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

def _read_image(path: str, resize_to=(128, 128)):
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    if resize_to is not None and im.size != resize_to:
        im = im.resize(resize_to, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def _label_from_fname(fname: str) -> int:
    # Works for patterns like: obj7__10.pgm, obj7_10.pgm, obj7__10.png, etc.
    m = re.search(r"obj\s*(\d+)", fname, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {fname}")
    return int(m.group(1)) - 1  # make labels 0..19

def load_coil20(dest="./data/coil20", flatten=True):
    """
    Returns:
      X: float32 array of shape [1440, 16384] (if flatten=True) or [1440, 128, 128]
      y: int64 labels in [0..19]
    """
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "coil20.zip")
    extracted_dir = os.path.join(dest, "extracted")

    # download (try processed then unprocessed)
    if not os.path.exists(zip_path):
        last_err = None
        for url in COIL20_URLS:
            try:
                print(f"Downloading {url} ...")
                _download(url, zip_path)
                break
            except Exception as e:
                last_err = e
                if os.path.exists(zip_path):
                    try: os.remove(zip_path)
                    except Exception: pass
                zip_path = os.path.join(dest, os.path.basename(url))
        else:
            raise RuntimeError(f"Failed to download COIL-20: {last_err}")

    # extract (idempotent)
    if not os.path.isdir(extracted_dir) or not os.listdir(extracted_dir):
        _extract(zip_path, extracted_dir)

    # find images (PGM/PNG), recurse (archives differ in folder names)
    patterns = ["**/*.pgm", "**/*.png", "**/*.ppm", "**/*.jpg", "**/*.jpeg"]
    files = []
    for pat in patterns:
        files.extend(glob(os.path.join(extracted_dir, pat), recursive=True))
    if not files:
        raise RuntimeError(f"No images found under {extracted_dir}")

    files = sorted(files)
    X_list, y_list = [], []
    for p in files:
        try:
            img = _read_image(p, resize_to=(128, 128))
            y = _label_from_fname(os.path.basename(p))
            X_list.append(img if not flatten else img.reshape(-1))
            y_list.append(y)
        except Exception as e:
            # Skip non-image files (e.g., README) or unparsable names
            continue

    if not X_list:
        raise RuntimeError("Found files but could not parse any COIL-20 images.")

    X = np.stack(X_list, 0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # COIL-20 should have 1440 images; if unprocessed set was used, counts can differ; that’s fine.
    return X, y



def load_20ng_tfidf(subset="train", max_features=200000, min_df=2, n_components=128, random_state=0):
    data = fetch_20newsgroups(subset=subset, remove=("headers","footers","quotes"))
    vec = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)
    X_tf = vec.fit_transform(data.data)                    # sparse CSR
    X = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(X_tf)
    X = normalize(X, norm="l2").astype(np.float32)         # dense (N, n_components)
    y = np.asarray(data.target)
    return X, y

def load_20ng_raw(subset="train", max_features=20000, min_df=2, random_state=0):

    data = fetch_20newsgroups(subset=subset, remove=("headers","footers","quotes"))
    vec = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)
    X_tf = vec.fit_transform(data.data)                    # sparse CSR
    y = np.asarray(data.target)
    return X_tf,y



def load_pbmc3k(n_hvg=2000, n_pcs=50, do_scale=False, resolution=1.0, random_state=0):
    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    if do_scale:
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=random_state)

    # Prefer Louvain; fall back to Leiden if Louvain isn’t available
    try:
        sc.tl.louvain(adata, resolution=resolution, random_state=random_state, key_added="clusters")
    except Exception:
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added="clusters")

    X = adata.obsm["X_pca"].astype(np.float32)
    cats = adata.obs["clusters"]
    if hasattr(cats, "cat"):
        y = cats.cat.codes.to_numpy(dtype=np.int32)
        names = list(cats.cat.categories)
    else:
        y = cats.to_numpy()
        names = None
    return X, y # names maps integer codes -> cluster label strings



def load_mnist(flatten=True, train=True):
    tfm = transforms.ToTensor()
    ds = tv.datasets.MNIST(root="./data", train=train, download=True, transform=tfm)
    X = ds.data.float().view(len(ds), -1) / 255.0 if flatten else ds.data.float()/255.0
    y = ds.targets.clone()

    return X.numpy(), y.numpy()

def load_fashion_mnist(flatten=True, train=True):
    tfm = transforms.ToTensor()
    ds = tv.datasets.FashionMNIST(root="./data", train=train, download=True, transform=tfm)
    X = ds.data.float().view(len(ds), -1) / 255.0 if flatten else ds.data.float()/255.0
    y = ds.targets.clone()

    return X.numpy(), y.numpy()

@torch.no_grad()
def load_cifar10_resnet18_feats(split="train", batch_size=256, device="cuda"):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    train = (split == "train")
    ds = tv.datasets.CIFAR10(root="./data", train=train, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    feat = torch.empty(len(ds), 512, device=device)
    off = 0
    for x, _ in dl:
        x = x.to(device)
        h = model.forward(torch.nn.functional.interpolate(x, size=(224,224)))
        extractor = create_feature_extractor(model, return_nodes={"avgpool":"feat"}).to(device)
        f = extractor(x)["feat"].squeeze(-1).squeeze(-1)
        n = f.size(0)
        feat[off:off+n] = f
        off += n
    X = feat.float().cpu().numpy()

    y = torch.tensor(ds.targets).numpy()
    return X, y



# ------------------------------------------------------------
# VISION (images): raw pixels or ImageNet-feature embeddings
# ------------------------------------------------------------

def _resnet18_embed_dataset(dataset, device="cuda", batch_size=256):
    """
    Given a torchvision dataset returning (C,H,W) tensors in [0,1] (after ToTensor)
    with ImageNet normalization applied, extract 512-d ResNet18 penultimate features.
    Returns: X (N, 512) float32, y (N,) int64
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

    model = tv.models.resnet18(
        weights=tv.models.ResNet18_Weights.IMAGENET1K_V1
    ).to(device).eval()

    extractor = create_feature_extractor(model, return_nodes={"avgpool": "feat"}).to(device)

    feat = torch.empty(len(dataset), 512, device=device, dtype=torch.float32)
    ofs = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            out = extractor(xb)["feat"].squeeze(-1).squeeze(-1)  # (B,512,1,1)->(B,512)
            n = out.size(0)
            feat[ofs:ofs+n] = out
            ofs += n

    X = feat.cpu().numpy().astype(np.float32)
    y = np.asarray(getattr(dataset, "targets", getattr(dataset, "labels", None)))
    if isinstance(y, list):  # SVHN uses list
        y = np.array(y)
    y = y.astype(np.int64)
    return X, y


def load_cifar100_resnet18_feats(split="train", device="cuda", batch_size=256):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    train = (split == "train")
    ds = tv.datasets.CIFAR100(root="./data", train=train, download=True, transform=tfm)
    return _resnet18_embed_dataset(ds, device=device, batch_size=batch_size)


def load_stl10_resnet18_feats(split="train", device="cuda", batch_size=256):
    """
    split ∈ {"train","test"}; ignores the unlabeled split on purpose.
    """
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    ds = tv.datasets.STL10(root="./data", split=split, download=True, transform=tfm)
    return _resnet18_embed_dataset(ds, device=device, batch_size=batch_size)


def load_svhn_resnet18_feats(split="train", device="cuda", batch_size=256):
    """
    SVHN labels are 1..10 with '10' meaning '0'. We remap to 0..9.
    """
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    split_map = {"train":"train", "test":"test"}
    ds = tv.datasets.SVHN(root="./data", split=split_map[split], download=True, transform=tfm)
    X, y = _resnet18_embed_dataset(ds, device=device, batch_size=batch_size)
    y = y.copy()
    y[y == 10] = 0
    return X, y


def load_olivetti_faces(flatten=True):
    """
    Olivetti faces (64x64 grayscale). Returns dense data.
    X in [0,1].
    """
    data = fetch_olivetti_faces()
    X = data.images.astype(np.float32)  # (400, 64, 64)
    y = data.target.astype(np.int64)
    if flatten:
        X = X.reshape(len(X), -1).astype(np.float32)
    return X, y


def load_sklearn_digits(flatten=True):
    """
    sklearn 'digits' (1797 samples of 8x8).
    Values are 0..16; we normalize to [0,1].
    """
    D = load_digits()
    X = D.images.astype(np.float32) / 16.0
    y = D.target.astype(np.int64)
    if flatten:
        X = X.reshape(len(X), -1).astype(np.float32)
    return X, y


# ------------------------------------------------------------
# TEXT: TF-IDF or SVD embeddings -> dense
# ------------------------------------------------------------

def load_rcv1_svd(n_components=256, normalize_rows=True, random_state=0):
    """
    Reuters RCV1-v2 (sparse CSR). We compute a dense SVD embedding.
    Note: multilabel → we return the dominant label index per sample for y
    (you can adapt as needed).
    """
    rcv1 = fetch_rcv1()  # (N, d) sparse
    X_tf = rcv1.data  # CSR
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X = svd.fit_transform(X_tf).astype(np.float32)
    if normalize_rows:
        X = normalize(X, norm="l2").astype(np.float32)

    # rcv1.target is multilabel CSR; we pick argmax per row (most positive topic)
    # This is a lossy reduction to single-label; adjust if you prefer multilabel.
    T = rcv1.target.tocsr()
    y = np.asarray(T.argmax(axis=1)).ravel().astype(np.int64)
    return X, y


def load_openml_20ng_tfidf_svd(n_components=256, normalize_rows=True, random_state=0):
    """
    Alternate 20NG via OpenML (id=42720) for reproducibility without sklearn's fetcher.
    """
    Xy = fetch_openml(data_id=42720, as_frame=False)  # "20 newsgroups"
    texts = Xy["data"]  # already vectorized? On OpenML this one is raw text — but not always consistent.
    # To be robust: if it's raw strings, vectorize; if numeric array, pass through.
    if isinstance(texts[0], str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", max_features=200000, min_df=2)
        X_tf = vec.fit_transform(texts)
    else:
        from scipy import sparse
        X_tf = sparse.csr_matrix(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X = svd.fit_transform(X_tf).astype(np.float32)
    if normalize_rows:
        X = normalize(X, norm="l2").astype(np.float32)

    # OpenML targets are strings; map to int
    y_raw = Xy["target"]
    _, y = np.unique(y_raw, return_inverse=True)
    y = y.astype(np.int64)
    return X, y


# ------------------------------------------------------------
# TABULAR (OpenML / UCI): dense numeric with standardization
# ------------------------------------------------------------

def _openml_dense_numeric(name_or_id, target_name=None, one_hot_categoricals=True, standardize=True):
    """
    Generic OpenML fetch → dense numpy features and int labels if classification.
    """
    D = fetch_openml(name=name_or_id, as_frame=True)
    X_df = D.data
    y_series = D.target

    # Handle categoricals
    import pandas as pd
    if one_hot_categoricals:
        X_df = pd.get_dummies(X_df, drop_first=False)

    X = X_df.to_numpy(dtype=np.float32)

    y = None
    if y_series is not None:
        y_raw = y_series.to_numpy()
        if y_raw.dtype.kind in "OUSb":  # strings/object/bool → encode
            _, y = np.unique(y_raw, return_inverse=True)
        else:
            y = y_raw.astype(np.int64)
        y = y.astype(np.int64)

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X).astype(np.float32)

    return X, y

def load_openml_covtype():
    """UCI Covertype (581,012 x 54)."""
    return _openml_dense_numeric("Covertype")

def load_openml_phoneme():
    """Phoneme (5 classes, 5404 x 5)."""
    return _openml_dense_numeric("phoneme")

def load_openml_credit_g():
    """German Credit (1000 x 20)."""
    return _openml_dense_numeric("credit-g")


# ------------------------------------------------------------
# SINGLE-CELL (Scanpy): produce PCA embeddings + cluster labels
# ------------------------------------------------------------

def load_pbmc68k_reduced():
    """
    Already PCA-reduced (Scanpy demo). Returns X (N, 50) float32, y int labels.
    """
    adata = sc.datasets.pbmc68k_reduced()
    # 'X_pca' present; clusters available as 'louvain'
    X = adata.obsm["X_pca"].astype(np.float32)
    cats = adata.obs["louvain"]
    if hasattr(cats, "cat"):
        y = cats.cat.codes.to_numpy(dtype=np.int64)
    else:
        y = cats.to_numpy().astype(np.int64)
    return X, y

import numpy as np
import scanpy as sc

def load_paul15(n_hvg=2000, n_pcs=50, n_neighbors=15,
                resolution=1.0, random_state=0):
    """
    Paul et al. hematopoiesis scRNA-seq.
    Returns
      X: float32, shape (N, n_pcs)  — PCA embedding (dense)
      y: int64, shape (N,)          — cluster ids
    """
    # Load and preprocess
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e4)   # library size normalize
    sc.pp.log1p(adata)                              # log1p
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)

    # PCA
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=random_state)

    # KNN graph for clustering/UMAP
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        random_state=random_state,
    )
    # Sanity: make sure neighbors are present (avoids the KeyError you saw)
    assert "neighbors" in adata.uns, "Scanpy neighbors graph was not computed."

    # Clustering: prefer Leiden; try new igraph flavor, fall back if unavailable
    clustered = False
    try:
        # Future-proof defaults: flavor="igraph", n_iterations=2, directed=False
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=random_state,
            key_added="clusters",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        clustered = True
    except TypeError:
        # Older Scanpy/leidenalg without 'flavor' or 'n_iterations'
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=random_state,
            key_added="clusters",
        )
        clustered = True
    except Exception:
        pass

    if not clustered:
        # Final fallback: try Louvain (also with igraph flavor if supported)
        try:
            sc.tl.louvain(
                adata,
                resolution=resolution,
                random_state=random_state,
                key_added="clusters",
                flavor="igraph",
                directed=False,
            )
        except TypeError:
            sc.tl.louvain(
                adata,
                resolution=resolution,
                random_state=random_state,
                key_added="clusters",
            )

    # Dense features + integer labels
    X = adata.obsm["X_pca"].astype(np.float32)
    cats = adata.obs["clusters"]
    if hasattr(cats, "cat"):
        y = cats.cat.codes.to_numpy(dtype=np.int64)
    else:
        y = cats.to_numpy().astype(np.int64)
    return X, y


# ------------------------------------------------------------
# EXTRA: raw pixel loaders for USPS / OpenML MNIST-like
# ------------------------------------------------------------

def load_usps(flatten=True):
    """
    USPS (16x16) via OpenML. Values are floats already; normalize to [0,1].
    """
    D = fetch_openml("usps", version=2, as_frame=False)
    X = D.data.astype(np.float32)
    # Values are in [-1, 1] typically; rescale to [0,1]
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    y_raw = D.target
    _, y = np.unique(y_raw, return_inverse=True)
    y = y.astype(np.int64)
    if not flatten:
        X = X.reshape(len(X), 16, 16).astype(np.float32)
    return X, y


def load_openml_mnist784(flatten=True):
    """
    MNIST via OpenML (duplicate of torchvision MNIST but useful in some setups).
    """
    D = fetch_openml("mnist_784", as_frame=False)
    X = (D.data.astype(np.float32) / 255.0)
    y_raw = D.target
    y = y_raw.astype(np.int64) if y_raw.dtype.kind in "i" else np.array(y_raw, dtype=str).astype(int)
    if not flatten:
        X = X.reshape(len(X), 28, 28).astype(np.float32)
    return X, y
