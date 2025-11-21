# ČechUMAP: Embeddings based on matching marginal probabilities of sampled Čech filtrations


This repository contains the ČechUMAP algorithm and experiments performed with it as specified in the paper 'A probabilistic perspective on fuzzy simplicial sets for geometric data analysis.'


In brief, the algorithm is based on the idea that UMAP may be interpreted as matching marginal probabilities of sampled Vietoris-Rips-Filtrations in embeddings and data space. 

We build on this idea and replace the VR-filtration by the Čech filtration and in turn sample triangles instead of edges. 


ČechUMAP is implemented with a PyTorch backend and supports GPU acceleration.

The API mimicks UMAP, that is data is transformed as follows:

```
from cech_umap.cumap import CechUMAP

cumap = CechUMAP()
embeddings = cumap.fit_transform(data)
```


# Installation

To install, if you only want to try the ČechUMAP algorithm, run 
```
pip install -e .
```
If you also want to run the experiments from the paper, run 

```
pip install -e .[experiments]
```
Alternatively, create the full Conda environment:
```
conda env create -f env.yml
conda activate cech-umap-paper
pip install -e .[experiments]
```










# Reproducing experiments

To run the full pipeline on a single machine:

```bash
# 1. Compute all embeddings (UMAP + ČechUMAP)
python create_embeddings.py

# 2. Evaluate all embeddings
python evaluate_embeddings.py

# 3. Merge evaluations into .npz files (per dataset)
python merge_evaluations.py

# 4. Regenerate plots
python plot_evaluations.py
```

All scripts also accept command line arguments to run only subsets of the experiments. For example:

```bash
python create_embeddings.py --dataset MNIST --neighbors 5 8 --seed 0 1
python evaluate_embeddings.py --dataset MNIST --neighbors 5 8 --seed 0 1
python merge_evaluations.py --neighbors 5 8 --seeds 0 1
```

For large-scale experiments we used a SLURM cluster. We provide a small example in slurm/:
```
cd slurm

# 1. Generate the task list (dataset, neighbors, seed triples)
python make_tasks.py

# 2. Submit an array job (adapt resources and array range to your cluster)
sbatch eval_array.sbatch
```

Each array task runs a single (dataset, k, seed) triple by calling:
```
python create_embeddings.py --dataset ... --neighbors ... --seed ...

python evaluate_embeddings.py --dataset ... --neighbors ... --seed ...
```



# Citation


