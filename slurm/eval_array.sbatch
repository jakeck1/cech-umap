#!/bin/bash
#SBATCH -J cech_eval
#SBATCH -o slurm_logs/%x-%A_%a.out
#SBATCH -e slurm_logs/%x-%A_%a.err
#SBATCH --array=0-999%60        # adjust upper bound and concurrency
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

mkdir -p slurm_logs

# Activate your env here (adapt to your cluster)
# e.g.:
# module load anaconda
# conda activate cech_umap
source .venv/bin/activate

# Map array index -> (DATASET, NEIGH, SEED)
read DATASET NEIGH SEED < <(awk -v n=$((SLURM_ARRAY_TASK_ID+1)) 'NR==n {print $1, $2, $3}' slurm/tasks.tsv)

echo "SLURM task $SLURM_ARRAY_TASK_ID: dataset=${DATASET}, k=${NEIGH}, seed=${SEED}"

python create_embeddings.py \
  --dataset "${DATASET}" \
  --neighbors "${NEIGH}" \
  --seed "${SEED}"

python evaluate_embeddings.py \
  --dataset "${DATASET}" \
  --neighbors "${NEIGH}" \
  --seed "${SEED}"
