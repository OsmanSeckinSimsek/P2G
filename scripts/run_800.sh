#!/bin/bash -l
#SBATCH --job-name=P2G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --partition=normal
#SBATCH --output=800.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NGPU can be injected by the caller (e.g. run_strong_scaling.sh); fall back to total MPI ranks
NGPU=${NGPU:-${SLURM_NTASKS}}
SUFFIX="${NGPU}gpu"

# Nearest neighbor interpolation
srun ./select_gpu.sh ./power_spectrum-cuda --checkpoint CHECKPOINT-PATH --interpolation nearest --stepNo 1 --gridSize 1024 --cuda-aware-full-pack --output P2G_nn_${SUFFIX}.txt

# Cell_average interpolation
srun ./select_gpu.sh ./power_spectrum-cuda --checkpoint CHECKPOINT-PATH --interpolation cell_avg --stepNo 1 --gridSize 1024 --cuda-aware-full-pack --output P2G_ca_${SUFFIX}.txt

# SPH interpolation
srun ./select_gpu.sh ./power_spectrum-cuda --checkpoint CHECKPOINT-PATH --interpolation sph --stepNo 1 --gridSize 1024 --cuda-aware-full-pack --output P2G_sph_${SUFFIX}.txt

mv 800.out scaling_${SUFFIX}.out