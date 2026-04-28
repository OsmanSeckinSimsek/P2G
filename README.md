# P2G — Particle-to-Grid

Particles are interpolated onto a uniform Cartesian mesh. As an application for the Cartesian mesh interpolation, a distributed 3D FFT is applied, and the result is averaged into spherical shells to produce a 1D power spectrum.

## Features

- Three particle-to-grid interpolation methods: nearest neighbour, SPH kernel-weighted, and cell-average
- Distributed 3D FFT via [HeFFTe](https://github.com/af-ayala/heffte) (slab and pencil decompositions)
- CPU/MPI and CUDA rasterization backends
- HDF5 checkpoint input via H5Part

## Third-party code

The following components are taken from the [SPH-EXA repository](https://github.com/sphexa-org/sphexa):

| Path | Description |
|------|-------------|
| `cstone/` | Cornerstone octree library — distributed domain decomposition and particle exchange |
| `extern/h5part/` | H5Part library — parallel HDF5 I/O for particle data |
| `extern/io/` | I/O abstraction layer — HDF5 reader/writer interface |

## Dependencies

| Dependency | Notes |
|---|---|
| C++20 compiler | GCC ≥ 11 or Clang ≥ 14 |
| MPI | |
| HDF5 (parallel) | |
| [HeFFTe](https://github.com/af-ayala/heffte) | Distributed FFT backend |
| OpenMP | |
| CUDA ≥ 11.6 | Enables GPU rasterization |

## Building

```bash
mkdir build && cd build
cmake .. \
    -DHEFFTE_PATH=/path/to/heffte \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCSTONE_WITH_GPU_AWARE_MPI=ON
make -j
```

Key CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `HEFFTE_PATH` | `$HOME/local/heffte` | Path to HeFFTe installation |
| `CSTONE_WITH_GPU_AWARE_MPI` | `ON` | Enable CUDA-aware MPI |

The build produces two binaries in `main/src/`:

- `power_spectrum` — CPU/MPI backend
- `power_spectrum-cuda` — CUDA backend (when a CUDA compiler is found)

## Usage

```
./power_spectrum-cuda [OPTIONS]

  --checkpoint        HDF5 checkpoint file
  --stepNo            Step number in the checkpoint file (default: 0)
  --gridSize          Mesh resolution (default: derived from particle count)
  --numShells         Number of spherical shells (default: gridSize / 2)
  --interpolation     Interpolation method: nearest (default), sph, cell_avg
  --output            Output file for the power spectrum (default: power_spectrum.txt)
  --pencils           Use HeFFTe pencil decomposition (default: slab)
  --cuda-aware-mpi    Enable CUDA-aware MPI exchange in CUDA backends
  --cuda-aware-full-pack  Enable full GPU rank-packing for CUDA-aware mode
```

The output file contains two whitespace-separated columns: shell index and power spectrum value.

## Running on a cluster (SLURM)

`scripts/select_gpu.sh` assigns one GPU per MPI rank via `CUDA_VISIBLE_DEVICES`.

A ready-made job script is provided:

```bash
sbatch scripts/run_800.sh
```

### Strong scaling study

```bash
bash scripts/run_strong_scaling.sh
```

This submits four jobs for 2, 4, 8, and 16 GPUs. Output files are named
`scaling_Ngpu.out` and spectrum files are suffixed with the GPU count (e.g. `P2G_nn_8gpu.txt`).

### Plotting results

```bash
python scripts/plot_strong_scaling.py [output_dir]
```

Reads the SLURM output files produced by the scaling study and saves `strong_scaling.png`
with speedup and parallel efficiency plots for the domain-sync and P2G phases across all
three interpolation methods.

## License

MIT — see [LICENSE](LICENSE).
