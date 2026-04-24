#!/usr/bin/env python3
"""
Parse SLURM output files produced by run_strong_scaling.sh and plot
strong-scaling speedup and parallel efficiency for the Sync + P2G phases.

Usage:
    python plot_strong_scaling.py [output_dir]

output_dir defaults to the current working directory.
Expected files: scaling_2gpu.out, scaling_4gpu.out, scaling_8gpu.out, scaling_16gpu.out
"""

import re
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

GPU_COUNTS   = [2, 4, 8, 16]
INTERP_ORDER = ["nearest", "cell_avg", "sph"]

# ── parser ────────────────────────────────────────────────────────────────────

def parse_timings(filepath):
    """Return a list of timing dicts (one per timing-summary block in the file)."""
    timings = []
    pattern = re.compile(
        r"--- Timing summary ---\s*\n"
        r"Checkpoint read time\s*:\s*([\d.eE+\-]+)\s*s\n"
        r"Sync time\s*:\s*([\d.eE+\-]+)\s*s\n"
        r"P2G time\s*:\s*([\d.eE+\-]+)\s*s\n"
        r"Power spectrum time\s*:\s*([\d.eE+\-]+)\s*s\n"
        r"-------------------------------"
    )
    with open(filepath) as fh:
        for m in pattern.finditer(fh.read()):
            timings.append({
                "checkpoint":     float(m.group(1)),
                "sync":           float(m.group(2)),
                "p2g":            float(m.group(3)),
                "power_spectrum": float(m.group(4)),
            })
    return timings

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # Collect timings: gpu_count -> [timing_nn, timing_ca, timing_sph]
    all_timings = {}
    for ngpu in GPU_COUNTS:
        path = os.path.join(out_dir, f"scaling_{ngpu}gpu.out")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        t = parse_timings(path)
        if not t:
            print(f"Warning: no timing summaries found in {path}.")
            continue
        all_timings[ngpu] = t

    if not all_timings:
        print("No timing data found. Exiting.")
        sys.exit(1)

    available = sorted(all_timings.keys())
    baseline_n = available[0]
    n_methods  = max(len(v) for v in all_timings.values())
    methods    = INTERP_ORDER[:n_methods]

    # ── build speedup / efficiency per method ────────────────────────────────
    colors  = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^"]

    fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(12, 5))

    for i, method in enumerate(methods):
        gpus, combined = [], []
        for ngpu in available:
            t = all_timings[ngpu]
            if i < len(t):
                gpus.append(ngpu)
                combined.append(t[i]["sync"] + t[i]["p2g"])

        if len(gpus) < 2:
            continue

        t_base   = combined[0]
        speedup  = [t_base / ct for ct in combined]
        eff      = [s / (n / baseline_n) for s, n in zip(speedup, gpus)]

        kw = dict(color=colors[i], marker=markers[i], linewidth=2, markersize=8)
        ax_sp.plot(gpus, speedup, label=method, **kw)
        ax_eff.plot(gpus, eff,    label=method, **kw)

    # ideal reference lines
    ideal_x  = [available[0], available[-1]]
    ideal_sp = [1.0, available[-1] / available[0]]
    ax_sp.plot(ideal_x, ideal_sp, "k--", linewidth=1.5, label="ideal")
    ax_eff.axhline(1.0,           color="k", linestyle="--", linewidth=1.5, label="ideal")

    # ── decorate ─────────────────────────────────────────────────────────────
    for ax in (ax_sp, ax_eff):
        ax.set_xlabel("Number of ranks")
        ax.set_xticks(available)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax_sp.set_ylabel("Speedup")
    ax_sp.set_title("Speedup  (Sync + P2G)")

    ax_eff.set_ylabel("Parallel efficiency")
    ax_eff.set_title("Parallel efficiency  (Sync + P2G)")
    ax_eff.set_ylim(0, 1.2)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "strong_scaling.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
