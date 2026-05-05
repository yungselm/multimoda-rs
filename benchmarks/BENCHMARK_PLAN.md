# Benchmark Plan: multimoda-rs Alignment Performance

## Overview

Two benchmarks to characterise the alignment performance of `multimodars`.

| # | What | Function | Dataset |
|---|------|----------|---------|
| 1 | Bruteforce vs. optimised at varying step sizes | `from_file_full` | IVUS rest/stress |
| 2 | CPU-core scaling (parallelism) | `from_array_single` | OCT (280 frames) |

Results land in `benchmarks/results/`.

---

## Benchmark 1 — Bruteforce vs. Optimised: Step-Size Scaling

### Goal

Quantify how bruteforce and optimised alignment scale as `step_rotation_deg` decreases
(i.e. as the rotation-search grid becomes finer).

### Setup

| Parameter | Value |
|-----------|-------|
| Function | `from_file_full` |
| Dataset | `examples/data/ivus_rest` / `examples/data/ivus_stress` (IVUS) |
| `range_rotation_deg` | 90.0° (±90° → 180° total sweep) |
| `write_obj` | `False` — avoids disk-I/O noise |
| `smooth` | `False` |
| `postprocessing` | `False` |
| `interpolation_steps` | `0` |
| Repetitions per condition | 3 (median wall time reported) |

### Independent variable

`step_rotation_deg` ∈ {5.0, 2.5, 1.0, 0.5, 0.1} degrees.

Corresponding number of rotation steps n = 180 / step (using the full ±90° range):

| step (°) | n\_steps |
|----------|---------|
| 5.0      | 36      |
| 2.5      | 72      |
| 1.0      | 180     |
| 0.5      | 360     |
| 0.1      | 1 800   |

### Expected behaviour

- **Bruteforce**: exhaustively scores every rotation angle for every frame pair →
  wall time grows linearly with n\_steps (O(n)).
- **Optimised**: coarse-to-fine hierarchical search → sub-linear in n\_steps,
  asymptotically O(log n).

The plot axes are logarithmic so the two complexity classes appear as straight lines
with clearly different slopes.

### Output

`benchmarks/results/bruteforce_stepsize.png` — log-log line plot:

- x-axis (bottom): number of rotation steps; (top): step size in degrees
- y-axis: median wall time (seconds)
- Line 1: bruteforce (red ○–) + O(n) reference dashed line
- Line 2: optimised (blue ■ --)
- Right y-axis: speedup factor (bruteforce / optimised)

---

## Benchmark 2 — CPU-Core Scaling

### Goal

Measure how parallelism scales from 1 → 16 CPU cores for both alignment modes,
using a large OCT pullback (280 frames) to stress the frame-level parallelism
in the rayon thread pool.

### Note on function choice

The OCT example data (`examples/data/oct_single/`) ships as
`oct_contours_raw.csv` + `oct_ref.csv`, which do not follow the AIVUS-CAA
file-naming convention expected by `from_file_single`.  The benchmark therefore
uses `from_array_single` (as demonstrated in the docs notebook), which accepts
raw numpy arrays directly.  The algorithm under test is identical.

### Setup

| Parameter | Value |
|-----------|-------|
| Function | `from_array_single` |
| Dataset | `examples/data/oct_single/` (280 frames, OCT) |
| `step_rotation_deg` | 0.01° (from docs/notebooks) |
| `range_rotation_deg` | 6.0° (from docs/notebooks) |
| `image_center` | (5.0, 5.0) (from docs/notebooks) |
| `n_points` | 40 (from docs/notebooks) |
| `smooth` | `False` |
| `write_obj` | `False` |
| Repetitions per condition | 3 (median wall time reported) |

### Independent variable

`RAYON_NUM_THREADS` ∈ {1, 2, 4, 8, 12, 16}

**Important**: rayon's global thread pool is initialised once per process from
the `RAYON_NUM_THREADS` environment variable.  To guarantee a fresh pool at each
thread count, every (n\_cores × bruteforce × repeat) combination is executed in
a **separate subprocess** with the variable set before any import of `multimodars`.

### Expected behaviour

- **Optimised**: near-linear speedup up to ~8 cores, then diminishing returns
  (Amdahl's law — serial fraction of the pipeline limits gains beyond that).
- **Bruteforce**: also benefits from rayon frame-level parallelism, but expected
  to saturate earlier because the per-step work is less cache-friendly.

### Output

`benchmarks/results/cpu_scaling.png` — two-panel figure:

- Left panel: wall time (s) vs. CPU cores, two lines (bruteforce, optimised)
- Right panel: speedup relative to 1 core vs. CPU cores +
  ideal linear-speedup reference; same two lines

---

## Running the Benchmarks

```bash
# From project root
python benchmarks/benchmark_bruteforce_stepsize.py
python benchmarks/benchmark_cpu_scaling.py
```

Estimated runtimes (rough, machine-dependent):

| Benchmark | Estimate |
|-----------|----------|
| Benchmark 1 (all step sizes × 2 modes × 3 reps) | 5–30 min |
| Benchmark 2 (6 core counts × 2 modes × 3 reps in subprocesses) | 20–60 min |

Results are saved as PNG files in `benchmarks/results/`.
