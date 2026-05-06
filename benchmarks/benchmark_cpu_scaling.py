"""Benchmark 2: CPU-core scaling for optimized vs. bruteforce alignment.

Tests from_array_single on the bundled OCT example dataset (280 frames) at
varying RAYON_NUM_THREADS values (1 → 16 cores).

Parameters follow the docs/notebooks:
  step_rotation_deg = 0.01, range_rotation_deg = 6.0

Because rayon's global thread pool is initialised once per process, every
(n_cores x bruteforce x repeat) combination is run in a fresh subprocess
so that RAYON_NUM_THREADS takes effect before the first rayon call.

Output: benchmarks/results/cpu_scaling.png
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
OCT_DATA_PATH = _ROOT / "examples" / "data" / "oct_single"
RESULTS_DIR = Path(__file__).parent / "results"

CORE_COUNTS: list[int] = [2, 4, 8, 12, 16]
REPEATS: int = 3

# Parameters from docs/notebooks (OCT single reconstruction)
OCT_STEP = 0.01
OCT_RANGE = 6.0
OCT_CENTER = (5.0, 5.0)
OCT_N_POINTS = 40

# ---------------------------------------------------------------------------
# Worker script template (run inside each subprocess)
# ---------------------------------------------------------------------------

_WORKER_TEMPLATE = """\
import os, sys, json, time
os.environ["RAYON_NUM_THREADS"] = "{n_cores}"

import numpy as np
sys.path.insert(0, r"{root}")
import multimodars as mm

data_dir = r"{data_path}"
oct_raw = np.genfromtxt(data_dir + "/oct_contours_raw.csv", delimiter=",")
oct_ref = np.genfromtxt(data_dir + "/oct_ref.csv",          delimiter=",")

oct_input = mm.numpy_to_inputdata(
    lumen_arr=oct_raw,
    ref_point=oct_ref,
    record=None,
    diastole=True,
    label="oct",
)

times = []
for _ in range({repeats}):
    t0 = time.perf_counter()
    mm.from_array_single(
        input_data=oct_input,
        step_rotation_deg={step},
        range_rotation_deg={rng},
        sample_size=200,
        image_center={center},
        n_points={n_points},
        write_obj=False,
        smooth=False,
        bruteforce={bruteforce},
    )
    times.append(time.perf_counter() - t0)

print(json.dumps(times))
"""


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _run_worker(n_cores: int, bruteforce: bool) -> list[float]:
    script = _WORKER_TEMPLATE.format(
        n_cores=n_cores,
        root=str(_ROOT),
        data_path=str(OCT_DATA_PATH),
        repeats=REPEATS,
        step=OCT_STEP,
        rng=OCT_RANGE,
        center=OCT_CENTER,
        n_points=OCT_N_POINTS,
        bruteforce=str(bruteforce),
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"\n  [stdout] {proc.stdout.strip()[:600]}", file=sys.stderr)
        print(f"  [stderr] {proc.stderr.strip()[:600]}", file=sys.stderr)
        raise RuntimeError(
            f"Worker failed (cores={n_cores}, bruteforce={bruteforce}, "
            f"returncode={proc.returncode})."
        )
    # The Rust layer may print progress lines before the final JSON row.
    last_line = proc.stdout.strip().rsplit("\n", 1)[-1]
    return json.loads(last_line)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, list[float]]:
    results: dict[str, list[float]] = {"bruteforce": [], "optimized": []}

    for n_cores in CORE_COUNTS:
        label_width = len(str(max(CORE_COUNTS)))

        print(
            f"cores={n_cores:{label_width}d}  bruteforce=True  … ",
            end="",
            flush=True,
        )
        times_bf = _run_worker(n_cores, bruteforce=True)
        med_bf = statistics.median(times_bf)
        results["bruteforce"].append(med_bf)
        print(f"{med_bf:.2f} s")

        print(
            f"cores={n_cores:{label_width}d}  bruteforce=False … ",
            end="",
            flush=True,
        )
        times_opt = _run_worker(n_cores, bruteforce=False)
        med_opt = statistics.median(times_opt)
        results["optimized"].append(med_opt)
        print(f"{med_opt:.2f} s")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: dict[str, list[float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bf_times = results["bruteforce"]
    opt_times = results["optimized"]

    bf_speedup = [bf_times[0] / t for t in bf_times]
    opt_speedup = [opt_times[0] / t for t in opt_times]
    ideal = [c / CORE_COUNTS[0] for c in CORE_COUNTS]

    fig, (ax_time, ax_speedup) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    # --- Left panel: wall time ---
    ax_time.plot(
        CORE_COUNTS,
        bf_times,
        "o-",
        color="tab:red",
        linewidth=2,
        markersize=7,
        label="Bruteforce",
    )
    ax_time.plot(
        CORE_COUNTS,
        opt_times,
        "s--",
        color="tab:blue",
        linewidth=2,
        markersize=7,
        label="optimized",
    )
    ax_time.set_xlabel("CPU cores  (RAYON_NUM_THREADS)", fontsize=12)
    ax_time.set_ylabel("Median wall time  (s)", fontsize=12)
    ax_time.set_title(
        "Wall Time vs. CPU Cores\n"
        "from_array_single · OCT · 280 frames · step=0.01° range=6°",
        fontsize=11,
    )
    ax_time.set_xticks(CORE_COUNTS)
    ax_time.legend(fontsize=11)
    ax_time.grid(True, alpha=0.3)

    # --- Right panel: speedup ---
    ax_speedup.plot(
        CORE_COUNTS,
        ideal,
        "k:",
        linewidth=1.5,
        alpha=0.6,
        label="Ideal (linear)",
    )
    ax_speedup.plot(
        CORE_COUNTS,
        bf_speedup,
        "o-",
        color="tab:red",
        linewidth=2,
        markersize=7,
        label="Bruteforce",
    )
    ax_speedup.plot(
        CORE_COUNTS,
        opt_speedup,
        "s--",
        color="tab:blue",
        linewidth=2,
        markersize=7,
        label="optimized",
    )
    ax_speedup.set_xlabel("CPU cores  (RAYON_NUM_THREADS)", fontsize=12)
    ax_speedup.set_ylabel("Speedup  (relative to 1 core)", fontsize=12)
    ax_speedup.set_title(
        "Parallel Speedup vs. CPU Cores\n"
        "from_array_single · OCT · 280 frames · step=0.01° range=6°",
        fontsize=11,
    )
    ax_speedup.set_xticks(CORE_COUNTS)
    ax_speedup.legend(fontsize=11)
    ax_speedup.grid(True, alpha=0.3)

    out = RESULTS_DIR / "cpu_scaling.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Benchmark 2: CPU-Core Scaling ===\n")
    print(f"Dataset : {OCT_DATA_PATH}")
    print(f"Cores   : {CORE_COUNTS}")
    print(f"Repeats : {REPEATS}  (median reported)\n")

    results = run_benchmark()
    print("\nPlotting results …")
    plot_results(results)
