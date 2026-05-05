"""Benchmark 1: bruteforce vs. optimised alignment at varying step sizes.

Tests from_file_full on the bundled IVUS rest/stress example data with
step_rotation_deg in {5.0, 2.5, 1.0, 0.5, 0.1} degrees.

write_obj / smooth / postprocessing are all disabled so only the alignment
algorithm itself is timed.

Output: benchmarks/results/bruteforce_stepsize.png
"""

import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

import multimodars as mm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
DATA_ROOT = _ROOT / "examples" / "data"
INPUT_AB = str(DATA_ROOT / "ivus_rest")
INPUT_CD = str(DATA_ROOT / "ivus_stress")
RESULTS_DIR = Path(__file__).parent / "results"

STEP_SIZES: list[float] = [5.0, 2.5, 1.0, 0.5, 0.1, 0.05]
RANGE_DEG: float = 90.0
REPEATS: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _n_steps(step: float) -> float:
    return (RANGE_DEG * 2) / step


def _run_once(step: float, bruteforce: bool) -> float:
    t0 = time.perf_counter()
    mm.from_file_full(
        input_path_ab=INPUT_AB,
        input_path_cd=INPUT_CD,
        step_rotation_deg=step,
        range_rotation_deg=RANGE_DEG,
        write_obj=False,
        smooth=False,
        postprocessing=False,
        bruteforce=bruteforce,
        interpolation_steps=0,
    )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, list[float]]:
    results: dict[str, list[float]] = {"bruteforce": [], "optimised": []}

    for step in STEP_SIZES:
        print(f"step={step:4.1f}°  bruteforce=True  ", end="", flush=True)
        times_bf = [_run_once(step, bruteforce=True) for _ in range(REPEATS)]
        med_bf = statistics.median(times_bf)
        results["bruteforce"].append(med_bf)
        print(f"{med_bf:.2f} s")

        print(f"step={step:4.1f}°  bruteforce=False ", end="", flush=True)
        times_opt = [_run_once(step, bruteforce=False) for _ in range(REPEATS)]
        med_opt = statistics.median(times_opt)
        results["optimised"].append(med_opt)
        speedup = med_bf / med_opt if med_opt > 0 else float("inf")
        print(f"{med_opt:.2f} s  (speedup {speedup:.1f}x)")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: dict[str, list[float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_steps_list = [_n_steps(s) for s in STEP_SIZES]
    bf_times = results["bruteforce"]
    opt_times = results["optimised"]
    speedups = [bf / opt for bf, opt in zip(bf_times, opt_times)]

    fig, (ax_main, ax_speedup) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    # --- Left panel: wall time ---
    ax_main.plot(
        n_steps_list,
        bf_times,
        "o-",
        color="tab:red",
        linewidth=2,
        markersize=7,
        label="Bruteforce",
    )
    ax_main.plot(
        n_steps_list,
        opt_times,
        "s--",
        color="tab:blue",
        linewidth=2,
        markersize=7,
        label="Optimised",
    )

    # O(n) reference anchored at the first bruteforce point
    ref_n, ref_t = n_steps_list[0], bf_times[0]
    linear_ref = [ref_t * (n / ref_n) for n in n_steps_list]
    ax_main.plot(
        n_steps_list,
        linear_ref,
        ":",
        color="tab:red",
        linewidth=1.5,
        alpha=0.55,
        label="O(n) reference",
    )

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel("Number of rotation steps  (2 × range / step)", fontsize=12)
    ax_main.set_ylabel("Median wall time  (s)", fontsize=12)
    ax_main.set_title(
        "Bruteforce vs. Optimised — Step-Size Scaling\n"
        "from_file_full · IVUS rest/stress · range = 90°",
        fontsize=11,
    )
    ax_main.legend(fontsize=11)
    ax_main.grid(True, which="both", alpha=0.3)

    # Secondary x-axis with step-size labels
    ax_top = ax_main.twiny()
    ax_top.set_xscale("log")
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks(n_steps_list)
    ax_top.set_xticklabels([f"{s}°" for s in STEP_SIZES])
    ax_top.set_xlabel("Step size  (degrees)", fontsize=12)

    # --- Right panel: speedup factor ---
    ax_speedup.plot(
        n_steps_list,
        speedups,
        "D-",
        color="tab:green",
        linewidth=2,
        markersize=7,
    )
    ax_speedup.axhline(1.0, color="gray", linewidth=1, linestyle=":")
    ax_speedup.set_xscale("log")
    ax_speedup.set_xlabel("Number of rotation steps  (2 × range / step)", fontsize=12)
    ax_speedup.set_ylabel("Speedup  (bruteforce / optimised)", fontsize=12)
    ax_speedup.set_title(
        "Speedup of Optimised over Bruteforce\n"
        "from_file_full · IVUS rest/stress · range = 90°",
        fontsize=11,
    )
    ax_speedup.grid(True, which="both", alpha=0.3)

    ax_top2 = ax_speedup.twiny()
    ax_top2.set_xscale("log")
    ax_top2.set_xlim(ax_speedup.get_xlim())
    ax_top2.set_xticks(n_steps_list)
    ax_top2.set_xticklabels([f"{s}°" for s in STEP_SIZES])
    ax_top2.set_xlabel("Step size  (degrees)", fontsize=12)

    out = RESULTS_DIR / "bruteforce_stepsize.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Benchmark 1: Bruteforce vs. Optimised — Step-Size Scaling ===\n")
    results = run_benchmark()
    print("\nPlotting results …")
    plot_results(results)
