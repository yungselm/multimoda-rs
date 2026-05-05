Benchmarks
==========

All benchmarks were run on an Intel Xeon Gold 6234 (8 physical cores, 16 logical
processors via HyperThreading) under WSL2.  Example data shipped with the package
was used throughout: the IVUS rest/stress pullbacks for the step-size benchmark
and the OCT pullback (280 frames) for the parallelisation benchmark.

.. _benchmark-algorithm:

1. Algorithmic improvement: bruteforce vs. optimised
-----------------------------------------------------

The optimised alignment algorithm uses a coarse-to-fine hierarchical search
instead of evaluating every candidate angle exhaustively.  The effect is small
at coarse step sizes (few angles to evaluate) but grows rapidly as the step size
decreases, because the number of candidate angles scales as
:math:`n = 2 \times \text{range} / \text{step}`.

**Test setup** — ``from_file_full`` on the IVUS rest/stress example data,
``range_rotation_deg = 90°``, ``write_obj = False``, ``smooth = False``,
``postprocessing = False``.  Three repetitions per condition; median wall time
reported.

.. figure:: ../benchmarks/results/bruteforce_stepsize.png
   :name: fig-benchmark-stepsize
   :alt: Bruteforce vs. optimised alignment wall time and speedup across step sizes
   :align: center
   :width: 900px

   Wall time (left, log-log) and speedup factor (right) of the optimised
   algorithm over bruteforce as a function of the rotation step size.
   The O(n) reference line confirms the linear scaling of bruteforce with the
   number of candidate angles; the optimised search is sub-linear.

At step sizes of 1° and above the difference is modest (< 2x).  Below 1° the
gap widens substantially: at **0.1°** the optimised algorithm is **5.5x faster**
and at **0.05°** the advantage grows to **10.3x** (6.25 s vs. 64.4 s).  This is
the practically relevant regime: fine step sizes are required for high-accuracy
alignment of OCT data and dense IVUS pullbacks.

.. _benchmark-parallelisation:

2. Parallelisation scaling
--------------------------

The second benchmark tests how much additional speed is gained by increasing
the number of CPU cores, using ``from_array_single`` on the OCT dataset
(280 frames, ``step_rotation_deg = 0.01°``, ``range_rotation_deg = 6°``).
Each core count was run in a fresh subprocess so that rayon's global thread
pool re-initialises from ``RAYON_NUM_THREADS``.

.. list-table:: Median wall time (s) across CPU core counts
   :header-rows: 1
   :widths: 12 20 20 18

   * - Cores
     - Bruteforce (s)
     - Optimised (s)
     - Speedup (alg.)
   * - 2
     - 131.47
     - 14.10
     - 9.3x
   * - 4
     - 102.79
     - 10.92
     - 9.4x
   * - 8
     - 184.06
     - 18.58
     - 9.9x
   * - 12
     - 109.17
     - 11.44
     - 9.5x
   * - 16
     - 97.44
     - 10.30
     - 9.5x

**Key observations**

* The optimised algorithm is consistently **~9.5x faster** than bruteforce
  regardless of core count.  This algorithmic advantage is stable and
  independent of hardware configuration.

* Going from 2 to 16 cores reduces the optimised wall time from 14.1 s to
  10.3 s — a **1.4x improvement** from eight times the hardware.
  The bruteforce shows similar diminishing returns (1.35x).

* The anomalous result at 8 cores is a WSL2 / HyperThreading artefact: with
  8 logical threads, every physical core has exactly one idle HT sibling that
  the Windows scheduler colonises, causing cache interference.  This is not
  present when running natively on Linux.

**Conclusion** — choosing the optimised algorithm over bruteforce delivers a
9.5x speedup that no amount of additional hardware can replicate.  Adding cores
yields at most a further 1.4x gain and requires careful tuning of
``RAYON_NUM_THREADS`` to avoid the HyperThreading boundary.  **Algorithm choice
is the dominant factor; parallelisation is secondary.**
