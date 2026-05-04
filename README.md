<p align="center">
  <a href="https://github.com/yungselm/multimoda-rs">
    <img
      src="https://raw.githubusercontent.com/yungselm/multimoda-rs/main/media/multimoda-rs.jpg"
      alt="multimoda-rs logo"
    >
  </a>
</p>

[![PyPI](https://img.shields.io/pypi/v/multimodars.svg)](https://pypi.org/project/multimodars)
[![License](https://img.shields.io/pypi/l/multimodars.svg)](LICENSE)
[![Docs](https://img.shields.io/readthedocs/multimoda-rs)](https://multimoda-rs.readthedocs.io)
[![Tests and Build](https://github.com/yungselm/multimoda-rs/actions/workflows/CI.yml/badge.svg)](https://github.com/yungselm/multimoda-rs/actions/workflows/CI.yml)
[![status](https://joss.theoj.org/papers/24992cb138d85b0bf08f01bc2b384a80/status.svg)](https://joss.theoj.org/papers/24992cb138d85b0bf08f01bc2b384a80)


<figure class="epigraph" style="text-align: center; font-style: italic;">
  <blockquote>
    "One package to fuse them all."
  </blockquote>
  <figcaption>— <cite>The Lord of the Rings (probably)</cite></figcaption>
</figure>

---

> A high‑performance, Rust‑accelerated toolkit for multi‑modality cardiac image fusion and registration ﮩ٨ـﮩﮩ٨ـ♡ﮩ٨ـﮩﮩ٨ـ.

---

## Overview

`multimoda-rs` aligns and fuses diverse cardiac imaging modalities — IVUS, OCT and CCTA — into unified high-resolution 3D models. Originally developed to quantify dynamic lumen deformation in coronary artery anomalies (CAAs), it is equally applicable to longitudinal studies (e.g., pre/post-stenting) and general coronary artery disease workflows. The Rust backend parallelizes computationally intensive registration steps for speeds well beyond pure Python.

## Key Features

- **Intravascular Registration**: align pullback sequences (rest/stress, diastole/systole) using Hausdorff distance on vessel contours and catheter centroids; four modes: *full*, *double-pair*, *single-pair*, *single*.
- **Centerline Alignment**: register intravascular geometries onto a CCTA-derived centerline via three-point landmark or manual rotation.
- **CCTA Fusion**: automatically label CCTA geometries by vessel region and morph them to match intravascular measurements.
- **Flexible Input**: accepts CSV files ([AIVUS](https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA) format) or raw NumPy arrays.

## Installation

```bash
pip install multimodars
```

Optional extras:

```bash
pip install "multimodars[viz]"      # mesh visualisation (pyglet)
pip install "multimodars[meshlab]"  # MeshLab integration
```

For building from source or the full developer setup (tests, linting, docs), see the [Installation guide](https://multimoda-rs.readthedocs.io/en/latest/installation.html).

## Quick Example

Download [examples.zip](https://github.com/yungselm/multimoda-rs/releases/latest/download/examples.zip) (SHA256: `d11ebc7607f43ab4571fb51c9ac9178caac57774cf5d97f4f068ace4eb070fee`) from the latest release and extract it to your working directory.

```python
import multimodars as mm
import numpy as np

# Align four intravascular phases (rest-dia, rest-sys, stress-dia, stress-sys)
rest, stress, dia, sys, _ = mm.from_file_full(
    "examples/data/ivus_rest",    # input_path_ab
    "examples/data/ivus_stress",  # input_path_cd
    write_obj=True,
    output_path_ab="output/rest",
    output_path_cd="output/stress",
    output_path_ac="output/diastole",
    output_path_bd="output/systole",
)

# Align geometry onto a CCTA-derived centerline
cl_raw = np.genfromtxt("examples/data/centerline_raw.csv", delimiter=",")
centerline = mm.numpy_to_centerline(cl_raw)
aligned_pair, cl_resampled = mm.align_three_point(
    centerline, rest,
    aortic_ref_pt=(12.2605, -201.3643, 1751.0554),
    upper_ref_pt=(11.7567, -202.1920, 1754.7975),
    lower_ref_pt=(15.6605, -202.1920, 1749.9655),
    write=True,
    output_dir="output/aligned",
)
```

## Pipeline

**1. Intrapullback alignment** — frames within each pullback are co-registered to remove cardiac-motion artefacts, yielding clean diastolic and systolic geometries:

![Dynamic lumen changes](https://raw.githubusercontent.com/yungselm/multimoda-rs/main/media/dynamic_lumen_changes.png)

**2. Inter-pullback alignment** — registered pullbacks (rest vs. stress, diastole vs. systole) are aligned against each other to reveal stress-induced and pulsatile deformation:

![Stress-induced diastolic lumen deformation](https://raw.githubusercontent.com/yungselm/multimoda-rs/main/docs/figures/animation_stress_induced_systolic_deformation.gif)

**3. CCTA labeling** — the CCTA-derived geometry is automatically segmented by vessel region (aorta, RCA, LCA, intramural) to prepare it for fusion:

![Initial CCTA labeling](https://raw.githubusercontent.com/yungselm/multimoda-rs/main/docs/figures/initial_labeling.jpg)

**4. CCTA–morphing** — the labeled CCTA geometry is morphed along the centerline to match the high-resolution intravascular geometry:

![CCTA scaling](https://raw.githubusercontent.com/yungselm/multimoda-rs/main/docs/figures/scaling.jpg)

**5. CCTA/intravascular fusio** — the morphed CCTA geometry is stitched to the intravascular geometry, replacing a section with a high resolution verison:

![CCTA fusion](https://raw.githubusercontent.com/yungselm/multimoda-rs/main/docs/figures/concept.jpg)

## Documentation

Full documentation — installation, step-by-step tutorials, interactive Jupyter notebooks, and API reference — is available at **[multimoda-rs.readthedocs.io](https://multimoda-rs.readthedocs.io)**.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

Please kindly cite the following paper if you use this repository.

```
@article{stark2025multimodars,
  title     = {multimodars: A Rust-powered toolkit for multi-modality cardiac image fusion and registration},
  author    = {Stark, Anselm W. and Ilic, Marc and Mokhtari, Ali and Mohammadi Kazaj, Pooya and Graeni, Christoph and Shiri, Isaac},
  journal   = {arXiv preprint arXiv:2510.06241},
  year      = {2025}
}
```

Stark, Anselm W., Marc Ilic, Ali Mokhtari, Pooya Mohammadi Kazaj, Christoph Graeni, and Isaac Shiri. "multimodars: A Rust-powered toolkit for multi-modality cardiac image fusion and registration." arXiv preprint arXiv:2510.06241 (2025).
