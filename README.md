<p align="center">
  <a href="https://github.com/yungselm/multimoda-rs">
    <img
      src="https://raw.githubusercontent.com/yungselm/multimoda-rs/main/media/multimoda-rs.jpg"
      alt="multimoda-rs logo"
    >
  </a>
</p>

[![PyPI](https://img.shields.io/pypi/v/multimodars.svg)](https://pypi.org/project/multimodars)
[![License](https://img.shields.io/pypi/l/multimodars.svg)](LICENSE.md)
[![Docs](https://img.shields.io/readthedocs/multimodars)](https://multimodars.readthedocs.io)

<figure class="epigraph" style="text-align: center; font-style: italic;">
  <blockquote>
    “One package to fuse them all.”
  </blockquote>
  <figcaption>— <cite>The Lord of the Rings (probably)</cite></figcaption>
</figure>

---

> A high‑performance, Rust‑accelerated toolkit for multi‑modality cardiac image fusion and registration.

---
# Overview
`multimoda-rs` addresses the challenge of aligning and fusing diverse cardiac imaging modalities, such as CCTA, IVUS, OCT, and MRI—into a unified, high‑resolution 3D model. While CCTA provides comprehensive volumetric context, intravascular modalities (IVUS and OCT) offer sub‑millimeter resolution along the vessel lumen, and MRI (LGE) reveals tissue characteristics like scar and edema. This library leverages Rust for computationally intensive registration steps, delivering faster performance than pure Python implementations.

## Key Features
- IVUS/OCT Contours Registration
  - Aligns pullback sequences (rest vs. stress, diastole vs. systole) using Hausdorff distance on vessel contours and catheter centroids.
  - Supports four alignment modes:
    - *Full*: register all four phases (rest‑dia, rest‑sys, stress‑dia, stress‑sys)
    - *Double-pair*: two pairs (rest vs. stress).
    - *Single-pair*: diastole vs. systole.
    - *Single*: one phase only.
- Centerline Alignment
  - Align registered geometries onto a vessel centerline using three‑point or manual rotation methods.
- Geometry Utilities
  - Smooth contours, reorder frames to minimize spatial and index jumps, compute areas and elliptic ratios, find farthest/closest point pairs, and more.
- MRI LGE Fusion (Planned)
  - Integrate 2D LGE slices into the CCTA mesh to visualize scar/edema volumes.

## Installation

Either directly from PyPI (recommended):
```bash
pip install multimodars
```

or by cloning the repo and building the project yourself:
```bash
git clone https://github.com/yungselm/multimoda-rs.git
pip install maturin
python -m venv .venv
source .venv/bin/activate
maturin develop
```

## Quickstart Example
Run the script with the provided test cases, to ensure sufficient set up.
```python
import multimodars as mm
import numpy as np

# IVUS pullbacks: full alignment of rest/stress & diastole/systole
rest_dia, rest_sys, stress_dia, stress_sys = mm.from_file(
    mode="full",
    rest_input_path="data/ivus_rest",
    stress_input_path="data/ivus_stress"
)

# Load raw centerline
cl_raw = np.genfromtxt("data/centerline_raw.csv", delimiter=",")
centerline = mm.numpy_to_centerline(cl_raw)

# Align geometry pair onto centerline
aligned_pair = mm.to_centerline(
    mode="three_pt",
    centerline=centerline,
    geometry_pair=rest_dia,                # e.g. Diastolic geometry
    aortic_ref_pt=(12.26, -201.36, 1751.06),
    upper_ref_pt=(11.76, -202.19, 1754.80),
    lower_ref_pt=(15.66, -202.19, 1749.97)
)
```
## API Reference
For detailed signatures and usage examples, see the [online documentation](https://multimodars.readthedocs.io).  

## License
Distributed under the MIT License. See LICENSE.md for details.

## Detailed Background
This package aims to register different cardiac imaging modalities together, while coronary computed tomography angiography (CCTA) is the undisputed goldstandard for 3D information, it has several downsides, when trying to create patient-specific geometries.

First, intravascular imaging (intravascular ultrasound (IVUS) and optical coherence tomography (OCT)) have a much higher image resolution. It would therefore desirable to replace the sections along the coronary artery depicted by the intravascular images with these high resolution images. Since this intravascular images are acquired during a pullback along a catheter with a certain shape in the 3D space, and the coronary vessel undergoes several motions (heartbeat breathing), are the images inside a pullback not perfectly aligned with each other. The first aim of this package is to register these images towards each other using Hausdorff distances of the vessel contours and the catheter position (center of the image). The full backend is written in Rust leveraging parallelization to achieve much faster results than using traditional python only.

! Not implemented yet !
Second, MRI has the potential to depict several tissue characteristics, most importantly scar tissue using LGE. Again only 2D images are acquired, in this case the 2D images should be placed at the correct position in the CCTA mesh and a 3D model should be created showing the volume of scar tissue (or edema) and it's corresponding region.

### IVUS registration - gated images
The initial idea for this package, was built with a focus on coronary artery anomalies, particularly anomalous aortic origin of a coronary artery (AAOCA). In these patients a dynamic stenosis is present, where the intramural section (inside of the aortic wall) undergoes a pulsatile lumen deformation during rest and stress with every heartbeat. Additionally undergoes the vessel a stress-induced lumen deformation from rest to stress for both diastole and systole. The `from_file()` and `from_array()` functions where both built having this four possible changes in mind.
![Dynamic lumen changes](https://raw.githubusercontent.com/yungselm/multimoda-rs/main/media/dynamic_lumen_changes.png)

The options to display are therefore:

*full*
```text
`Rest`:                             `Stress`:
diastole  ---------------------->   diastole
   |                                   |
   |                                   |
   v                                   v
systole   ---------------------->   systole
```

*double pair*
```text
`Rest`:                             `Stress`:
diastole                            diastole
   |                                   |
   |                                   |
   v                                   v
systole                             systole
```

*single pair*
```text
                 `Rest`/`Stress`:
                    diastole
                       |
                       |
                       v
                    systole
```

*single*
```text
diastole rest / systole rest / diastole stress / systole stress
```

The expected input data for contours is the following for a csv file:
```text
 Expected format .csv file, e.g.:
--------------------------------------------------------------------
|      185     |       5.32     |      2.37       |        0.0     |
|      ...     |       ...      |      ...        |        ...     |
No headers -> frame index, x-coord [mm], y-coord [mm], z-coord [mm] 
```
The contours can also be in pixels, but results of the `.get_area()` function will be wrong.

### IVUS registration - pre- and post-stenting
IVUS registration works in the same way. An example is provided in `data/ivus_prestent` and `data/ivus_poststent`.

### OCT registration

