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

> A Rust‑powered cardiac imaging multi‑modality fusion package using parallelization.

---
## Background


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

## Example Workflow
Run the script with the provided test cases, to ensure sufficient set up.
```python
import multimodars as mm
import numpy as np

rest, stress, dia, sys = mm.from_file(mode="full", rest_input_path="data/ivus_rest", stress_input_path="data/ivus_stress")

cl_raw = np.genfromtxt("data/centerline_raw.csv", delimiter=",")
cl = mm.numpy_to_centerline(cl_raw)

aligned = mm.to_centerline(mode="three_pt", centerline=cl, geometry_pair=rest, aortic_ref_pt=(12.2605, -201.3643, 1751.0554), upper_ref_pt=(11.7567, -202.1920, 1754.7975), lower_ref_pt=(15.6605, -202.1920, 1749.9655))
```

