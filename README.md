<!-- Title, badges, logo -->
<h1 align="center">
    <img src="media/multimoda-rs.jpg" alt="multimoda-rs logo">
</h1>

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

> A Rust‑powered cardiac imaging multi‑modality fusion package.

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

## Example Usage
Run the script with the provided test cases, to ensure sufficient set up.
```python
import multimodars as mm

# reads in ivus contours from .csv and aligns them
rest, stress, dia, sys = mm.from_file_full("data/ivus_rest", "ivus_stress")

# get the diastolic rest geometry and align with centerline
geometry = rest.dia_geom
```

