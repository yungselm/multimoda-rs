---
title: 'multimodars: A Rust-powered toolkit for multi-modality cardiac image fusion and registration'
tags:
  - Rust
  - Python
  - Cardiology
  - Intravascular imaging
  - CCTA
  - Image Fusion
  - Image registration
authors:
  - name: Anselm W. Stark
    orcid: 0000-0002-5861-3753
    affiliation: [1, 2] # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Ali Mokthari
    orcid: 0009-0004-8864-6988
    affiliation: [1, 2]
  - name: Pooya Mohammadi Kazaj
    orcid: 0000-0003-2747-1489
    affiliation: [1, 2]
  - name: Marc Ilic
    orcid: 0009-0000-6443-8739
    affiliation: [1, 2]
  - name: Isaac Shiri
    orcid: 0000-0002-5735-0736
    affiliation: 1
  - name: Christoph Gräni
    orcid: 0000-0001-6029-0597
    affiliation: 1
    corresponding: true
affiliations:
  - name: Department of Cardiology, Inselspital, Bern University Hospital, University of Bern, Switzerland
    index: 1
  - name: Graduate School for Cellular and Biomedical Sciences, University of Bern, Bern, Switzerland
    index: 2
date: 15 August 2025
bibliography: bibliography.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Cardiac computed‑tomography angiography (CCTA) provides comprehensive 3D coronary anatomy but lacks the sub‑millimeter resolution and dynamic tissue insights offered by intravascular ultrasound (IVUS) or optical coherence tomography (OCT). `multimodars` bridges this gap by fusing high‑resolution pullback frames—acquired under varying hemodynamic states—with CCTA‑derived centerlines, yielding anatomically and physiologically consistent 3D vessel models. Originally conceived for anomalous aortic origin of a coronary artery (AAOCA), `multimodars` supports four gating paradigms (full, double‑pair, single‑pair, and single), enabling rest/stress and diastole/systole registration of IVUS pullbacks to capture pulsatile or stress-induced lumen deformation [@stark2025true],[@stark2025ivus]. It works additionally on OCT by e.g. registering pre- and post-stenting results or any other two pullbacks acquired in the same patient.

Built as a Maturin project, `multimodars` exposes a Rust‑powered core—parallelized with Rayon—for key algorithms (Hausdorff‐distance frame alignment, three‑point rotational fitting, Hamiltonian ordering), while offering Python/Numpy wrappers. With comprehensive CI‑driven unit and integration tests, example notebooks, and ReadTheDocs documentation, `multimodars` delivers performant, reproducible workflows for cardiovascular researchers and clinicians developing patient‑specific digital twins across multiple imaging modalities. The current modular setup is built with the aim to allow for integration of different image modalities in the future (i.e. cardiovascular magnetic resonance imaging and functional imaging) for either Python or Rust developers.

# Statement of need
Currently, different imaging modalities are assessed independently in cardiac imaging. However, when aiming for digital twin models in the future it will be essential to combine information accross different imaging modalities. Intravascular imaging lacks 3D information, however has the highest resolution of all modalities, CCTA on the other hand is the goldstandard for 3D anatomy, but has lower resolution compared to intravascular imaging and suffers from blooming artefacts in the context of coronary artery disease. The idea of image modality fusion therefore is not a new one, and several studies have already performed image-fusion of intravascular images and CCTA [@van20103d], [@kilic2020evolution], [@wu20203d] and [@bourantas2013new]. However, so far no opensource solution providing the flexibility needed in research is available. The aim of this project is to provide a flexible solution for research by providing a toolkit that can easily be implemented in differing workflows. To achieve this, we decided on a python package, since it is the most widely used programming language in research and returing and excepting numpy arrays as a combined entry point, since it is one of the most performant and widely used packages in python. To overcome performance limitations of the computationally heavy alignment, the core algorithms of the package are written in Rust, leveraging it parallelization capabilities.
The project is built having scalability in mind and in the future this can be expanded to include additional imaging modalities such as cardiac magnetic resonance imaging (CMR) for tissue characterization or functional imaging modalities.

# Features
`multimodars` is a Rust-powered, Python-accessible toolkit for multi-modality intravascular/CCTA image fusion and registration that combines a high-performance native core with a NumPy-first I/O model for easy integration into research pipelines. Core algorithms (Hausdorff-based frame alignment, three-point rotational fitting, Hamiltonian ordering) are implemented in Rust and parallelized with Rayon, while ergonomic PyO3 bindings expose a compact, well-typed data model (`PyContourPoint`, `PyContour`, `PyGeometry`, `PyGeometryPair`) that maps 1:1 to simple (N,4) NumPy arrays. The package supports four processings paradigms (full, double-pair, single-pair, single), round-trip conversion to/from CSV or binary arrays, and deterministic reconstruction of geometries from array inputs; utility methods provide centroiding, area/ellipticity metrics, smoothing, reordering, rotation/translation, and summary stenosis metrics.

![Illustration of `multimodars`'s different processing modes in the context of coronary artery anomalies. "full" returns all 4 geometry pairs (rest pulsatile lumen deformation, stress pulsatile lumen deformation, stress-induced diastolic lumen deformation, stress-induced systolic lumen deformation), "double-pair" returns pulsatile lumen deformation in rest and stress, "single-pair" can be used to compare any two states in the same patient (pulsatile lumen deformation or pre- and post-stent) and "single" aligns just the frames in a pullback.\label{aim}](figures/Figure1.jpg){width=80%}

## Overview and Datahandling
Frame alignment in `multimodars` is a two-stage registration procedure that produces a spatially and rotationally consistent mapping of intravascular frames both **within a single pullback** (intra-pullback, *align-within*) and **between paired pullbacks** (inter-pullback, *align-between*). The result is deterministic and reproducible suitable for visualization, geometric analysis and downstream processing. The algorithmic design prioritises robustness and computational efficiency, however also leaves the option for a precise but inefficient bruteforce approach.

### Overview
- **Input**: one or more `PyGeometry` objects (each a sequence of `PyContours`and a reference_point plus  optional catheter/wall layers), or pairs of `PyGeometry` for between-pullback registration. Coordinates are assumed in millimetres (mm). If now catheter or walls are provided, they can be automatically created by the algorithm.
- **Outputs**: aligned `PyGeometry` / `PyGeometryPair` objects, per-frame `AlignLog` entries (rotation, cumulative rotation, centroid translations), optional OBJ meshes for interpolated geometries.
- **Key ideas**: pick a stable reference contour, rotate/translate all frames to that baseline, find local best rotations by minimising a point-set distance (Hausdorff), use multiscale angular search to balance accuracy and cost, downsample for speed, and parallelise angle / distance computations.

The package is designed to work flawlessly with numpy, while information inside the module is stored in Hierarchical order in `PyGeometryPair`-, `PyGeometry`-, `PyContour`- and `PyContourPoint`-structs, they can easily be converted back and forth to numpy arrays. Additionally, all the different structs can be directly saved to .obj files.

![Illustration of `multimodars`'s internal data handling and the differenty entry and exit points.\label{data types}](figures/Figure2.jpg){width=80%}


# Alignment algorithms
## Intra-pullback alignment
The first stage of the alignment process establishes spatial consistency within a single pullback sequence. The proximal-most frame (i.e., the frame with the highest index) is selected as the rotational reference. Anatomical landmarks (aortic walls) are identified using the reference point provided in the CCTA geometry. The reference frame is then normalized by rotating its major axis (defined by the two farthest points) to the vertical orientation, and the aortic quadrant is rotated to the positive x-axis (rightward orientation). Subsequent proximal frames are sequentially aligned to their distal neighbors. This involves two steps: (1) translation, where the centroids of the two frames are matched, and (2) rotation, where the optimal rotation angle is found by minimizing the Hausdorff distance between the two point sets. The rotation optimization uses a multiscale approach: an initial coarse search (e.g., 1 degree steps) is followed by progressively finer searches (e.g., 0.1 and 0.01 degrees) within the neighborhood of the best coarse solution. This strategy balances accuracy and computational efficiency. The rotation applied to each frame is cumulative, ensuring that the vessel torsion is consistently propagated along the pullback.

The brute-force intra-pullback alignment algorithm requires **O(n × m² × k)** operations, scaling cubically when frame count, contour size, and angular steps are comparable. The optimized implementation reduces this to **O(n × s² × k/p)** by fixing contour size through downsampling, applying multiscale angular search to shrink *k*, and leveraging parallelization across *p* processors. In practice, this yields near-linear scaling with frame count, enabling 200-frame pullbacks to be aligned in under 15 seconds on consumer hardware.

![Illustration of `multimodars`'s internal alignment algorithm, in a first step a catheter with certain number of points is created around the original image center (the more points the higher the influence of the image centern on the final alignment), in a second step all contours distal to the reference contour are translated towards the reference contour to match it's centroid. in a last step the optimal rotation is found between a contour i and contour i+1 by doing coars rotations first 1° steps, then 0.1° in decreasing ranges to balance accuracy and comptuational cost.\label{algorithm}](figures/Figure3.jpg){width=80%}

## Inter-pullback alignment
The second stage aligns two pullbacks (e.g., rest and stress) that have been independently normalized in the first stage. The two pullbacks are aligned by first matching the centroids of their distal frames (global translation). The z-coordinates (along the pullback direction) are harmonized using averaged slice spacing. Then, the entire systolic geometry is rotated as a rigid body relative to the diastolic reference to minimize the mean Hausdorff distance across all corresponding frames.

## Parallelization
Key computational bottlenecks are addressed via data parallelism. Point rotations and distance calculations are parallelized across CPU cores using Rayon's data-parallel iterators. The Hausdorff distance computation between two frames is optimized by parallelizing the nearest-neighbor searches for each point. Additionally, the processing of independent pullbacks (e.g., rest and stress) is distributed across threads. Within each pullback, the alignment of different frames can also be processed in parallel. This hierarchical task parallelism maximizes core utilization. To maintain real-time performance, point clouds are downsampled to 200-500 points per frame without sacrificing sub-pixel accuracy. The Rust compiler's LLVM backend further accelerates computation via SIMD instructions for coordinate transformations.

# Centerline Alignment Algorithms
`multimodars` offers specialized methods for fusing intravascular imaging with CCTA-derived centerlines, particularly valuable for complex anatomies like anomalous coronary arteries. The toolkit implements two distinct alignment approaches:

The three-point registration method establishes spatial correspondence using anatomical landmarks. This algorithm identifies the optimal rigid rotation that minimizes distances between three reference points (aortic position, upper/lower sections) on the base contour and their CCTA-derived counterparts. An angular search identifies the transformation that best aligns these landmarks before propagating the rotation to all frames.

The manual alignment alternative allows direct specification of rotation angles and starting points when anatomical landmarks are ambiguous. Both methods share a common transformation pipeline: after initial orientation correction (distal-to-proximal ordering), contours undergo centroid-based translation to centerline points followed by normal vector alignment.

Key preprocessing ensures anatomical consistency. Centerlines are resampled to match contour spacing, automatically trimmed to aortic references, and reindexed. Contour normals guide rotational alignment through stable cross-product calculations. Post-alignment, the framework generates interpolated 3D meshes spanning diastolic-systolic states with UV coordinates for texturing, enabling direct visualization in biomedical software.

# Acknowledgements

# References