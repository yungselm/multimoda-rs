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
Cardiac computed tomography angiography (CCTA) provides comprehensive 3D coronary anatomy but lacks the sub-millimetre resolution and dynamic tissue detail available from intravascular imaging (IVUS, OCT). `multimodars` registers high-resolution intravascular pullbacks to CCTA-derived centerlines resulting in locally enhanced fusion 3D vessel representations for visualization, geometric analysis, and patient-specific modelling. The toolkit supports four gating paradigms (full, double-pair, single-pair, single) to compare pullbacks acquired under different haemodynamic states [@stark2025true; @stark2025ivus] (see Figure 1). Implemented as a Maturin package, `multimodars` exposes a Rust core parallelised with Rayon and ergonomic PyO3 bindings that accept and return NumPy arrays. The project includes CI tests, an example notebook, and ReadTheDocs documentation to support reproducible workflows for researchers and clinicians developing patient-specific digital twins. Importantly, the workflow is modality-agnostic and applies equally to OCT and to longitudinal comparisons such as pre- and post-stenting in coronary artery disease (or any continuous imaging series in the same patient).

# Statement of need
Combining complementary imaging modalities is critical for developing reliable digital twins and improving plaque and lumen assessment. Intravascular imaging offers unmatched resolution but limited 3D context; CCTA provides 3D anatomy but suffers from lower resolution and artefacts such as blooming. Prior studies demonstrated intravascular/CCTA fusion [@van20103d; @kilic2020evolution; @wu20203d; @bourantas2013new], yet no open, flexible toolkit exists that balances high performance, deterministic behaviour, and easy integration into research workflows. `multimodars` addresses this gap by providing a Python package with a high-performance Rust backend that links to numpy arrays, enabling reproducible experiments and scalable processing. Choosing a Python package rather than a standalone software has several advantages: researchers are not constrained to a specific graphical interface or workflow, but can seamlessly integrate multimodal fusion into their own analysis pipelines. Python’s ubiquity in scientific computing lowers the barrier to adoption, while the Rust backend provides the computational efficiency of low-level languages. This combination makes `multimodars` both accessible and performant, ensuring it can serve as a foundation for method development, benchmarking, and large-scale studies.

![Illustration of `multimodars`'s different processing modes in the context of coronary artery anomalies. "full" returns all 4 geometry pairs (rest pulsatile lumen deformation, stress pulsatile lumen deformation, stress-induced diastolic lumen deformation, stress-induced systolic lumen deformation), "double-pair" returns pulsatile lumen deformation in rest and stress, "single-pair" can be used to compare any two states in the same patient (pulsatile lumen deformation or pre- and post-stent) and "single" aligns just the frames in a pullback.\label{Figure 1: Aim}](figures/Figure1.jpg){width=80%}

# Features
`multimodars` couples a performant native core with a compact typed data model (PyContourPoint, PyContour, PyGeometry, PyGeometryPair) that maps losslessly to simple (N,4) NumPy arrays (see Figure 2) with the 4 dimensions representing frame id (to group points), x-, y- and z-coordinates. The package supports round-trip CSV/NumPy I/O, deterministic reconstruction of geometries, optional OBJ export (with deformation mapping), and utilities for centroiding, area and ellipticity metrics, smoothing, reordering, rigid transforms (rotation/translation), and stenosis summaries. Users may select processing modes to return either all four geometry pairs (full), pulsatile deformation in rest and stress (double-pair), any two states (single-pair), or intra-pullback alignment only (single). The API is designed for easy embedding in analysis pipelines while allowing users to trade accuracy vs speed through downsampling and angular resolution. Additionally all bottlenecks are parallelised.

![Illustration of `multimodars`'s internal data handling and the different entry and exit points.\label{Figure 2: Data Types}](figures/Figure2.jpg){width=80%}

# Algorithms
Alignment is a two-stage procedure producing spatially and rotationally consistent mappings both within single pullbacks (intra-pullback) and between paired pullbacks (inter-pullback). Intra-pullback alignment selects the most proximal frame as rotational reference, normalizes orientation via the major axis and aortic quadrant. Distal frames are sequentially aligned with their proximal neighbouring frame by minimising the Hausdorff distance between point sets through centroid translation and rotation. Rotation uses a multiscale angular search (coarse to fine, e.g., 1° → 0.1° → 0.01°) and cumulative rotation propagation to preserve vessel torsion. Naive brute-force complexity is **O(n × m² × k)** (n=number of frames, m=number of points, k=angle steps); the optimized pipeline fixes contour size via downsampling, reduces angular search space with multiscale refinement, and parallelises computations to achieve near-linear scaling with frame count in practice (see Figure 3).

Inter-pullback alignment harmonizes distal centroids, averages slice spacing to align z-coordinates, and applies a rigid rotation to one geometry to minimize mean Hausdorff distance across corresponding frames. For fusion with CCTA centerlines, two complementary strategies are provided: a three-point registration that matches anatomical landmarks (e.g., aortic position and section markers) and a manual alignment mode for ambiguous anatomies; both pipelines resample centerlines to contour spacing, translate centroids to matched points, align normals via cross-product computations, and optionally generate interpolated 3D meshes with UV coordinates for visualization.

The algorithm uses the undirected Hausdorff distance, which is the maximum of the directed distances from set A to B and from set B to A. The directed Hausdorff distance from set A to B is defined as: $$h(A,B)=\underset{a \in A}{max} \underset{b \in B}{min}||a - b ||$$
where $||a - b||$ is the Euclidean distance. The objective is to minimize this distance during rotation search. In the implementation, the mean of the directed distances is used as a similarity metric for inter-pullback alignment, weighted by contour ellipticity to prioritize non-round lumens in stenotic segments.

![Illustration of `multimodars`'s internal alignment algorithm, in a first step a catheter with certain number of points is created around the original image center (the more points the higher the influence of the image center on the final alignment), in a second step all contours distal to the reference contour are translated towards the reference contour to match its centroid. in a last step the optimal rotation is found between a contour i and contour i+1 by doing coarse rotations first 1° steps, then 0.1° in decreasing ranges to balance accuracy and computational cost.\label{Figure 3: Alignment Algorithm}](figures/Figure3.jpg){width=80%}

**Clinical impact:** This precise, deterministic alignment is clinically valuable as it enables the accurate quantification of dynamic vessel deformation under different haemodynamic states (e.g., rest vs. stress) and the assessment of longitudinal changes (e.g., pre- vs. post-stent). This provides crucial insights for diagnosing coronary anomalies, planning interventions, and developing patient-specific digital twins.

# Performance and parallelisation
Performance bottlenecks are addressed with hierarchical data parallelism: Rayon's iterators parallelise point rotations and nearest-neighbour searches within Hausdorff computations, independent pullbacks and independent frames are processed concurrently where dependencies allow, and Rust/LLVM enables SIMD optimisations for coordinate transforms. Typical workflows downsample contours to 200–500 points per frame to preserve sub-pixel accuracy while reducing compute. On consumer hardware this approach reduces alignment times for 200-frame pullbacks to sub one minute, subject to chosen parameters and CPU resources. As an example on a 16-core CPU, for a OCT pullback with 280 frames, a rotation range of ±3° degrees and an accuracy of 0.01° the performance time can be reduced from ~150 seconds to 18 seconds with a mean difference of -0.01° applying the algorithm. For an accuracy of 0.1° from 19 seconds to 14 seconds, while for an accuracy of 1° there is no performance difference (5 seconds).

# Implementation, reproducibility and usage
The core is implemented in Rust and exposed to Python via PyO3 and packaging uses Maturin. The package is available on PyPI for direct pip installation. The API is NumPy-centric to simplify integration with existing tools and additional CSV entry (e.g., from AIVUS-CAA [@stark2025automated]) are supported. Deterministic behaviour, unit and integration tests in CI, example notebooks with example data, and hosted documentation with a dedicated tutorial section (ReadTheDocs) facilitate reproducible workflows. Users can tune downsampling, catheter reconstruction and angular resolution to balance precision and throughput.

# Acknowledgements
None

# References