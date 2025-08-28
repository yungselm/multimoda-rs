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
  - name: Marc Ilic
    orcid: 0009-0000-6443-8739
    affiliation: [1, 2]
  - name: Ali Mokthari
    orcid: 0009-0004-8864-6988
    affiliation: [1, 2]
  - name: Pooya Mohammadi Kazaj
    orcid: 0000-0003-2747-1489
    affiliation: [1, 2]
  - name: Isaac Shiri
    orcid: 0000-0002-5735-0736
    affiliation: 1
  - name: Christoph Gräni
    orcid: 0000-0001-6029-0597
    affiliation: 1
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
Coronary artery anomalies (CAAs) and coronary artery disease (CAD) require precise morphological and functional assessment for diagnosis and treatment planning. Cardiac computed tomography angiography (CCTA) provides comprehensive 3D coronary anatomy but lacks the sub-millimetre resolution and dynamic tissue detail available from intravascular imaging (intravascular ultrasound [IVUS], optical coherence tomography [OCT]). Developed initially to study dynamic lumen changes in CAAs, `multimodars` is a general-purpose toolkit that registers high-resolution intravascular pullbacks to CCTA-derived centerlines, resulting in locally enhanced fusion 3D vessel representations for visualization, geometric analysis, and patient-specific modelling. The toolkit supports four alignment paradigms (full, double-pair, single-pair, single) essential for comparing pullbacks acquired under different haemodynamic states (e.g., rest and stress in CAAs) or at different timepoints (e.g., pre- and post-stenting in CAD) [@stark2025true; @stark2025ivus] (see Figure 1).

# Statement of need
Combining complementary imaging modalities is critical for developing reliable 3D models and improving plaque and lumen assessment, particularly in complex conditions like coronary artery anomalies (CAAs) where dynamic vessel deformation under stress is a key diagnostic feature. Intravascular imaging offers unmatched resolution for assessing these dynamic changes but provides limited 3D context; CCTA provides 3D anatomy but suffers from lower resolution and artefacts such as blooming. The need to quantify haemodynamic-induced lumen changes in our CAA research was the primary driver for developing `multimodars`. Prior studies demonstrated intravascular/CCTA fusion [@van20103d; @kilic2020evolution; @wu20203d; @bourantas2013new] for general coronary imaging, yet no open, flexible toolkit exists that is specifically designed for multi-state analysis (e.g., rest/stress, pre/post intervention) and balances high performance, deterministic behaviour, and easy integration into research workflows. `multimodars` addresses this gap by providing a Python package with a high-performance Rust backend that links to numpy arrays, enabling reproducible experiments and scalable processing for both CAAs and general CAD applications.

![Illustration of `multimodars`'s different processing modes, demonstrated in the context of coronary artery anomalies where they were first applied. The "full" mode returns all four geometry pairs necessary to analyze rest and stress haemodynamics (1: rest pulsatile deformation, 2: stress pulsatile deformation, 3: stress-induced diastolic deformation, 4: stress-induced systolic deformation). The "double-pair" mode returns pulsatile deformation in rest and stress. The "single-pair" mode can be used to compare any two states in the same patient (e.g., pulsatile deformation or pre- and post-stent). The "single" mode aligns frames within a single pullback.\label{Figure 1: Aim}](figures/Figure1.jpg){width=80%}

# Features
`multimodars` couples a performant native core with a compact typed data model (PyContourPoint, PyContour, PyGeometry, PyGeometryPair) that maps losslessly to simple (N,4) NumPy arrays (see Figure 2) with the 4 dimensions representing frame id (to group points), x-, y- and z-coordinates. The package supports round-trip CSV/NumPy I/O, deterministic reconstruction of geometries, optional OBJ export (with deformation mapping), and utilities for centroiding, area and ellipticity metrics, smoothing, reordering, rigid transforms (rotation/translation), and stenosis summaries. Users may select processing modes to return either all four geometry pairs (full), pulsatile deformation in rest and stress (double-pair), any two states (single-pair), or intra-pullback alignment only (single).

These modes were specifically designed to enable the analysis of haemodynamic responses in CAA (e.g., quantifying pulsatile and stress-induced deformation) but are equally applicable to any longitudinal comparison, such as assessing the results of percutaneous coronary intervention in CAD.

The API is designed for easy embedding in analysis pipelines while allowing users to trade accuracy vs speed through downsampling and angular resolution. Additionally all bottlenecks are parallelised.

![Illustration of `multimodars`'s internal data handling and the different entry and exit points.\label{Figure 2: Data Types}](figures/Figure2.jpg){width=80%}

# Algorithms
Alignment is a two-stage procedure producing spatially and rotationally consistent mappings both within single pullbacks (intra-pullback) and between paired pullbacks (inter-pullback). Intra-pullback alignment selects the proximal-most frame as rotational reference, then sequentially aligns proximal frames to distal neighbours by centroid translation and rotation minimising a point-set distance (Hausdorff). Rotation uses a multiscale angular search (coarse to fine, e.g., 1° → 0.1° → 0.01°) and cumulative rotation propagation to preserve vessel torsion. Naive brute-force complexity is $O(n × (\frac{R}{S}) × m²)$ (n=number of frames, m=number of points, R=range, S=step size); the optimized pipeline fixes contour size via downsampling, reduces angular search space with multiscale refinement resulting in $O(n × (R + c) × m²)$ making it independent of step size for small S (see Figure 3).

Inter-pullback alignment harmonizes distal centroids, averages slice spacing to align z-coordinates, and applies a rigid rotation to one geometry to minimize mean Hausdorff distance across corresponding frames. For fusion with CCTA centerlines, two complementary strategies are provided: a three-point registration that matches anatomical landmarks (e.g., aortic position and section markers) and a manual alignment mode for ambiguous anatomies; both pipelines resample centerlines to contour spacing, translate centroids to matched points, align normals via cross-product computations, and optionally generate interpolated 3D meshes with UV coordinates for visualization.

The algorithm uses the undirected Hausdorff distance, which is the maximum of the directed distances from set A to B and from set B to A. The directed Hausdorff distance from set A to B is defined as: $$h(A,B)=\underset{a \in A}{max} \underset{b \in B}{min}||a - b ||$$
where $||a - b||$ is the Euclidean distance. The objective is to minimize this distance during rotation search. In the implementation, the mean of the directed distances is used as a similarity metric for inter-pullback alignment, weighted by contour ellipticity to prioritize non-round lumens in stenotic segments.

![Illustration of `multimodars`'s internal alignment algorithm, in a first step a catheter with certain number of points is created around the original image center (the more points the higher the influence of the image center on the final alignment), in a second step all contours distal to the reference contour are translated towards the reference contour to match its centroid. In a last step the optimal rotation is found between a contour i (reference) and contour i+1 (target) by doing coarse rotations first 1° steps, then 0.1° in decreasing ranges to balance accuracy and computational cost.\label{Figure 3: Alignment Algorithm}](figures/Figure3.jpg){width=80%}

**Clinical impact in CAA and beyond:** This precise, deterministic alignment was developed to quantify dynamic vessel deformation under different haemodynamic states (e.g., rest vs. stress) in coronary artery anomalies, a primary application and driver for this work. It provides crucial insights for diagnosing CAAs, planning interventions, and developing patient-specific 3D models. Furthermore, the same functionality is directly applicable to assessing longitudinal changes in coronary artery disease, such as pre- vs. post-stent deployment, making `multimodars` a versatile tool for the broader cardiovascular research community.

# Performance and parallelisation
Performance bottlenecks are addressed with hierarchical data parallelism: Rayon's iterators parallelise point rotations and nearest-neighbour searches within Hausdorff computations, independent pullbacks and independent frames are processed concurrently where dependencies allow, and Rust/LLVM enables SIMD optimisations for coordinate transforms. Typical workflows downsample contours to 200–500 points per frame to preserve sub-pixel accuracy while reducing compute. On consumer hardware this approach reduces alignment times for 200-frame pullbacks to sub one minute, subject to chosen parameters and CPU resources. As an example on a 16-core CPU, for an OCT pullback with 280 frames, a rotation range of ±3° degrees and an accuracy of 0.01° the performance time can be reduced from ~150 seconds to 18 seconds with a mean difference of -0.01° over all frames. For an accuracy of 0.1° from 19 seconds to 14 seconds, while for an accuracy of 1° there is no performance difference (5 seconds).

# Implementation, reproducibility and usage
The core is implemented in Rust and exposed to Python via PyO3 and packaging uses Maturin. The package is available on PyPI for direct pip installation. The API is NumPy-centric to simplify integration with existing tools and additional CSV entry (e.g., from [AIVUS-CAA](https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA) [@stark2025automated]) are supported. Deterministic behaviour, unit and integration tests in CI, example notebooks with example data, and hosted documentation with a dedicated tutorial section ([ReadTheDocs](https://multimoda-rs.readthedocs.io/en/latest/)) facilitate reproducible workflows. Users can tune downsampling, catheter reconstruction and angular resolution to balance precision and throughput.

# Acknowledgements
None

# References