# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.

# Centerline alignment module
(- [] Create Centerline directly from mesh) (v1.0.0)

# Mesh combination (v0.3.0)
- [] Write a more stable function to find intramural points on aorta (for now works with excluded RCA points)
- [] fix centerline resampling
- [] Create an aortic and coronary wall with a consistent connection

# CCTA clinical assessment (v0.4.0)
- [] automatically find the required reference points for alignment
- [] automatically assess lumen area, minor-, major axis, mla, intramural length etc. from the CCTA mesh.
- [] return hausdorff distances per point between aligned ivus mesh and ccta mesh.
- [] function to automatically assign wall thickness to ccta and combined ivus/ccta mesh
- [] helper function to automatically export only certain points on mesh (e.g. RCA or LCA)
- [] function to combine anomalous section only with the distal part

# Miscellaneous
- [] Improve print statements in the whole package

# Prepublish checklist (JOSS review feedback — open until resolved):

## Data & archival
- [] Upload example data (`examples.zip`) to a persistent archive (Zenodo or similar) with a DOI.
  Upload the source code snapshot there as well, either in the same record or a separate one linked from the paper.
  Leave open until the DOI is minted and cited in the paper/README.
- [] Clarify whether the sample data (repo + examples.zip) is real human data or synthetic data.
  If real, add a statement on ethical approval / data-use conformity per JOSS policy
  (https://joss.readthedocs.io/en/latest/policies.html#joss-policies).

## Dependencies / packaging
- [x] Add `scipy` as an explicit runtime dependency in `pyproject.toml` (required by `mm.label_geometry()`).
- [x] Convert `optional-dependencies` groups that are not user-facing extras (docs, test, dev)
  to PEP 735 `dependency-groups` so they are not shipped with the package on PyPI.
  Note: `trimesh` was **not** moved to dev — it is imported throughout the core package
  (`_converters.py`, `ccta/` modules) and must remain a runtime dependency.
  Moved `plotly`, `nbformat`, `ipykernel` into `[dependency-groups.dev]`; `[dependency-groups.test]`
  and `[dependency-groups.docs]` replace the former optional-dependency groups.
  `viz` and `meshlab` remain as `[project.optional-dependencies]` (user-facing extras).

## Examples / notebooks
- [x] Include the `examples/` folder (with data) directly in the repository so users can clone
  and run without a separate download, and so CI can execute the notebooks as tests.
- [] Fix `ivus_to_centerline.ipynb`: create `output/` and `output/unprocessed/` directories
  on-the-fly with `os.makedirs(..., exist_ok=True)` so the notebook does not error on a fresh clone.
- [] Add a unit test that runs the example notebook end-to-end (e.g. using `nbmake` or `pytest-notebook`)
  so regressions are caught automatically.

## Performance claim
- [] Provide reproducible code + sample data for the performance claim in the paper (lines 102-104):
  "Empirical performance on a 16-core CPU: an OCT pullback with 280 frames and a rotation search range
  of ±3° (final accuracy 0.01°) saw alignment time reduced from 150 s to 18 s with the optimized
  multiscale search."
  Add a benchmark script or notebook to `examples/` so the claim can be independently reproduced.

## Documentation — Tutorial Intravascular
- [x] Section "1. Workflow csv files": add column descriptions with units after the data table,
  and add header names to the first data table — done (frame_index, x (mm), y (mm), z (mm)
  headers present; "Column descriptions:" bullet block added).
- [x] Section "1. Workflow csv files": add `:func:` / `:class:` Sphinx cross-references —
  done throughout section 2 and the rest of the tutorial.
- [x] Section "3. Workflow numpy arrays": explain how CSV and numpy workflows differ — done via
  `.. note:: **CSV vs. numpy workflow** …` block at the top of section 3.
- [x] Fix section numbering — done: sections run 1–7 with no duplicates.
- [x] Section "6. Utility functions / to_array": clarify Returns block — done with prose
  descriptions and typed bullet list per return type; `:func:` cross-reference present.
- [x] Fix `translate((0.0, 1.0, 2.0))` tuple bug — done: all three call-sites use positional
  args `translate(0.0, 1.0, 2.0)`.
- [] Fix remaining undefined variables in tutorial snippets (reviewer: "cannot be run due to
  undefined variables"):
  - `stress` (section 6, to_array snippet, line ~392) — only defined if the user followed the
    `from_file_full` path; breaks on the `singlepair` path.  Change to `rest` or add a note.
  - `cl_resampled` (section 6, to_array snippet, line ~394) — variable returned in section 4
    is named `resampled_cl`; rename to match.
  - `results["rca_points"]` (section 4, align_combined code block, line ~358) — `results` is
    never defined in this tutorial.  Add an inline comment: `# from mm.label_geometry() — see
    CCTA tutorial` and a placeholder assignment above the block.
  - `geometry` (section 5, to_obj code block, line ~374) — undefined at that point; replace
    with `aligned` (returned from align_combined in section 4).

## Documentation — general
- [x] Consider converting tutorials to Jupyter notebooks rendered via `nbsphinx` or `myst-nb`
  so outputs are visible, results are verifiable, and documentation build failures reveal
  broken code early. (Non-blocking suggestion from reviewer crnh / richardkoehler.)
- [] Cite the case-report paper in the Research impact statement (it is cited elsewhere but
  not in that section).

## CI / quality
- [] Integrate coverage measurement (`pytest-cov`) and upload reports to Coveralls or Codecov.
- [] Add a coverage status badge to the README.
- [] Add `mypy` to pre-commit hooks.
- [] Add `scipy` to pre-commit / lint hooks.
