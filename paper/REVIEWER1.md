We thank @crnh for the thorough evaluation of our manuscript and codebase and the appreciation of our work. The detailed feedback was genuinely helpful and led to several meaningful improvements.

## Code
### Running the examples
I tested the examples with multimodars version 0.2.1, which is the latest version available on PyPI.

**Answer:** Just a heads-up, the review should now be performed on the latest version v0.2.2, which addresses many of the points raised below.

### Quickstart example in Readme
- [x] from_file should check if all required parameters are present and raise an exception with a clear message if not (the same applies to other wrapper functions). The example code does not seem to specify all required parameters.

**Answer:** I agree, the old ``from_file`` wrapper had insufficient error handling. Rather than patching it, we decided to remove the wrapper functions entirely and focus on the more explicit entry points (e.g. ``from_file_full``), which already have proper error handling in place.

*Removed:* ``multimodars/io/_wrappers.py``

- [x] The file data/centerline_raw.csv and other example data is distributed as part of the releases, which is unintuitive. This should at least be mentioned in the Readme, but ideally all example data should just be included in the repository.

**Answer:** We completely agree with you! We originally kept all example data in the repository, but as the data volume grew with added 3D functionality, attaching data to specific releases felt like the cleaner approach, it also ensures a consistent setup going forward. That said, we've made it much easier to find by linking directly to the download in the README.

*Changed:* README.md "Download [examples.zip](https://github.com/yungselm/multimoda-rs/releases/latest/download/examples.zip) (SHA256: `d11ebc7607f43ab4571fb51c9ac9178caac57774cf5d97f4f068ace4eb070fee`) from..."

### ``ivus_to_centerline.ipynb`` from the examples folder

- [x] The data folders in the examples folder (e.g. examples/data/ivus_rest) are empty, causing from_file_full to raise an unclear exception:

**Answer:** Thanks for catching this early, it was a leftover issue from a refactor where example data was moved out of the package. The correct ``examples.zip`` has now been added to the repository, and the Jupyter Notebook has been updated accordingly. The faulty _wrappers have also been removed as mentioned above.

*Added:* Complete example data added to v0.2.2

*Removed:* ``multimodars/io/_wrappers.py``

- [x] RuntimeError: Failed to prepare geometries for full processing

**Answer:** Thanks for flagging this, as discussed on the JOSS review page, this should now be resolved in v0.2.2.

- [x] The required data should be included in the data folder (I now copied it from the repository) and the exception should specify what exactly went wrong.

**Answer:** The reasoning for keeping data out of the repository is to avoid unnecessary bloat as additional 3D data is added over time. That said, if the reviewer considers this a significant usability concern, we are happy to reconsider and adjust the repository structure accordingly.

- [x] Cell 2 loads the example data from different file names than the names used to save them in cell 1 (e.g. dia_lumen.obj vs rest_dia.obj), which causes the code to raise an exception.
- [x] The notebook imports the same dependencies multiple times.
- [x] I was unable to run the stent example, because I could not find the required data in the repository or the releases.
- [x] The data required for the 'Alignment from array' example is not included in the examples archive and had to be downloaded separately from the repository. The output of the example is not visualized.
- [x] The other examples in the notebook run without errors and the visualizations are clear.

**Answer:** We thank you sincerely for your patience! The entire Jupyter Notebook has been overhauled to match v0.2.2, including updated filenames, consolidated imports in the first two cells, and all required data included in the examples archive.

*Changed:* Completely reworked examples Jupyter Notebook.

- [x] Rendering the meshes depends on nbformat, but this dependency is not listed. The notebook also depends on scipy, which is not listed.

**Answer:** I agree with the reviewer, that this can be confusing to the user. Our thinking was that the notebooks are somewhat separate from the core package, so their dependencies are only loaded if a user chooses to download the examples folder. We're happy to add dev dependencies and move the full files to the main directory if that's preferred, just let us know.

### Tutorial - Intravascular Module (from the documentation)

- [x] It would be very helpful if the tutorial could be downloaded as a Jupyter notebook right from the documentation. The documentation states the notebook should be in the releases, but I could not find it there.

**Answer:** We've added a direct download link to the latest example files at the top of the tutorial pages in the documentation.

- [x] The tutorial seems to have been written for a version of the software with a different interface, e.g. arguments like rest_input_path and rest_output_path are not supported by from_file, but it does take two parameters named input_path_a and input_path_b. However, the docstring still mentions the old parameters. Furthermore, other required parameters (e.g. label) seem to be missing from the tutorial.

**Answer:** We apologies for the confusion, the documentation has been updated to correctly reflect the current interface in v0.2.2.

### Code style and interface design

- [x] The code uses old-style annotations (e.g. List instead of list, Union instead of |). These are required if supporting Python 3.8 and older, but since this Python version is EOL, the authors should consider dropping support for it and using modern type annotations.
- [x] The Python code uses star imports, which makes the interface opaque. For example, PyContour is explicitly exported in ``__init__.py`` but imported through a star import.
- [x] The Python code uses imports outside the top level, which is generally discouraged.

**Answer:** Thank you very much for this much appreciated input. We've updated all modules to use only top-level imports and modern type annotations, and confirmed compatibility with Python >=3.10.

- [x] No type stubs are included for the parts of the interface that are implemented in Rust, which makes the interface less clear and prevents type checking of code that uses the library. The authors should consider adding stubs for at least the parts of the interface that they consider to be public. This would significantly improve the user experience. See e.g. https://www.maturin.rs/project_layout.html#adding-python-type-information and https://pyo3.rs/v0.27.1/type-stub.html for more information on how to do this.

**Answer:** We weren't aware of this feature, this was a really useful suggestion! The automatic generation approach looked promising but seemed a bit experimental and didn't appear to support function-based declarations, so we've gone with a manually written stub for now.

- [x] ``multimodars`` provides a unified interface to multiple functions through a set of wrapper functions, such as from_file and from_array. This is good, but any parameters that are required by the underlying functions are passed through **kwargs, which requires the user to consult the documentation of these functions to understand the meaning of the parameters. The interface could be improved in a few ways:
    - [x] The wrapper functions could explicitly document the meaning of all supported parameters;
    - [x] The wrapper functions could use a combination of TypedDict unpacking (to specify which parameters are supported) combined with overloads to specify which parameters are required for which mode. This would improve the user experience by enabling auto-completions and type checking for the parameters of the wrapper functions. More information: https://typing.python.org/en/latest/spec/callables.html#unpack-kwargs and https://typing.python.org/en/latest/spec/overload.html.

**Answer:** These are great suggestions and something we'd like to revisit in a future version. For now, we've removed the wrapper functions entirely where error handling and interface clarity is much more straightforward through the direct Rust bindings (e.g. ``from_file_full``).

*Removed:* ``io/_wrappers.py``

### Contributing

- [x] CONTRIBUTING.md lacks the following sections that are mentioned in the TOC: Suggesting Enhancements, Your First Code Contribution, Pull Request Process, Coding Style & Tests, Writing Documentation, Where to Get Help.
These sections should be added to the guidelines or removed from the TOC.
- [x] CONTRIBUTING.md does not mention the required code style (probably Black) and docstring style (looks like NumPyDoc).
- [x] The code style can partially be inferred from the pre-commit configuration, but it should be explicitly stated in the contributing guidelines.
- [x] The bug issue template can be improved. It currently asks which browser is used and asks for information about the smartphone running the software, both of which are not relevant for this project.

**Answer:** All of these have been addressed, the missing sections have been added to ``CONTRIBUTING.md`` (including code style, docstring conventions, PR process, etc.), and the bug report template has been cleaned up to only ask for relevant information.

*Added:* Added the corresponding sections to ``CONTRIBUTING.md`` 

*Changed:* Updated ``bug_report.md``

### Documentation

- [x] The documentation contains phrases like "There are two ways you can use pyradiomics: 1. Install via pip 2. Install from source", which should be removed or updated to match the actual module name.
- [x] https://multimoda-rs.readthedocs.io/en/latest/installation.html mentions a minimum Python version of 3.12, which does not match the minimum of 3.8 in pyproject.toml.
- [x] The project combines multiple docstring styles, e.g. NumPyDoc in from_file, Google-style in centerline_to_obj and no predefined style in read_mesh. The documentation would be improved by using a consistent style across the codebase.
- [x] The repository could link more clearly to the documentation by listing it in the 'About' section.
- [x] API documentation for core functions like from_file and from_array are difficult to find, because they are located in a wrappers submodule. The discoverability of these functions could be improved.

**Answer:** All points have been addressed. We apologize for the version inconsistency, this was an oversight on our part when updating ``pyproject.toml`` without carefully reviewing the corresponding documentation. The minimum Python version (>=3.10) is now consistently stated across both (and tested see below). We have also updated the module name throughout, standardized to NumPyDoc style across the codebase, and added the documentation link to the About section. Since the wrapper functions have been removed, their discoverability is no longer a concern.

### CI

- [x] The CI pipeline only tests the library against Python 3.12, and could be improved by testing with all supported Python versions.

**Answer:** Good point! We've updated the CI pipeline to test against all supported Python versions (>=3.10), consistent with the updated ``pyproject.toml``.

## Paper
### Authors

- [x] Christoph Gräni does not seem to have contributed to the code and the role of this author is unclear.

**Answer:** Prof. Christoph Gräni was my supervisor during the project and is responsible for the clinical direction of the work, including providing access to the patient data that the project is built around.

### Figures

- [x] The text in Figure 2 is too small.

**Answer:** The small text has been removed entirely.

### Citations

- [x] Citations are not formatted consistently. For example, in one place a paper by Stark is cited as "Stark, Anselm Walter", while in another place it is cited as "Stark, Anselm W.".
- [x] DOIs are not provided for citations 1, 4, and 6, but these are available and should be included.

**Answer:** Citation formatting has been unified and the missing DOIs have been added.

### Language

- [x] Figure 3 caption: 'inital' should be 'initial'.
- [x] Line 104: 'mutliscale' should be 'multiscale'.
- [x] Line 102: 'documentation docstrings' is a pleonasm. Remove one of the two words, or rephrase to 'documentation and docstrings'.

**Answer:** All three corrected, thank you for the careful read!