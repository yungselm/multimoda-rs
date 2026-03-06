We thank @crnh for the careful evaluation of our manuscript and codebase and the appreciation of our work.

## Code
### Running the examples
I tested the examples with multimodars version 0.2.1, which is the latest version available on PyPI.

**Answer:** Please not that the review is now performed on the latest version v0.2.2

### Quickstart example in Readme
- [x] from_file should check if all required parameters are present and raise an exception with a clear message if not (the same applies to other wrapper functions). The example code does not seem to specify all required parameters.

**Answer:** I agree that the current from_file wrapper had insufficient error handling. However, for the current version I decided to remove the wrapper functions entirely from the project and focus on the more explicit entry points (e.g. from_file_full), which have  error handling already implemented.

*Removed:* multimodars/io/_wrappers.py

- [x] The file data/centerline_raw.csv and other example data is distributed as part of the releases, which is unintuitive. This should at least be mentioned in the Readme, but ideally all example data should just be included in the repository.

**Answer:** The reviewer is correct that this is less intuitive, and I originally had all the example data in the repository. Since the datavolume kept increasing, with added 3D functionality, I decided on attaching data to specific relases, this ensures the same set up even if more 3D model data will be provided in the future. However, I made it clearer how to access the data by linking in the README.md

*Changed:* README.md "Download the [example.zip](https://github.com/yungselm/multimoda-rs/releases/tag/v0.2.2) from the latest release..."

### ``ivus_to_centerline.ipynb`` from the examples folder

- [x] The data folders in the examples folder (e.g. examples/data/ivus_rest) are empty, causing from_file_full to raise an unclear exception:

**Answer:** I thank the reviewer for spotting this error early. As I previously explained, this error occured due to a refactor where example data was moved out of the package. I now added the correct ``examples.zip`` to the github repository, and updated the Jupyter Notebook. Additionally as stated above removed the faulty _wrappers.

*Added:* Complete example data added to v0.2.2

*Removed:* multimodars/io/_wrappers.py

- [x] RuntimeError: Failed to prepare geometries for full processing

**Answer:** I thank the reviewer for their valuable feedback, as discussed on the JOSS review page, this should now be resolved.

- [x] The required data should be included in the data folder (I now copied it from the repository) and the exception should specify what exactly went wrong.

**Answer:** As stated above, the reasoning for this was to keep the repository lean for future additional data. If the reviewer does not agree with this design choice, I will adjust the repository accordingly.

- [x] Cell 2 loads the example data from different file names than the names used to save them in cell 1 (e.g. dia_lumen.obj vs rest_dia.obj), which causes the code to raise an exception.
- [] The notebook imports the same dependencies multiple times.
- [x] I was unable to run the stent example, because I could not find the required data in the repository or the releases.
- [x] The data required for the 'Alignment from array' example is not included in the examples archive and had to be downloaded separately from the repository. The output of the example is not visualized.
- [x] The other examples in the notebook run without errors and the visualizations are clear.

**Answer:** I thank the reviewer for his patience, I updated the whole Jupyter notebook, to match the new version v0.2.2.

*Changed:* Completely overworked examples Jupyter Notebook.

- [x] Rendering the meshes depends on nbformat, but this dependency is not listed. The notebook also depends on scipy, which is not listed.

**Answer:** I agree with the reviewer, that this can be confusing to the user. My idea was that the Jupyter notebooks are something seperate from the core repository, therefore the dependencies are loaded only if a user decides to download the examples folder. If the reviewer wants this changed I will add dev dependencies and add the full files to the main directory.

### Tutorial - Intravascular Module (from the documentation)

- [x] It would be very helpful if the tutorial could be downloaded as a Jupyter notebook right from the documentation. The documentation states the notebook should be in the releases, but I could not find it there.

**Answer:** I added the corresponding section on top of the tutorial pages in the documentation, with a direct download link to the latest example files.

- [x] The tutorial seems to have been written for a version of the software with a different interface, e.g. arguments like rest_input_path and rest_output_path are not supported by from_file, but it does take two parameters named input_path_a and input_path_b. However, the docstring still mentions the old parameters. Furthermore, other required parameters (e.g. label) seem to be missing from the tutorial.

**Answer:** I am very sorry about this confusion, I have now correctly updated documentation again to match the new version.

### Code style and interface design

- [x] The code uses old-style annotations (e.g. List instead of list, Union instead of |). These are required if supporting Python 3.8 and older, but since this Python version is EOL, the authors should consider dropping support for it and using modern type annotations.
- [x] The Python code uses star imports, which makes the interface opaque. For example, PyContour is explicitly exported in ``__init__.py`` but imported through a star import.
- [x] The Python code uses imports outside the top level, which is generally discouraged.

**Answer:** Thank you very much for this careful evaulation, I adusted all the modules to use only top level imports and adjusted the type annotation.

- [x] No type stubs are included for the parts of the interface that are implemented in Rust, which makes the interface less clear and prevents type checking of code that uses the library. The authors should consider adding stubs for at least the parts of the interface that they consider to be public. This would significantly improve the user experience. See e.g. https://www.maturin.rs/project_layout.html#adding-python-type-information and https://pyo3.rs/v0.27.1/type-stub.html for more information on how to do this.

**Answer:** I did not know this feature! The automatic generation sounds very interesting, however since it was tagged as still experimental I decided for a manual version for now, since it also does not seem to support the function-based declaration.

- [x] ``multimodars`` provides a unified interface to multiple functions through a set of wrapper functions, such as from_file and from_array. This is good, but any parameters that are required by the underlying functions are passed through **kwargs, which requires the user to consult the documentation of these functions to understand the meaning of the parameters. The interface could be improved in a few ways:
    - [x] The wrapper functions could explicitly document the meaning of all supported parameters;
    - [x] The wrapper functions could use a combination of TypedDict unpacking (to specify which parameters are supported) combined with overloads to specify which parameters are required for which mode. This would improve the user experience by enabling auto-completions and type checking for the parameters of the wrapper functions. More information: https://typing.python.org/en/latest/spec/callables.html#unpack-kwargs and https://typing.python.org/en/latest/spec/overload.html.

**Answer:** These were very helpful suggestions by the reviewer, I decided for now to remove them entirely, since error handling and so on is much easier by using the Rust bindings (e.g. from_file_full).

### Contributing

- [x] CONTRIBUTING.md lacks the following sections that are mentioned in the TOC: Suggesting Enhancements, Your First Code Contribution, Pull Request Process, Coding Style & Tests, Writing Documentation, Where to Get Help.
These sections should be added to the guidelines or removed from the TOC.
- [x] CONTRIBUTING.md does not mention the required code style (probably Black) and docstring style (looks like NumPyDoc).
- [x] The code style can partially be inferred from the pre-commit configuration, but it should be explicitly stated in the contributing guidelines.

**Answer:** Thank you very much for highlighting this important point. I added the corresponding sections to the CONTRIBUTING.md file.

- [x] The bug issue template can be improved. It currently asks which browser is used and asks for information about the smartphone running the software, both of which are not relevant for this project.

**Answer:** I thank the reviewer for highlighting this, I adjusted the report sheet accordingly.

### Documentation

- [x] The documentation contains phrases like "There are two ways you can use pyradiomics: 1. Install via pip 2. Install from source", which should be removed or updated to match the actual module name.

**Answer:** I thank the reviewer for highlighting this. I changed to ``multimodars``.

- [] https://multimoda-rs.readthedocs.io/en/latest/installation.html mentions a minimum Python version of 3.12, which does not match the minimum of 3.8 in pyproject.toml.
- [] The project combines multiple docstring styles, e.g. NumPyDoc in from_file, Google-style in centerline_to_obj and no predefined style in read_mesh. The documentation would be improved by using a consistent style across the codebase.
- [] The repository could link more clearly to the documentation by listing it in the 'About' section.
- [] API documentation for core functions like from_file and from_array are difficult to find, because they are located in a wrappers submodule. The discoverability of these functions could be improved.

### CI

- [] The CI pipeline only tests the library against Python 3.12, and could be improved by testing with all supported Python versions.

## Paper
### Authors

- [] Christoph Gräni does not seem to have contributed to the code and the role of this author is unclear.

### Figures

- [x] The text in Figure 2 is too small.

**Answer:** Removed the small text entirely.

### Citations

- [x] Citations are not formatted consistently. For example, in one place a paper by Stark is cited as "Stark, Anselm Walter", while in another place it is cited as "Stark, Anselm W.".
- [x] DOIs are not provided for citations 1, 4, and 6, but these are available and should be included.

**Answer:** Unified the citation style.

### Language

- [x] Figure 3 caption: 'inital' should be 'initial'.
- [x] Line 104: 'mutliscale' should be 'multiscale'.
- [x] Line 102: 'documentation docstrings' is a pleonasm. Remove one of the two words, or rephrase to 'documentation and docstrings'.

**Answer:** Adjusted all three points.