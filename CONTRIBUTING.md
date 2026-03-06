#  ✩°｡⋆ Contributing to the project ⋆｡°✩

First off, thank you for considering contributing! **ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧**  
Whether you’re filing a bug, proposing a new feature, improving documentation, or refactoring code, your help makes this project better for you and the whole community.

## Code of Conduct
This project adheres to the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).  
By participating, you agree to respect everyone in this community. **˗ˏˋ ♡ ˎˊ˗**

## Table of Contents

1. [Getting Started](#getting-started)  
2. [I Have a Question](#i-have-a-question)  
3. [Reporting Bugs](#reporting-bugs)  
4. [Suggesting Enhancements](#suggesting-enhancements)  
5. [Your First Code Contribution](#your-first-code-contribution)  
6. [Pull Request Process](#pull-request-process)  
7. [Coding Style & Tests](#coding-style--tests)  
8. [Writing Documentation](#writing-documentation)  
9. [Where to Get Help](#where-to-get-help)

## Getting started
1. Fork the repo and clone your fork
```bash
git clone https://github.com/yungselm/multimoda-rs.git
cd multimoda-rs
```
2. Create a new branch
```bash
git checkout -b feature/sick-feature
```
3. install dependencies and run tests
```bash
pip install -e
pytest
```
## I have a Question 
Before opening an issue:
- Read the [Documentation](https://multimoda-rs.readthedocs.io/en/latest/index.html)
- Search existing [Issues](https://github.com/yungselm/multimoda-rs/issues?q=is%3Aissue) 

If you still need help:
- Open a new issue: [Click here](https://github.com/yungselm/multimoda-rs/issues/new/choose)
- Provide:
    - A clear, descriptive title
    - Context: what you're trying to do, expected vs. actual behaviour
    - Project verion, OS/platform, Python version, and any relevant logs

## Reporting Bugs
If you find a bug, please help us fix it by opening a [a new issue](https://github.com/yungselm/multimoda-rs/issues/new/choose) and providing:
- **Title**: A short descriptive title
- **Steps to reproduce**: Minimal code snippet or sequence to trigger the bug
- **Expected behaviour** vs **Actual behaviour**
- **Environment**:
    - `multimoda-rs` version
    - Python version
    - OS and architecture

## Suggesting Enhancements
Have an idea for a new feature or an improvement to existing behaviour? Open a [feature request issue](https://github.com/yungselm/multimoda-rs/issues/new/choose) and include:
- **Summary**: A concise description of the enhancement
- **Motivation**: Why would this be useful? What problem does it solve?
- **Proposed solution**: How you envision it working (API sketch, example usage, etc.)
- **Alternatives considered**: Other approaches you've thought about

We appreciate well-scoped proposals with clear use cases.

## Your First Code Contribution
Not sure where to start? Look for issues labelled [`good first issue`](https://github.com/yungselm/multimoda-rs/issues?q=is%3Aissue+label%3A%22good+first+issue%22) or [`help wanted`](https://github.com/yungselm/multimoda-rs/issues?q=is%3Aissue+label%3A%22help+wanted%22).

**Setup** (Python + Rust hybrid via [maturin](https://github.com/PyO3/maturin)):
```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/multimoda-rs.git
cd multimoda-rs

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package in development mode (compiles the Rust extension)
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify everything works
pytest
cargo test --lib
```

## Pull Request Process
1. **Branch** – create a descriptive branch (`feature/my-feature`, `fix/issue-123`).
2. **Commit** – write clear, atomic commit messages.
3. **Tests** – add or update tests to cover your changes; all existing tests must pass.
4. **Docs** – update docstrings and, if applicable, the RST docs under `docs/`.
5. **Open the PR** – target the `main` branch. Fill in the PR template with a summary, motivation, and testing notes.
6. **Review** – address reviewer comments; once approved and CI is green, a maintainer will merge.

PRs that break existing tests or skip the pre-commit hooks will not be merged.

## Coding Style & Tests

### Python
- **Formatter**: [Black](https://black.readthedocs.io/) (enforced automatically via pre-commit).
- **Linter**: [Ruff](https://docs.astral.sh/ruff/) with `--fix` (enforced automatically via pre-commit).
- **Docstrings**: [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
  Example:
  ```python
  def my_function(x: float, y: float) -> float:
      """Short one-line summary.

      Longer description if needed.

      Parameters
      ----------
      x : float
          Description of x.
      y : float
          Description of y.

      Returns
      -------
      float
          Description of the return value.
      """
  ```
- **Type annotations**: use `from __future__ import annotations` at the top of every module and annotate all public functions and methods.
- **Tests**: place tests under `tests/` and use [pytest](https://docs.pytest.org/). Run with `pytest`.

### Rust
- **Formatter**: `cargo fmt --all` (enforced automatically via pre-commit).
- **Tests**: unit tests live alongside the source (`#[cfg(test)]` blocks). Run with `cargo test --lib`.

All checks are wired up via pre-commit and run automatically on every commit. You can also run them manually:
```bash
pre-commit run --all-files
```

## Writing Documentation
Documentation lives under `docs/` and is built with [Sphinx](https://www.sphinx-doc.org/).

- Public API is documented via **NumPy-style docstrings** (see above) — Sphinx picks these up automatically.
- Narrative docs (tutorials, how-tos) go in `.rst` files under `docs/`.
- Build locally with:
  ```bash
  cd docs
  make html
  # open _build/html/index.html in your browser
  ```
- Keep prose clear and concise. Include runnable code examples where possible.

## Where to Get Help
| Channel | When to use |
|---------|-------------|
| [Documentation](https://multimoda-rs.readthedocs.io/en/latest/index.html) | First stop, API reference and tutorials |
| [GitHub Issues](https://github.com/yungselm/multimoda-rs/issues) | Bug reports and feature requests |
| [GitHub Discussions](https://github.com/yungselm/multimoda-rs/discussions) | Questions, ideas, and general discussion |

When asking for help, please include your `multimoda-rs` version, Python version, OS, and a minimal reproducible example.


**ฅ^>⩊<^ ฅ**
------------------------------------------------------------------------------------