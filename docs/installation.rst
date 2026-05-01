============
Installation
============

There are three ways you can install multimodars:

1. Install via pip
2. Install from source
3. Developer setup (contributing / building docs)

------------------
1. Install via pip
------------------

Pre-built binaries are available on PyPI for installation via pip. For the python versions
mentioned below, wheels are automatically generated for each release of ``multimodars``, allowing you to
install multimodars without having to compile anything.

* Ensure that you have ``python`` installed on your machine, version 3.10 or higher (64-bits).

* Install multimodars::

    python -m pip install multimodars

Optional extras for mesh visualisation or MeshLab integration::

    pip install "multimodars[viz]"
    pip install "multimodars[meshlab]"

----------------------
2. Install from source
----------------------

multimodars can be installed from source with the following steps.

* Clone the repository and install Rust and Maturin::

    # Install rust in case you don't have it on your system
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    git clone https://github.com/yungselm/multimoda-rs.git
    python -m venv .venv
    source .venv/bin/activate
    pip install maturin
    . "$HOME/.cargo/env" # Set rust env
    maturin develop

.. note::

   In case you get the following error::

    💥 maturin failed
    Caused by: rustc, the rust compiler, is not installed or not in PATH.
    This package requires Rust and Cargo to compile extensions. Install it
    through the system's package manager or via https://rustup.rs/.

   execute the following commands::

    unset -v VIRTUAL_ENV
    maturin develop

-------------------
3. Developer Setup
-------------------

Clone the repo and install with the ``dev`` dependency group, which includes testing tools
(``pytest``, ``ruff``, ``black``, ``mypy``) and notebook utilities (``plotly``, ``ipykernel``,
``nbmake``)::

    git clone https://github.com/yungselm/multimoda-rs.git
    cd multimoda-rs
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

    # Install Rust and build the extension
    . "$HOME/.cargo/env"
    pip install maturin
    maturin develop

    # Install Python dev dependencies (requires pip >= 25.0 or uv)
    pip install --group dev
    # or: uv sync --group dev

Optional extras::

    pip install -e ".[viz]"
    pip install -e ".[meshlab]"

**Running tests:**

.. code-block:: bash

    pytest               # Python test suite
    cargo test --lib     # Rust unit tests

**Building docs:**

.. code-block:: bash

    pip install --group docs          # install doc dependencies
    cd docs && make html              # output: docs/_build/html/index.html
