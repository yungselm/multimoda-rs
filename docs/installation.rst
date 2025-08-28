============
Installation
============

There are two ways you can use pyradiomics:
1. Install via pip
2. Install from source

------------------
1. Install via pip
------------------

Pre-built binaries are available on PyPi for installation via pip. For the python versions
mentioned below, wheels are automatically generated for each release of multimodars, allowing you to
install multimodars without having to compile anything.

* Ensure that you have ``python`` installed on your machine, version 3.12 or higher (64-bits).

* Install multimodars::

    python -m pip install multimodars

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