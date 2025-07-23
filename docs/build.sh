#!/bin/bash
set -e

# Get the absolute path to the project root
PROJECT_ROOT=$(dirname $(dirname $(realpath "$0")))

# Build Rust extension
cd "$PROJECT_ROOT"
maturin develop --release

# Build docs
cd "$PROJECT_ROOT/docs"
make html