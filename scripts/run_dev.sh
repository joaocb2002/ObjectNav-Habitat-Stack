#!/usr/bin/env bash
set -e

IMAGE="ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main"
WORKDIR="/workspace"

# Set up data and output directories in the current working directory by default.
DATA_DIR="${DATA_DIR:-$PWD/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"

# Create data and output directories if they don't exist. If they already exist, do nothing.
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

# If no command is provided, drop into an interactive shell after installing the package.
if [ $# -eq 0 ]; then
  docker run --rm -it \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$(pwd)":$WORKDIR \
    -v "$DATA_DIR":/data:ro \
    -v "$OUTPUT_DIR":/outputs \
    -w $WORKDIR \
    $IMAGE \
    bash -lc "pip install --user -e . && exec bash"
else
  # Run the provided command after installing the package.
  docker run --rm -it \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$(pwd)":$WORKDIR \
    -v "$DATA_DIR":/data:ro \
    -v "$OUTPUT_DIR":/outputs \
    -w $WORKDIR \
    $IMAGE \
    bash -lc "pip install --user -e . && exec \"\$@\"" -- "$@"
fi
