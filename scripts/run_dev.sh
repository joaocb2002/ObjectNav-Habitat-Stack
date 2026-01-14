#!/usr/bin/env bash
set -e

# Docker image configuration
IMAGE="ghcr.io/joaocb2002/object-nav-habitat-stack/habitat-project:main"
WORKDIR="/workspace"

# Set default directories (can be overridden by environment variables)
DATA_DIR="${DATA_DIR:-$HOME/datasets}"  # Default: ~/datasets
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"  # Default: ./outputs

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the Docker container with GPU support
# Mount current directory, data, and outputs
docker run --rm -it \
  --gpus all \  # Enable all GPUs
  -v "$(pwd)":$WORKDIR \  # Mount current directory
  -v "$DATA_DIR":/data:ro \  # Mount data directory (read-only)
  -v "$OUTPUT_DIR":/outputs \  # Mount outputs directory
  -w $WORKDIR \
  $IMAGE \
  "$@"  # Pass all script arguments to the container

  # Folder structure inside the container:
    # /workspace      -> Current project directory (host's pwd)
    # /data           -> Mounted data directory
    # /outputs       -> Mounted outputs directory