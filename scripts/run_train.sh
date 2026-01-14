#!/usr/bin/env bash
set -e

# ----------------------------
# Configuration
# ----------------------------
IMAGE="ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main"
WORKDIR="/workspace"

DATA_DIR="${DATA_DIR:-$HOME/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"

# ----------------------------
# Prepare host directories
# ----------------------------
mkdir -p "$OUTPUT_DIR"

# ----------------------------
# Run container (training)
# ----------------------------
docker run --rm \
  --gpus all \
  --ipc=host \ 
  -v "$(pwd)":$WORKDIR \
  -v "$DATA_DIR":/data:ro \
  -v "$OUTPUT_DIR":/outputs \
  -w $WORKDIR \
  $IMAGE \
  "$@"

# Differences from run_dev.sh:
    # Uses --ipc=host for shared memory
    # No interactive terminal (-it) flag (can be added if needed)
    # Intended for training runs

# Note: Add 'bash' at the end of the docker run command to start an interactive shell session inside the container.