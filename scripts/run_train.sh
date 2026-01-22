#!/usr/bin/env bash
set -e

IMAGE="ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main"
WORKDIR="/workspace"

# Set up data and output directories in the current working directory by default.
DATA_DIR="${DATA_DIR:-$PWD/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"

# Create data and output directories if they don't exist. If they already exist, do nothing.
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

if [ $# -eq 0 ]; then
  echo "Usage: ./scripts/run_train.sh <command...>"
  echo "Example: ./scripts/run_train.sh python train.py"
  exit 2
fi

docker run --rm \
  --gpus all \
  --ipc=host \
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

# Differences from run_dev.sh:
    # Uses --ipc=host for shared memory
    # No interactive terminal (-it) flag (can be added if needed)
    # Intended for training runs

# Note: Add 'bash' at the end of the docker run command to start an interactive shell session inside the container.