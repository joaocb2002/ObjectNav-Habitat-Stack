#!/usr/bin/env bash
set -e

IMAGE="ghcr.io/joaocb2002/object-nav-habitat-stack/habitat-project:main"
WORKDIR="/workspace"

DATA_DIR="${DATA_DIR:-$HOME/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs}"

mkdir -p "$OUTPUT_DIR"

docker run --rm -it \
  --gpus all \
  -v "$(pwd):$WORKDIR" \
  -v "$DATA_DIR:/data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  -w "$WORKDIR" \
  "$IMAGE" \
  "$@"
