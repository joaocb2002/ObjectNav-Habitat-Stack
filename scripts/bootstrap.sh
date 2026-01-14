#!/usr/bin/env bash
set -e

IMAGE="ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main"

echo "=== ObjectNav Habitat: bootstrap ==="

# 1) Check Docker CLI
if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker is not installed or not in PATH."
  echo "Please install Docker Engine before continuing."
  echo "Visit: https://docs.docker.com/get-docker/"
  exit 1
fi
echo "✔ Docker CLI found"

# 2) Check Docker daemon
if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker daemon is not running."
  echo "Please start the Docker daemon with the appropriate command."
  echo "For Linux: sudo systemctl start docker"
  exit 1
fi
echo "✔ Docker daemon is running"

# 3) Dataset reminder
echo
echo "⚠️  DATASETS"
echo "By default, datasets are expected at:"
echo "  \$HOME/datasets"
echo
echo "If your datasets live elsewhere, set:"
echo "  export DATA_DIR=/path/to/datasets"
echo

# 4) Pull project image
echo "Pulling project image:"
echo "  $IMAGE"
docker pull "$IMAGE"

echo
echo "✔ Bootstrap complete."
echo "You can now run:"
echo "  ./scripts/run_dev.sh python scripts/sanity_check.py"
echo "or:"
echo "  ./scripts/run_train.sh python train.py"
