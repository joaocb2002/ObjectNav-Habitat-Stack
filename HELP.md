# HELP (Maintainer notes)

This file contains **maintenance and contributor operations** that are intentionally kept out of the main README.

---

## Dependency management and lockfiles

The project uses a two-layer Docker image design:

- `docker/base/`: heavy, slow-changing dependencies (Habitat-Sim / Habitat-Lab, CUDA bindings, system libs)
- `docker/project/`: repo-specific Python dependencies that change more frequently

### Base image: conda lock

Source:

- `docker/base/environment.yml`

Lockfile:

- `docker/base/conda-lock.yml`

Regenerate the lockfile:

```bash
conda-lock -f docker/base/environment.yml -p linux-64 --lockfile docker/base/conda-lock.yml
```

Notes:

- Use the same `conda-lock` version as the base Dockerfile expects.
- If you change the Python version in the environment, update related tooling accordingly.

### Project image: pip lock (pip-tools)

Source:

- `docker/project/requirements.txt`

Expected lockfile:

- `docker/project/requirements.lock` (hashed, reproducible)

Regenerate the lockfile (run from within the project image / matching Python version):

```bash
python -m piptools compile \
  --generate-hashes \
  --allow-unsafe \
  -o docker/project/requirements.lock \
  docker/project/requirements.txt
```

---

## Image build / publish behavior

CI is expected to rebuild images when relevant directories change:

- Changes under `docker/base/` → rebuild base image
- Changes under `docker/project/` → rebuild project image

Published images (GHCR):

- `ghcr.io/joaocb2002/object-nav-habitat/habitat-base`
- `ghcr.io/joaocb2002/object-nav-habitat/habitat-project`

Tags:

- `:main` — latest from main branch
- `:sha-<commit>` — immutable builds

---

## Canonical run commands

Interactive dev container:

```bash
./scripts/run_dev.sh bash
```

Smoke test / import sanity:

```bash
./scripts/run_dev.sh python scripts/sanity_check.py
```

Non-interactive long runs:

```bash
./scripts/run_train.sh python <script>.py
```

---

## Notes on datasets and outputs

- Datasets are expected under `${DATA_DIR:-$PWD/datasets}` on the host and mounted into the container.
- Outputs should go to `${OUTPUT_DIR:-$PWD/outputs}` (mounted to `/outputs`).
- Avoid committing large artifacts under `outputs/` or datasets.
