# ObjectNav Habitat 🧭

A reproducible Docker-based research environment for running Habitat-Sim / Habitat-Lab experiments locally or on remote GPU servers 🚀. The project is designed to support structured experimentation at the intersection of deep reinforcement learning and probabilistic (Bayesian) inference.

The repository separates **infrastructure concerns** (containers, dependencies, execution scripts) from **research logic** (models, training loops, configurations), enabling repeatable experiments and easier extension over time.

---

## Scope and Goals

* Provide a deterministic, versioned runtime for Habitat-based experiments
* Support both interactive development and long-running training jobs
* Enable future expansion of research components (algorithms, models, evaluation)
* Minimize environment-related variability across machines and servers

This README documents the *structure and usage* of the stack. Scientific details of methods, models, and experiments are expected to live in `src/` and `configs/` and evolve independently.

---

## Repository Structure

```
.
├── docker/
│   ├── base/               # Stable, heavy dependencies
│   │   ├── Dockerfile
│   │   └── environment.yml
│   └── project/            # Project-specific Python dependencies
│       ├── Dockerfile
│       └── requirements.txt
├── .github/workflows/       # CI: build & push Docker images
├── scripts/                 # Container and execution helpers
├── src/                     # Research code (models, training, evaluation)
├── configs/                 # Experiment configuration files
└── outputs/                 # Experiment outputs (not versioned)
```

**Build logic**

* Changes in `docker/base/` trigger a rebuild of the base image
* Changes in `docker/project/` trigger a rebuild of the project image

---

## Docker Image Architecture

The stack uses a two-layer image design:

* **habitat-base**

  * Habitat-Sim
  * Habitat-Lab
  * System libraries and CUDA bindings
  * Conda / Python runtime

* **habitat-project**

  * Research-specific Python dependencies
  * Computer vision and ML libraries
  * Lightweight and frequently changing

This separation reduces rebuild time and improves reproducibility.

---

## Published Images (GHCR)

* Base image:
  `ghcr.io/joaocb2002/object-nav-habitat/habitat-base`

* Project image:
  `ghcr.io/joaocb2002/object-nav-habitat/habitat-project`

### Tags

* `:main` – latest build from the main branch
* `:sha-<commit>` – immutable, fully reproducible builds

---

## Host Requirements

* Linux (Ubuntu 22.04 recommended)
* NVIDIA GPU
* NVIDIA driver (560.x or newer recommended)
* Docker Engine
* NVIDIA Container Toolkit (required for GPU access)

---

## Data and Outputs

* Datasets are **not** included in Docker images
* Expected default dataset location: `~/datasets`
* Outputs are written to `./outputs` (created automatically)

To use a custom dataset location:

```bash
export DATA_DIR=/path/to/datasets
```

---

## Basic Usage

### Clone the Repository

```bash
git clone https://github.com/joaocb2002/object-nav-habitat.git
cd object-nav-habitat
```

### Pull the Project Image

```bash
docker pull ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main
```

### Sanity Check

```bash
./scripts/bootstrap.sh
./scripts/run_dev.sh python scripts/sanity_check.py
```

---

## Development Workflow

Use `run_dev.sh` for interactive development and debugging.

```bash
./scripts/run_dev.sh bash
./scripts/run_dev.sh python script.py
```

Mounted paths:

* Repository → `/workspace`
* Datasets → `/data` (read-only)
* Outputs → `/outputs`

This mode is suitable for rapid iteration and IDE attachment (e.g. VS Code).

---

## Training Workflow

Use `run_train.sh` for long-running, non-interactive jobs.

```bash
./scripts/run_train.sh python train.py
```

Characteristics:

* Non-interactive execution
* Uses `--ipc=host` for improved multiprocessing performance
* Intended for remote servers or unattended runs

---

## Managing Dependencies

Project-level Python dependencies are defined in:

```
docker/project/requirements.txt
```

Lockfiles (required for reproducibility):

* docker/base/conda-lock.yml — exact conda environment for the base image
* docker/project/requirements.lock — exact pip environment for the project image

**Update workflow**

* If you change docker/base/environment.yml, regenerate docker/base/conda-lock.yml and rebuild the base image.
* If you change docker/project/requirements.txt, regenerate docker/project/requirements.lock and rebuild the project image.

Lockfile generation commands:

```bash
# Note: conda-lock must be the same version as in base dockerfile
conda-lock -f docker/base/environment.yml -p linux-64 --lockfile docker/base/conda-lock.yml
```

```bash
# Note: python version must be the same as in environment.yml (3.9)
python -m piptools compile --generate-hashes --allow-unsafe -o requirements.lock requirements.txt
```

**Build behavior**

* Base image installs from docker/base/conda-lock.yml.
* Project image installs from docker/project/requirements.lock with hashes enforced.

After modifying dependencies:

```bash
git commit -am "Update project dependencies"
git push
```

CI will rebuild and publish a new project image. Update locally with:

```bash
docker pull ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main
```

---

## Notes on Docker and GPUs (Linux)

Reliable GPU support requires Docker Engine with the NVIDIA Container Toolkit.
Docker Desktop for Linux runs inside a VM and may not correctly expose CUDA,
EGL, or OpenGL, which can break Habitat-Sim or PyTorch GPU execution.

---

## Troubleshooting

* **GPU not visible in container**: verify NVIDIA Container Toolkit installation
* **`import habitat_sim` fails**: check driver compatibility and base image
* **Unexpected behavior after dependency changes**: pull the latest image or rebuild locally

---

## Extending the Project

* Add new algorithms, models, and training logic under `src/`
* Define experiments and hyperparameters in `configs/`
* Document scientific methodology and results alongside code

The infrastructure is intended to remain stable while research components evolve.

---

## Useful command

For inspecting a digital scene visually, run the following command in a habitat conda env:
```bash
habitat-viewer --dataset /path/to/<scene_dataset_config>.json <scene_name>
# example: from repo root
# habitat-viewer --dataset datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json FloorPlan1_physics
```

