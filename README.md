# 🧭 ObjectNav Habitat

A reproducible **Docker-based stack** for running **Habitat-Sim / Habitat-Lab** experiments locally or on remote GPU servers.

---

## ✨ What You Get

- 🔁 **Fully reproducible environments** via Docker images hosted on GHCR
- 🧱 **Two-layer image architecture**
  - **`habitat-base`**: heavy, stable dependencies  
    *(Habitat-Sim, Habitat-Lab, Miniconda)*
  - **`habitat-project`**: project-specific dependencies  
    *(YOLO / Ultralytics, CV & ML libraries, etc.)*
- 🚀 **Simple commands to launch containers**
  - `scripts/run_dev.sh` — interactive development (local machine-oriented)
  - `scripts/run_train.sh` — long training runs (server-oriented)

---

## 📁 Repository Layout

```
.
├── docker/
│   ├── base/
│   │   ├── Dockerfile
│   │   └── environment.yml
│   └── project/
│       ├── Dockerfile
│       └── requirements.txt
├── .github/workflows/
│   └── GitHub Actions (build & push images to GHCR)
├── scripts/
│   └── Container helper scripts
├── src/
    └── Source code for experiments
├── configs/
    └── Experiment configurations
└── outputs/
    └── Experiment outputs (not pushed)
```

**Notes**
- Changes in `docker/base/` trigger a rebuild of **habitat-base**
- Changes in `docker/project/` trigger a rebuild of **habitat-project**

---

## 🐳 Docker Images (GHCR)

- **Base image**  
  `ghcr.io/joaocb2002/object-nav-habitat-stack/habitat-base`
- **Project image**  
  `ghcr.io/joaocb2002/object-nav-habitat-stack/habitat-project`

### 🏷️ Tags
- `:main` — latest build from the `main` branch
- `:sha-<commit>` — immutable, fully reproducible builds

---

## 🖥️ Host Prerequisites

- Ubuntu 22.04 *(recommended)*
- NVIDIA GPU driver installed
- Docker Engine
- NVIDIA Container Toolkit *(required for `--gpus all`)*

---

## ⚡ Quickstart (New Machine)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/joaocb2002/object-nav-habitat.git
cd object-nav-habitat
```

### 2️⃣ Pull the project image
```bash
docker pull ghcr.io/joaocb2002/object-nav-habitat/habitat-project:main
```

### 3️⃣ Prepare datasets & outputs

**Expected defaults**
- Datasets: `~/datasets`
- Outputs: `./outputs` *(auto-created)*

If datasets live elsewhere:
```bash
export DATA_DIR=/path/to/datasets
```

### 4️⃣ Bootstrap and Sanity check
```bash
./scripts/bootstrap.sh # Check if infrastructure is set
./scripts/run_dev.sh python scripts/sanity_check.py # Check if containers can import habitat
```

### 5️⃣ Enter a container with interactive shell
```bash
./scripts/run_train.sh bash
```

### 6️⃣ Start developing (VS Code attachment)
- Folder: `workspace/src`

---

## 🛠️ Development Workflow

Use **`run_dev.sh`** for debugging and fast iteration:

```bash
./scripts/run_dev.sh bash                  # Enter container with interactive shell (most common)
./scripts/run_dev.sh python script.py      # Run a specific script
```

**Mount-binded points**
- Repo → `/workspace`
- Datasets → `/data` *(read-only)*
- Outputs → `/outputs` *(mapped to host `./outputs`)*

---

## 🧪 Training Workflow

Use **`run_train.sh`** for long-running jobs on servers:

```bash
./scripts/run_train.sh python train.py
```

**Differences**
- Non-interactive
- Uses `--ipc=host` for better multiprocessing performance

---

## 🐧 Docker Desktop (Linux) — Important Note

On Linux, reliable GPU support requires:
- **Docker Engine**
- **NVIDIA Container Toolkit**

⚠️ Docker Desktop for Linux runs inside a VM and may fail to expose:
- CUDA
- EGL / OpenGL

This can break **Habitat-Sim** or **PyTorch GPU** execution.

---

## ➕ Adding Python Dependencies

Add project-level dependencies to:
```
docker/project/requirements.txt
```

Then:
```bash
git add docker/project/requirements.txt
git commit -m "Add dependency: <name>"
git push
```

GitHub Actions will rebuild and push a new **habitat-project** image.

Update on any machine:
```bash
docker pull ghcr.io/joaocb2002/object-nav-habitat-stack/habitat-project:main
```

---

## 📦 Datasets

- Datasets are **not included** in Docker images
- Must be downloaded/stored on each machine
- Mounted via `DATA_DIR`


---

## 🧯 Troubleshooting

- ❌ **Docker can’t see GPU**  
  → Install & configure NVIDIA Container Toolkit

- ❌ **`import habitat_sim` fails**  
  → Check host NVIDIA driver and container libraries  
  → Run base image sanity check

- 🔄 **Changed Dockerfiles or env specs**  
  → Pull latest image or rebuild locally

---

✅ Happy experimenting!
