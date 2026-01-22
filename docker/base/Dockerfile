# Base image with CUDA + OpenGL/EGL (GLVND) support for headless rendering on NVIDIA GPUs
# Works on any Linux host with NVIDIA drivers + NVIDIA Container Toolkit.
FROM nvidia/cudagl:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# System-level dependencies
# - downloading/build tooling
# - common runtime libs
# - GLVND/OpenGL/EGL user-space loader libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libopengl0 libglvnd0 libgl1 libegl1 libglx0 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Ensure GLVND knows about NVIDIA EGL.
# The actual lib (libEGL_nvidia.so.0) is provided by the host driver and mounted by nvidia-container-toolkit.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{\n  "file_format_version": "1.0.0",\n  "ICD": { "library_path": "libEGL_nvidia.so.0" }\n}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Make GLVND pick NVIDIA by default (prevents Mesa/conda from hijacking EGL)
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Install Miniconda for package management
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Make conda available in PATH and accept Anaconda TOS
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy the environment specification into the image
COPY environment.yml /tmp/environment.yml

# Create the conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate the environment by default
ENV CONDA_DEFAULT_ENV=habitat
ENV PATH=$CONDA_DIR/envs/habitat/bin:$PATH

# Basic sanity check (import habitat-sim)
RUN python - <<'EOF'
import habitat_sim
print("Habitat-Sim imported successfully")
EOF

# Set working directory
WORKDIR /workspace
