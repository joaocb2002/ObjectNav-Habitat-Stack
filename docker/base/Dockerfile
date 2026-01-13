# 1) Start from a clean Ubuntu base
FROM ubuntu:22.04

# 2) Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# 3) Install system-level dependencies needed for:
#    - downloading things
#    - building native extensions
#    - OpenGL / EGL headless rendering
RUN apt-get update && apt-get install -y \
    wget curl git ca-certificates build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libopengl0 libglvnd0 libgl1 libegl1 libglx0 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# 4) Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# 5) Make conda available in PATH and accept Anaconda TOS
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 6) Copy the environment specification into the image
COPY environment.yml /tmp/environment.yml

# 7) Create the conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# 8) Activate the environment by default
ENV CONDA_DEFAULT_ENV=habitat
ENV PATH=$CONDA_DIR/envs/habitat/bin:$PATH

# 9) Basic sanity check (import habitat-sim)
RUN python - <<EOF
import habitat_sim
print("Habitat-Sim imported successfully")
EOF

# 10) Set working directory
WORKDIR /workspace
