# 1) Start from your already-built base image
FROM ghcr.io/joaocb2002/object-nav-habitat/habitat-base:main

# 2) Copy project requirements
COPY requirements.txt /tmp/requirements.txt

# 3) Install project-level Python dependencies 
# Install CUDA-enabled torch wheels explicitly
# Note: torch and torchvision versions are from 2025 and match to CUDA 12.6)
RUN pip install --no-cache-dir \
    torch==2.9.0 torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install the rest from normal PyPI
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4) Set working directory
WORKDIR /workspace
