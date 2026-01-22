# 1) Start from your already-built base image
FROM ghcr.io/joaocb2002/object-nav-habitat/habitat-base:main

# 2) Copy project requirements
COPY requirements.txt /tmp/requirements.txt

# 3) Install project-level Python dependencies 
# Install CUDA-enabled torch wheels explicitly
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install the rest from normal PyPI
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4) Set working directory
WORKDIR /workspace
