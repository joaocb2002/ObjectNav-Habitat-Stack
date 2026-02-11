# 1) Start from your already-built base image (see ../habitat-base/Dockerfile)
ARG BASE_IMAGE=ghcr.io/joaocb2002/object-nav-habitat/habitat-base:main
FROM ${BASE_IMAGE}

# 2) Copy project requirements
COPY requirements.lock /tmp/requirements.lock

# 3) Install project-level Python dependencies
# Keep pip version aligned with the base image for reproducibility
RUN python -m pip install --no-cache-dir --upgrade pip==23.3.2

# Install from the lockfile with hashes for reproducibility
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

# 4) Set working directory
WORKDIR /workspace 
