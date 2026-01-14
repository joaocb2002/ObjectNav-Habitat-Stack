# 1) Start from your already-built base image
FROM ghcr.io/joaocb2002/object-nav-habitat/habitat-base:main

# 2) Copy project requirements
COPY requirements.txt /tmp/requirements.txt

# 3) Install project-level Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4) Set working directory
WORKDIR /workspace
