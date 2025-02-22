# First stage: Use an Ubuntu-based image with Python 3.10.15
FROM python:3.10.15-slim AS python-base

# Set working directory
WORKDIR /app

# Install system dependencies (including zlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir biopython matplotlib numpy

# Second stage: Use a Debian-based Clustal Omega image instead of BioContainers
FROM ubuntu:22.04 AS clustalo-base

# Set working directory
WORKDIR /app

# Install Clustal Omega and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    clustalo zlib1g zlib1g-dev python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python from the first stage
COPY --from=python-base /usr/local /usr/local

# Copy script into the container
COPY analysis_script.py clustalo /app/

# Set the default command
CMD ["python3", "analysis_script.py"]
