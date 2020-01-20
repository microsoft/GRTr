# syntax = docker/dockerfile:experimental
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /miniconda \
 && rm ~/miniconda.sh
ENV PATH=/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install conda-build \
&& /miniconda/bin/conda create -y --name py37 python=3.7.5 \
&& /miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.0 environment.yml
COPY environment.yml /tmp/
RUN --mount=type=cache,target=/root/.cache/pip conda env update --name py37 --file /tmp/environment.yml && conda clean -afy
RUN rm -f /tmp/environment.yml

# Install apex
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# RUN cd apex && pip install -v --no-cache-dir .

RUN --mount=type=cache,target=/root/.cache/pip conda install pillow=6.1
