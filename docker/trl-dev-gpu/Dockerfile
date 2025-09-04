# Builds GPU docker image of PyTorch with TRL dev dependencies
# Uses multi-staged approach to reduce size
#
# Build Instructions:
# 1. Navigate to the directory containing this Dockerfile:
#    cd docker/trl-dev-gpu/
#
# 2. Build the Docker image:
#    docker build -t trl-dev-gpu:latest .
#
# 3. Run the container with GPU support:
#    docker run --gpus all -it trl-dev-gpu:latest
#
# 4. Alternative: Run with volume mounting for development:
#    docker run --gpus all -v $(pwd):/workspace -it trl-dev-gpu:latest
#
# 5. Build with specific tag:
#    docker build -t trl-dev-gpu:v1.0 .
#
# Note: This image includes all TRL development dependencies (trl[dev])
# and uses Tsinghua PyPI mirror for faster package downloads in China.
#
# Image Features:
# - CUDA 12.2.2 support for GPU acceleration
# - Python 3.10 with conda environment
# - All TRL development tools (pytest, pre-commit, etc.)
# - Audio processing libraries (librosa, soundfile, ffmpeg)
# - Quantization support (bitsandbytes, auto-gptq)
# - DeepSpeed support for large model training
# - Vision-Language Model support
#
# Use Cases:
# - TRL development and testing
# - Running TRL examples and tutorials
# - Model fine-tuning with reinforcement learning
# - Research and experimentation with transformer models
# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.10
# Install apt libs - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN apt-get update && \
    apt-get install -y curl git wget software-properties-common git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Install audio-related libraries 
RUN apt-get update && \
    apt install -y ffmpeg

RUN apt install -y libsndfile1-dev
RUN git lfs install

# Create our conda env - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN conda create --name trl python=${PYTHON_VERSION} ipython jupyter pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Configure pip to use Tsinghua PyPI mirror
RUN python3 -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/trl/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Stage 2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Configure pip to use Tsinghua PyPI mirror in the build stage
RUN python3 -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN source activate trl && \ 
    python3 -m pip install --no-cache-dir bitsandbytes optimum auto-gptq

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Activate the conda env and install transformers + accelerate from source
RUN source activate trl && \
    python3 -m pip install -U --no-cache-dir \
    librosa \
    "soundfile>=0.12.1" \
    scipy \
    transformers \
    accelerate \
    peft \
    trl[dev]@git+https://github.com/huggingface/trl

RUN source activate trl && \ 
    pip freeze | grep trl

RUN echo "source activate trl" >> ~/.profile

# Activate the virtualenv
CMD ["/bin/bash"]
