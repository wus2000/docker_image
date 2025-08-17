# ====================================================================================
# Final Dockerfile for vLLM - Legacy & Platform Compliant
#
# Target Platform: Linux (amd64/x86_64)
# Docker Compatibility: v17.03 or older
#
# This Dockerfile integrates specific platform requirements:
# - Sets LANG and Timezone (Asia/Shanghai)
# - Installs `s3cmd` and `zip` utilities
# - Adheres to a single-stage build process for legacy Docker versions
# ====================================================================================

# We use the 'devel' image as it contains all build tools. The final image
# will be larger but ensures build consistency in a single stage.
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# --- Arguments and Environment Variables ---
# All ARGs must be declared at the top before their first use.

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV ALL_PROXY=${ALL_PROXY}

ENV MAX_JOBS=8
ENV NVCC_THREADS=1

ARG PYTHON_VERSION=3.12
ARG TARGETPLATFORM
ARG GIT_REPO_CHECK=0
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'

# [PLATFORM REQUIREMENT] Set Language and Timezone
ENV LANG=en_US.UTF-8
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# Other environment variables required by vLLM build
ENV UV_HTTP_TIMEOUT=500
ENV MAX_JOBS=${max_jobs}
ENV NVCC_THREADS=$nvcc_threads
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}
ENV VLLM_USAGE_SOURCE production-docker-image
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# --- System & Python Dependencies Installation ---
# This single RUN command handles all system-level setup.

RUN \
    # [PLATFORM REQUIREMENT] Apply timezone configuration
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    \
    # 1. Initial update and installation of prerequisites for adding repositories
    apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    \
    # 2. Add the PPA for newer Python versions
    add-apt-repository ppa:deadsnakes/ppa && \
    \
    # 3. Update package lists AGAIN to fetch package info from the new PPA
    apt-get update -y && \
    \
    # 4. Now, install all required packages in a single command
    apt-get install -y --no-install-recommends \
        # Base dependencies
        ccache git curl sudo wget vim \
        # Python from PPA
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
        # Libraries for vLLM & other packages
        libsm6 libxext6 libgl1 libibverbs-dev ffmpeg \
        # [PLATFORM REQUIREMENT] Install s3cmd and zip
        s3cmd zip \
        # GCC 10 upgrade
        gcc-10 g++-10 && \
    \
    # 5. Set Python alternatives
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
    \
    # 6. Install pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    \
    # 7. Set GCC alternatives
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10 && \
    \
    # 8. Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Verify versions
RUN python3 --version && python3 -m pip --version && gcc --version

# Install uv (modern Python package installer) for faster dependency resolution
RUN python3 -m pip install uv

# Triton/PyTorch compatibility workaround
RUN ldconfig /usr/local/cuda-12.4/compat/

WORKDIR /workspace

# --- vLLM Build Process ---
# Copy all necessary files for the build
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
COPY requirements/build.txt requirements/build.txt

# Install build-time and common Python dependencies
RUN uv pip install --system -r requirements/cuda.txt
RUN uv pip install --system -r requirements/build.txt

# Copy the entire source code into the image to build it
COPY . .

# (Optional) Run repo check if enabled during build
RUN if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

# Build the vLLM wheel. Caching mechanisms like sccache are removed for compatibility.
RUN rm -rf .deps && \
    export MAX_JOBS=64 && \
    export NVCC_THREADS=32 && \
    mkdir -p .deps && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

# --- Install vLLM and Final Runtime Dependencies ---
# Install the wheel we just built from the 'dist' directory
RUN uv pip install --system dist/*.whl --verbose

# Install FlashInfer (assuming x86_64, as arm64 logic was conditional)
RUN if [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        uv pip install --system https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post2/flashinfer_python-0.2.1.post2+cu124torch2.6-cp38-abi3-linux_x86_64.whl ; \
    fi

# Install final runtime dependencies for the OpenAI server endpoint
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        uv pip install --system accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.42.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    else \
        uv pip install --system accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.3' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    fi

