# Inference-profiling image for the short-sequence SSM track-fitter.
# Target: RTX 5000 Ada (sm_89) workstations; portable (Triton JIT-autotunes).
# Lean: no flash-attn / colliderml / datasets (not needed for SSM-CLS inference).
#
# Versions are PINNED to the exact set validated on the CERN pod (python 3.12.0,
# torch 2.9.1+cu128, mamba-ssm 2.3.0, causal-conv1d 1.6.0, ...), so the built
# environment matches what produced the measured numbers.
#
# Build (from the repo ROOT so the context sees src/ and docker/):
#   docker build -f docker/rtx_infer/Dockerfile -t ssm-rtx-infer .
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build ca-certificates curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Exact python 3.12 via micromamba (matches the validated environment)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
RUN micromamba create -y -p /opt/env -c conda-forge python=3.12.0 pip && micromamba clean -a -y
ENV PATH=/opt/env/bin:$PATH

RUN pip install --upgrade pip packaging wheel setuptools
# torch first: the CUDA extensions below build against it (--no-build-isolation)
RUN pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
# pinned pure-python runtime deps (triton ships inside torch)
RUN pip install "numpy==2.4.4" "pandas==3.0.2" "lightning==2.5.2" \
                "lion-pytorch==0.2.4" "h5py==3.16.0" "einops==0.8.2" "pyyaml"
# CUDA extensions, compiled from source for Ada (8.9) + Hopper (9.0)
ENV TORCH_CUDA_ARCH_LIST="8.9 9.0"
RUN pip install --no-build-isolation "causal-conv1d==1.6.0" "mamba-ssm==2.3.0"

# --- OPTIONAL: Nsight Systems CLI in-image so `--entrypoint nsys` works (adds ~1 GB).
# RUN apt-get update && apt-get install -y --no-install-recommends nsight-systems-cli \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY src /workspace/src
COPY docker/rtx_infer /workspace/docker/rtx_infer

ENV PYTHONPATH=/workspace/src \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    TRK_MATMUL_PRECISION=highest

ENTRYPOINT ["/workspace/docker/rtx_infer/entrypoint.sh"]
