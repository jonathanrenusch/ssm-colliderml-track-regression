# Inference-profiling image for the short-sequence SSM track-fitter.
# Target: RTX 5000 Ada (sm_89) workstation GPUs, but portable (the Triton kernel
# JIT-autotunes for whatever GPU is present). Lean: no flash-attn / colliderml /
# datasets (not needed for SSM-CLS inference) — only the CUDA extensions the stock
# Mamba2 module needs, which the encoder is then swapped off of (v5pc).
#
# Build (from the repo ROOT so the build context sees src/ and docker/):
#   docker build -f docker/rtx_infer/Dockerfile -t ssm-rtx-infer .
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip git build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# torch first — the CUDA extensions below build against it (--no-build-isolation)
RUN pip install --upgrade pip packaging wheel setuptools
RUN pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128

# pure-python runtime deps (triton ships with torch)
RUN pip install "numpy>=2.1,<3" "pandas>=2.3,<4" "lightning==2.5.2" \
                "lion-pytorch>=0.2.3,<0.3" "h5py>=3.14,<4" "einops>=0.8.1,<0.9" pyyaml

# CUDA extensions, compiled for Ada (8.9) + Hopper (9.0) so the image also runs
# on an H100 for cross-checking. Built from source (no prebuilt wheels).
ENV TORCH_CUDA_ARCH_LIST="8.9 9.0"
RUN pip install --no-build-isolation "causal-conv1d==1.6.0" "mamba-ssm==2.3.0"

# --- OPTIONAL: Nsight Systems CLI in-image, so `--entrypoint nsys` works.
#     Uncomment to bake it in (adds ~1 GB). Otherwise profile from the host.
# RUN apt-get update && apt-get install -y --no-install-recommends nsight-systems-cli \
#     && rm -rf /var/lib/apt/lists/*

# repo source: only the package + these docker tools (see .dockerignore)
WORKDIR /workspace
COPY src /workspace/src
COPY docker/rtx_infer /workspace/docker/rtx_infer

ENV PYTHONPATH=/workspace/src \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    TRK_MATMUL_PRECISION=highest

ENTRYPOINT ["/workspace/docker/rtx_infer/entrypoint.sh"]
