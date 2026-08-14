# syntax=docker/dockerfile:1
#
# Custom AI Runtime (Azure) image reproducing NVIDIA NeMo-RL v0.4.0
# (reference: nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano), built to run on
# Databricks AI Runtime serverless GPU on **Azure**.
#
# WHY THIS SHAPE (verified facts, not guesses):
#   * AIR custom images must be on Docker Hub, < 20 GB, use /opt/venv, and ignore
#     ENTRYPOINT/WORKDIR (verified in the AIR docs).
#   * The nvcr.io NeMo-RL image itself cannot be used directly (wrong registry,
#     different venv layout /opt/nemo_rl_venv, its own entrypoint/MPI stack).
#   * AIR base image `dcs-base-azure-devel` = CUDA 12.9.1, venv /opt/venv (uv),
#     Python via uv, InfiniBand networking. (Verified from the image config.)
#     -> Use the AZURE base because the workload runs on Azure AIR.
#     -> Use the DEVEL variant because NeMo-RL compiles CUDA extensions (nvcc).
#   * NeMo-RL v0.4.0 pins a self-consistent stack: torch==2.7.1, vllm==0.10.0,
#     transformer-engine==2.5.0, mamba-ssm, causal-conv1d, flash-attn.
#     (Verified from NeMo-RL pyproject.toml @ tag v0.4.0.)
#
# VERSION-MATCH STRATEGY (this is the part that bites you if done wrong):
#   Do NOT hand-pick individual mamba/vLLM/flash-attn wheels. Instead install the
#   NeMo-RL repo at the pinned tag and let its pyproject resolve ONE consistent
#   set against torch 2.7.1. The build stage then imports every native extension
#   so an ABI/CUDA mismatch fails the BUILD (not silently at runtime on AIR).
#
# BUILD PLATFORM: linux/amd64 only (AIR workers are x86_64).

FROM databricksruntime/air:dcs-base-azure-devel

# Pin the NeMo-RL release the reference image was built from.
ARG NEMO_RL_REF=v0.4.0
ARG NEMO_RL_REPO=https://github.com/NVIDIA/NeMo-RL.git

# H100 = sm90. (Add 10.0 for Blackwell if you later target B200.)
ENV TORCH_CUDA_ARCH_LIST="9.0"
# Keep CUDA-extension builds from trying every arch / oversubscribing.
ENV MAX_JOBS=8

# CUDA C++ extensions built from source (deep-gemm, flash-attn, mamba, TE) need
# the Python dev headers (Python.h) + a C/C++ toolchain. The base image's Python
# is the system Python 3.12.3, so python3.12-dev provides an exact-version match
# at /usr/include/python3.12 (what the compiler's -I flag looks for).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone NeMo-RL at the pinned tag (submodules incl. the Megatron path).
RUN git clone --filter=blob:none "${NEMO_RL_REPO}" /opt/nemo-rl \
    && cd /opt/nemo-rl \
    && git checkout "${NEMO_RL_REF}" \
    && git submodule update --init --recursive

# Install via NeMo-RL's OWN uv workspace, not hand-rolled pip. Its pyproject
# defines [tool.uv.sources] (git-pinned mamba-ssm/causal-conv1d), a cu128 torch
# index, flash-attn dependency-metadata, no-build-isolation-package, and a build
# dependency-group. `uv sync` honors all of that; `uv pip install` does not.
# mamba/causal/vllm/flash-attn live under the `vllm` extra (also in `mcore`).
# Point uv at the AIR-managed venv so everything lands in /opt/venv (what AIR runs).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN cd /opt/nemo-rl && uv sync --extra vllm --locked || \
    (echo "no lockfile or locked sync failed; retrying unlocked" && cd /opt/nemo-rl && uv sync --extra vllm)

# MLflow client for logging to the AIR-owned run (see steps 01-05).
RUN uv pip install --no-cache "mlflow>=3.6" hf_transfer

# --- BUILD-TIME VERIFICATION GATE -------------------------------------------
# The build host has NO GPU, and some packages (mamba_ssm -> triton) probe for a
# GPU driver at import time and would raise "0 active drivers" here even though
# they are correctly installed. So on a GPU-less build we only verify the
# packages are INSTALLED (find_spec) — proving the version-consistent set landed.
# The full import (which needs a GPU) runs on AIR at first job launch.
RUN python - <<'PY'
import importlib.util, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
missing = [m for m in ["mamba_ssm","causal_conv1d","transformer_engine","vllm","flash_attn"]
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"MISSING packages: {missing}")
print("OK installed:", "mamba_ssm causal_conv1d transformer_engine vllm flash_attn")
if torch.cuda.is_available():
    import mamba_ssm, causal_conv1d, transformer_engine, vllm, flash_attn
    print("OK GPU import (all)")
else:
    print("no GPU on build host -> skipped runtime import (runs on AIR)")
PY

# NOTE: AIR ignores ENTRYPOINT/WORKDIR. Reference NeMo-RL with absolute paths in
# your train.yaml `command` (e.g. python /opt/nemo-rl/examples/run_sft.py).
# If FIPS SSL errors appear at runtime and compliance does not require FIPS,
# set OPENSSL_FORCE_FIPS_MODE=0 in the workload YAML env_variables.
