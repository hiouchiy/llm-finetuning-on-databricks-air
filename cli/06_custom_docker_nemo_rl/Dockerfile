# syntax=docker/dockerfile:1
#
# Custom AI Runtime image that reproduces an NVIDIA NeMo-RL environment
# (equivalent to nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano) on top of the
# Databricks AI Runtime base image.
#
# WHY NOT USE nvcr.io/nvidia/nemo-rl DIRECTLY?
#   - AI Runtime custom images must live on **Docker Hub** (nvcr.io / ECR / GCR
#     / GHCR are NOT supported), and must be < 20 GB.
#   - The NeMo-RL image has its own venv layout (/opt/nemo_rl_venv), its own
#     entrypoint (/opt/nvidia/nvidia_entrypoint.sh) and its own MPI/HPCX stack,
#     which do not match what AI Runtime expects (Python at /opt/venv managed by
#     uv; AIR injects torchrun rendezvous env vars and runs your `command`).
#   So instead we START from the AIR base and re-install the NeMo-RL stack into
#   the AIR-managed venv.
#
# BASE IMAGE CHOICE
#   Use the **devel** variant: NeMo-RL pulls in CUDA extensions (Transformer
#   Engine, optionally custom vLLM/Megatron) that compile with nvcc.
#   The runtime variant has no nvcc. The AIR base already ships CUDA 12.x, NCCL
#   and AWS EFA — matching the NeMo-RL image's CUDA 12.9 / NCCL 2.26 generation.
#
# BUILD (must be linux/amd64 — do NOT build the final image on arm64 Macs):
#   docker buildx build --platform linux/amd64 \
#     -t <your-dockerhub-user>/air-nemo-rl:v1 --push .
#
# Keep the result < 20 GB (use --no-cache installs, avoid dev toolchains in the
# final layer where possible; a multi-stage build can help if you approach 20GB).

FROM databricksruntime/air:dcs-base-aws-devel

# Pin the NeMo-RL revision that v0.4.0.nemotron_3_nano was built from, so the
# reproduced environment matches the reference image.
ARG NEMO_RL_COMMIT=8bdfd48ed1e34d01b8e7a2b635b367c654c564fd
ARG NEMO_RL_REPO=https://github.com/NVIDIA/NeMo-RL.git

# TE / custom kernels build for Hopper (H100 = sm90). Add 10.0 for Blackwell if needed.
ENV TORCH_CUDA_ARCH_LIST="9.0"

# The AIR base manages a venv at /opt/venv via uv; install into THAT venv (do
# not create /opt/nemo_rl_venv — AIR runs your command against /opt/venv).
# `uv pip install` targets the active AIR venv.
RUN git clone --filter=blob:none "${NEMO_RL_REPO}" /opt/nemo-rl \
    && cd /opt/nemo-rl \
    && git checkout "${NEMO_RL_COMMIT}" \
    && git submodule update --init --recursive

# Install NeMo-RL and its dependencies into the AIR venv. --no-cache keeps the
# image small. NeMo-RL declares its own pyproject; installing it (editable)
# pulls transformers / trl / megatron-core / transformer-engine, etc.
RUN uv pip install --no-cache -e /opt/nemo-rl

# Nemotron-Nano-v2 is a Mamba-Transformer hybrid → mamba_ssm + causal_conv1d.
# Use prebuilt wheels matching the AIR base torch/cuda/python (torch2.9/cu12/cp312).
# (Same wheels used by steps 01–05; pin the versions your AIR base ships.)
RUN uv pip install --no-cache \
      "causal_conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
      "mamba_ssm @ https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# MLflow client used by the training scripts to log to the AIR-owned run.
RUN uv pip install --no-cache "mlflow>=3.6" hf_transfer

# NOTE: WORKDIR is IGNORED by AI Runtime at run time. Reference baked-in files
# with absolute paths in your train.yaml `command` (e.g. `python /opt/nemo-rl/...`).
# Do NOT set a custom ENTRYPOINT expecting AIR to honor it — AIR runs your
# `command` from the YAML.
