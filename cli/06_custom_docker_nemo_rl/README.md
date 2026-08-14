# 06 — Custom Docker image (NeMo-RL) on AI Runtime (Azure)

Reproduce an **NVIDIA NeMo-RL v0.4.0** environment (reference:
`nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano`) as a **Docker Hub** image that
**Azure** AI Runtime can run.

**→ For the step-by-step, copy-paste build procedure, see [RUNBOOK.md](RUNBOOK.md).**
This README is the background/rationale.

## Why you can't use the `nvcr.io` image directly

AI Runtime custom images have hard constraints:

| Constraint | NeMo-RL `nvcr.io` image | Consequence |
|---|---|---|
| Registry must be **Docker Hub** (ACR/ECR/GCR/GHCR not supported) | hosted on `nvcr.io` | must be rebuilt and pushed to Docker Hub |
| Size **< 20 GB** | large (full CUDA+TE+vLLM+Megatron) | may need trimming |
| Python venv at **`/opt/venv`** (uv) | uses `/opt/nemo_rl_venv` | layout mismatch |
| AIR runs your `command`; **ENTRYPOINT/WORKDIR ignored** | has its own entrypoint/WORKDIR | won't be honored |
| **FIPS-enabled** hosts | not FIPS-aware | may need `OPENSSL_FORCE_FIPS_MODE=0` |

The reference image is CUDA 12.9 / Python 3.12. The AIR **Azure** base
(`dcs-base-azure-devel`, verified from its image config) is **CUDA 12.9.1, venv
`/opt/venv` (uv), InfiniBand** — same CUDA generation — so rebuilding on the AIR
base is feasible; it's the packaging / registry / layout that differ.

## Where to build (verified the hard way)

- **Databricks clusters (GPU or CPU) CANNOT build images.** Tested on a live
  cluster: no docker/nvcc; a hand-installed BuildKit *starts* but the build
  fails with `bind mount ... permission denied` — the cluster sandbox blocks the
  mounts image builds require. This is by design (AIR/clusters run your code in
  an unprivileged sandbox). So **build outside Databricks.**
- **Build needs no GPU.** nvcc compiles for sm90 (H100) without a physical GPU.
  Build on a cheap CPU VM; do the GPU smoke-test on AIR (1×H100) at the end.
- Chosen path: **build on an Azure CPU VM, push to Docker Hub, run on Azure AIR.**
  See [RUNBOOK.md](RUNBOOK.md).

## Version matching (the part that bites)

NeMo-RL **v0.4.0** pins a self-consistent stack: **torch==2.7.1**, vllm==0.10.0,
transformer-engine==2.5.0, mamba-ssm, causal-conv1d, flash-attn (from the tag's
`pyproject.toml`). The AIR base ships CUDA 12.9 but **no torch**, so:

- The Dockerfile installs **torch 2.7.1 first**, then `uv pip install -e` the
  NeMo-RL repo and lets its pyproject resolve everything else against that torch.
- **Do not hand-pick individual mamba/vLLM/flash-attn wheels** — that is exactly
  how versions drift (an earlier draft pinned torch2.9 wheels, which would have
  clashed with NeMo-RL's torch 2.7.1).
- The Dockerfile ends with an **import gate** (`import mamba_ssm, transformer_engine,
  vllm, flash_attn`) so any ABI/CUDA mismatch **fails the build**, not silently
  at runtime on AIR.

## ⚠️ Status: designed + version-reconciled, NOT yet built

The Dockerfile and RUNBOOK are complete and internally consistent, but the image
has **not been built or run yet** (no x86+docker environment available at authoring
time). Confirm during the first real build:

- [ ] The `uv pip install -e /opt/nemo-rl` resolve succeeds against torch 2.7.1
  on the `dcs-base-azure-devel` base (Python 3.12 line).
- [ ] The import gate passes (all native extensions load).
- [ ] Final image **< 20 GB** (`docker images`); add a multi-stage slim step if not.
- [ ] Pick the real NeMo-RL entry point (SFT/GRPO recipe) for `train.yaml`'s `command`.

## Azure AIR caveats

- **Multi-node H100 not available on Azure AIR** yet — single-node ≤ 8×H100 only
  (and subject to capacity). Set `compute.num_accelerators` ≤ 8.
- Custom images are **Beta** and **incompatible with Serverless Egress Control (SEG)**
  workspaces.
