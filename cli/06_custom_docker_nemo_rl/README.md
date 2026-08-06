# 06 — Custom Docker image (NeMo-RL) on AI Runtime

Reproduce an **NVIDIA NeMo-RL** environment (equivalent to
`nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano`) as a **Docker Hub** image that
AI Runtime can run.

## Why you can't use the `nvcr.io` image directly

AI Runtime custom images have hard constraints:

| Constraint | NeMo-RL `nvcr.io` image | Consequence |
|---|---|---|
| Registry must be **Docker Hub** | hosted on `nvcr.io` | must be rebuilt/re-pushed to Docker Hub |
| Size **< 20 GB** | large (full CUDA+TE+vLLM+Megatron) | must trim to fit |
| Python venv at **`/opt/venv`** (uv) | uses `/opt/nemo_rl_venv` | layout mismatch |
| AIR runs your `command`; **ENTRYPOINT/WORKDIR ignored** | `ENTRYPOINT /opt/nvidia/nvidia_entrypoint.sh`, `WORKDIR /opt/nemo-rl` | won't be honored |
| **FIPS-enabled** hosts | not FIPS-aware | may need `OPENSSL_FORCE_FIPS_MODE=0` |

The reference image is CUDA 12.9 / Python 3.12 / NCCL 2.26 with EFA. The AIR
base (`dcs-base-aws-devel`, verified from its image config) is **CUDA 12.9.1 /
NCCL 2.27 / venv at `/opt/venv` (uv)** with EFA — the same CUDA generation — so
**rebuilding on the AIR base is feasible**; it's the packaging / registry /
layout that differ, not the CUDA stack. (The AIR base's exact **torch** version
is not visible in its config — confirm it and match the Mamba wheel tags; see
the checklist below.)

## Approach

Start `FROM databricksruntime/air:dcs-base-aws-devel` (the **devel** variant —
NeMo-RL compiles CUDA extensions, which needs `nvcc`) and reinstall the NeMo-RL
stack into the AIR-managed venv. See [Dockerfile](Dockerfile).

## Build → register → run

```bash
# 1) BUILD for linux/amd64 (NOT on an arm64 Mac natively — cross-build or build
#    on an x86_64 Linux/GPU host). Push to Docker Hub.
docker buildx build --platform linux/amd64 \
  -t <your-dockerhub-user>/air-nemo-rl:v1 --push .

# 2) REGISTER the image with AIR (Docker Hub only)
air register image <your-dockerhub-user>/air-nemo-rl:v1 \
  --interactive-authenticate -p <your-profile>

# 3) RUN — edit train.yaml: set docker_image.url + the real NeMo-RL entry point
air run -f train.yaml -p <your-profile> --watch
```

## ⚠️ Status / not yet build-verified

This Dockerfile is a **design**, produced by inspecting the reference image's
config (CUDA 12.9, Python 3.12, uv, `NEMO_RL_COMMIT=8bdfd48…`, `/opt/nemo-rl`,
`/opt/nemo_rl_venv`) — it has **not been built or run yet**. The build machine
here is an arm64 Mac with no Docker/GPU, so the image cannot be built or
validated locally. Before using it for the PoC, verify on an x86_64 + GPU host:

- [ ] Confirm the **AIR base image's exact torch version** and swap the
  `mamba_ssm` / `causal_conv1d` wheel tags to match (the pinned wheels assume
  `torch2.9 / cu12 / cp312`, matching the `databricks_ai_v5` inline base — the
  docker base may differ).
- [ ] `uv pip install -e /opt/nemo-rl` resolves against the AIR venv without
  conflicting with the base image's pinned torch (NeMo-RL may want to pull its
  own torch — you may need `--no-deps` for torch or constraints).
- [ ] Final image size **< 20 GB** (trim build toolchains / use multi-stage if
  needed).
- [ ] Pick the real NeMo-RL entry point (SFT/GRPO recipe) for `command`.
