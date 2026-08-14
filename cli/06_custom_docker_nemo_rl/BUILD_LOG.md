# BUILD_LOG — NeMo-RL custom image for AI Runtime (Azure): trial-and-error journal

A chronological record of the hurdles hit while producing the NeMo-RL v0.4.0
custom Docker image for Databricks AI Runtime (Azure), and how each was solved.
Kept so the next person doesn't re-discover these the hard way.

Placeholders: `<registry-user>` (Docker Hub user), `<profile>` (Databricks CLI
profile), `<catalog>/<schema>` (UC volume). No account/customer identifiers.

Goal: reproduce `nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano` as a Docker Hub
image runnable on Azure AI Runtime serverless GPU. Final image tag:
`<registry-user>/air-nemo-rl:v0.4.0`.

---

## Summary of the final working recipe

- Base: `databricksruntime/air:dcs-base-azure-devel` (CUDA 12.9.1, venv `/opt/venv`, uv).
- Install `python3.12-dev` + `build-essential` (source builds need `Python.h`).
- Point compiler/linker at the pip cuDNN headers (`site-packages/nvidia/cudnn`).
- Install via **NeMo-RL's own uv workspace**: `uv sync --extra vllm --extra mcore`
  with `UV_PROJECT_ENVIRONMENT=/opt/venv` (lets uv honor its pinned torch cu128
  index, git-pinned mamba/causal, flash-attn metadata, no-build-isolation).
- Clean uv/pip/pyc/.git caches **in the same layer** to stay under 20 GB.
- Build-time gate: on a GPU-less host, verify packages are *installed*
  (`find_spec`); the full GPU import runs on AIR.
- Build must be `--platform linux/amd64`. Build needs NO GPU.

---

## Hurdle 1 — Databricks clusters cannot build images

- **Tried:** build on a Databricks GPU/CPU cluster (no docker present; installed
  BuildKit by hand).
- **Result:** `buildkitd` started, but every build failed with
  `failed to mount ... : permission denied` (bind mount blocked).
- **Root cause:** clusters run user code in an unprivileged sandbox that blocks
  the mounts image builds require. By design (AIR/clusters *run* images, they
  don't *build* them).
- **Fix:** build outside Databricks. Chose an Azure CPU VM.

## Hurdle 2 — GPU VM capacity failure

- **Tried:** a `p5.4xlarge` (1×H100) Databricks cluster as the build host.
- **Result:** `AWS_INSUFFICIENT_INSTANCE_CAPACITY_FAILURE` (no p5 capacity in AZ).
- **Lesson:** don't put the build on scarce GPU capacity — **the build doesn't
  need a GPU** (nvcc cross-compiles for sm90 without one). Use a cheap CPU VM;
  do the GPU smoke-test on AIR.

## Hurdle 3 — OS disk was 30 GB, not the requested 256 GB

- **Symptom:** `uv sync` died with `No space left on device` while unpacking
  `torch==2.7.1+cu128` (torch alone ~1 GB; the CUDA libs + vLLM are tens of GB).
- **Root cause:** `az vm create --os-disk-size-gb 256` did not take effect; the
  Ubuntu 22.04 gen2 image booted with a ~30 GB root (`/dev/root 29G`), and
  `lsblk` showed `sda` at 30 GB.
- **Fix:** create the VM with a genuinely large OS disk (recreate, or
  `az vm deallocate` → `az disk update --size-gb 256` → start →
  `growpart /dev/sda 1` + `resize2fs`). Verify with `df -h /` before building.

## Hurdle 4 — torch resolved to cu126 and mamba_ssm was missing

- **Tried (wrong):** hand-install `torch==2.7.1` then `uv pip install -e /opt/nemo-rl`.
- **Result:** `torch 2.7.1+cu126` (NeMo-RL wants **cu128**) and
  `ModuleNotFoundError: No module named 'mamba_ssm'`.
- **Root cause:** NeMo-RL's `pyproject.toml` puts mamba/causal/vllm/flash-attn in
  **optional-dependency extras**, and pins torch/torchvision to a **cu128 index**
  via `[tool.uv.sources]` (mamba/causal are git-pinned there; flash-attn uses
  dependency-metadata; several are `no-build-isolation-package`). A hand-rolled
  `pip install -e .` + manual torch bypassed all of that.
- **Fix:** stop hand-picking. Install through NeMo-RL's own uv workspace:
  `uv sync` with `UV_PROJECT_ENVIRONMENT=/opt/venv`, so uv applies the pinned
  sources/index/build-config and lands packages in the AIR venv.

## Hurdle 5 — `Python.h: No such file or directory`

- **Symptom:** `deep-gemm` (a NeMo-RL dep) failed to compile: `fatal error:
  Python.h`.
- **Root cause:** the base image uses system Python 3.12.3 but ships no dev
  headers; the compiler looks in `/usr/include/python3.12`.
- **Fix:** `apt-get install -y python3.12-dev build-essential`
  (candidate `3.12.3`, an exact-version match) before `uv sync`.

## Hurdle 6 — import gate false-failed on a GPU-less host

- **Symptom:** the verification `import mamba_ssm` raised
  `RuntimeError: 0 active drivers ([]). There should only be one.`
- **Root cause:** `mamba_ssm` → triton probes for a GPU driver at import; the
  build VM has no GPU. Not an image defect (works on AIR, which has GPUs).
- **Fix:** on a GPU-less build, verify packages are *installed* via
  `importlib.util.find_spec`; only do the full runtime import when
  `torch.cuda.is_available()`. Real GPU import happens on AIR at job launch.

## Hurdle 7 — transformer_engine genuinely absent

- **Symptom:** gate reported `MISSING packages: ['transformer_engine']`; a probe
  build confirmed nothing named `transformer_engine*` was installed.
- **Root cause:** in NeMo-RL v0.4.0, `transformer-engine[pytorch]==2.5.0` lives
  in the **`mcore`** extra, not `vllm`. `uv sync --extra vllm` skipped it.
- **Fix:** `uv sync --extra vllm --extra mcore` (vllm → mamba/causal/vllm/
  flash-attn; mcore → transformer-engine + the Megatron path).

## Hurdle 8 — `cudnn.h: No such file or directory` (TE compile)

- **Symptom:** Transformer-Engine failed to compile: torch's
  `ATen/cudnn/cudnn-wrapper.h` does `#include <cudnn.h>`, not found.
- **Root cause:** the base image has no system cuDNN headers.
- **Key insight:** NeMo-RL already pulls the `nvidia-cudnn-cu12` wheel (installed
  by `uv sync` *before* TE builds), which ships `cudnn.h` under
  `site-packages/nvidia/cudnn/include`. So **don't apt-install a second cuDNN**
  (version-duplication risk) — just expose the existing one.
- **Fix:** set `CPATH` / `LIBRARY_PATH` / `LD_LIBRARY_PATH` to the pip cuDNN
  `include`/`lib` dirs before `uv sync`.

## Hurdle 9 — image 21.6 GB > AIR's 20 GB limit

- **Symptom:** build+push succeeded (all extensions installed), but
  `docker images` showed **21.6 GB**; AIR custom images must be < 20 GB.
- **Fix (first attempt):** clean uv/pip caches, `*.pyc`/`__pycache__`, `/tmp`,
  and the NeMo-RL `.git` **in the same RUN layer** as `uv sync` (cleaning in a
  later layer doesn't shrink the earlier one).
- **If still over:** switch to a multi-stage build that copies only `/opt/venv`
  into a clean final stage (drops build-essential/apt residue). [pending]

---

## Verified facts worth keeping

- AIR custom images: **Docker Hub only** (ACR/ECR/GCR/GHCR unsupported), **<20 GB**,
  **ENTRYPOINT/WORKDIR ignored**, FIPS hosts (`OPENSSL_FORCE_FIPS_MODE=0` to opt
  out if compliance allows), **incompatible with Serverless Egress Control**, Beta.
- AIR `dcs-base-azure-devel`: CUDA 12.9.1, venv `/opt/venv`, uv-managed Python
  3.12.3, InfiniBand. No torch preinstalled (you bring it — which is good: you
  control the version to match the framework).
- NeMo-RL v0.4.0 pins: torch 2.7.1 (cu128), vllm 0.10.0, transformer-engine 2.5.0.
- Azure AIR: **single-node only** (no multi-node H100 yet) — keep
  `num_accelerators` ≤ 8.

## Still open

- Confirm the slimmed image is < 20 GB (else multi-stage).
- Register on AIR (`air register image ...`) and run a real 1×H100 job on AIR to
  confirm the GPU import path (`import transformer_engine`, etc.) actually works
  — the build only proves the packages are installed, not that they run on GPU.
