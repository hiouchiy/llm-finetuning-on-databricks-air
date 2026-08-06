# One-time setup: stage Mamba wheels in a UC Volume

Nemotron-Nano-9B-v2 is a Mamba-Transformer hybrid and needs `mamba_ssm` +
`causal_conv1d` (CUDA extensions). They must match the AI Runtime environment
**exactly**:

- AI Runtime `databricks_ai_v5` → **torch 2.9.0+cu129, Python 3.12**
- So we use wheels tagged `cu12torch2.9cxx11abiTRUE-cp312`.

Building these from source on a worker is slow and flaky, so we stage the
official prebuilt wheels in a UC Volume once and reference them from each
`train.yaml` (`environment.dependencies`).

## Steps

```bash
export DATABRICKS_CONFIG_PROFILE=<your-profile>

# 1) Create the volume (once)
databricks schemas create <schema> <catalog>
databricks volumes create <catalog> <schema> wheels MANAGED

# 2) Download the prebuilt wheels (torch2.9 / cu12 / cp312 / cxx11abiTRUE)
CC=causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
MS=mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
curl -sL -o "$CC" "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/$CC"
curl -sL -o "$MS" "https://github.com/state-spaces/mamba/releases/download/v2.3.1/$MS"

# 3) Upload to the UC Volume
databricks fs cp "$CC" "dbfs:/Volumes/<catalog>/<schema>/wheels/$CC" --overwrite
databricks fs cp "$MS" "dbfs:/Volumes/<catalog>/<schema>/wheels/$MS" --overwrite
```

If your runtime's torch/cuda/python ever change, pick matching wheels from:

- causal-conv1d releases: <https://github.com/Dao-AILab/causal-conv1d/releases>
- mamba releases: <https://github.com/state-spaces/mamba/releases>

The wheel's `torchX.Y` tag must equal the runtime torch version, or you'll get a
CUDA/ABI mismatch at import.
