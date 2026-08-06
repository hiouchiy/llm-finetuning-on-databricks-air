# llm-finetuning-on-databricks-air

Databricks **AI Runtime (Serverless GPU / SGC)** version of
[`llm-finetuning-on-databricks`](https://github.com/hiouchiy/llm-finetuning-on-databricks).

The original repo fine-tunes **NVIDIA Nemotron-Nano-9B-v2** on the
`databricks-dolly-15k-ja-gozaru` dataset using **classic GPU clusters** (Spark
`TorchDistributor` / `DeepspeedTorchDistributor`). This repo ports every step to
run on **AI Runtime**, where there is no Spark cluster to manage — you submit
jobs to serverless H100s via the `air` CLI, or launch multi-GPU work from a
notebook with the `@distributed` decorator.

> Validated on `<your-profile>` (AWS). AI Runtime is Public Preview and, at
> the time of writing, US-region AWS/Azure only. Multi-node H100 (step 05) is
> available on AWS; **not yet on Azure**.

## What changed vs. the classic repo

| Concern | Classic (ML Runtime cluster) | AI Runtime |
|---|---|---|
| Launcher (multi-GPU) | `pyspark ... TorchDistributor` / `DeepspeedTorchDistributor` | `torchrun` (CLI) or `@distributed` (notebook) |
| Cluster | You size & manage a GPU cluster | Serverless — request `GPU_1xH100` / `GPU_8xH100` |
| Multi-node | `nnodes` on a Spark cluster | `num_accelerators: 16/64/...` (multiple of 8) |
| Spark / `dbutils` | Used for creds, `current_user`, FS | Not available; use env vars + UC Volume paths |
| MLflow | Manual `start_run` | AIR starts the run for you — **do not** `start_run` again |
| Env | ML Runtime 17.3 (torch pinned by DBR) | `databricks_ai_v5` base: **torch 2.9.0+cu129 / py3.12** |

## Repository layout

```
cli/                         # AI Runtime CLI (.py + .yaml), `air run -f ...`
  01_run_model/              #   inference smoke test (1xH100)
  02_sft_lora_single/        #   SFT + LoRA, single H100
  03_sft_lora_multi/         #   SFT + LoRA, 8xH100 DDP (torchrun)
  04_sft_fullft_zero2/       #   full-param SFT, DeepSpeed ZeRO-2, single 8xH100
  05_sft_fullft_multinode/   #   full-param SFT, ZeRO-2, MULTI-NODE (16x = 2 nodes)
  06_custom_docker_nemo_rl/  #   custom Docker image (NeMo-RL) — Dockerfile design only
notebook/                    # notebook (@distributed) variants — single node
_original_reference/         # the original classic-cluster notebooks, unchanged
setup/                       # one-time environment prep (wheels, volumes)
```

## Prerequisites

1. **AI Runtime CLI** (`air`), Public Preview:
   see <https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/>.
2. A Databricks profile with AI Runtime enabled (this repo uses
   `-p <your-profile>`).
3. A UC Volume for outputs and staged wheels (this repo uses the
   `<catalog>.<schema>.*` volumes; change the paths in the YAMLs to your own).
4. One-time setup: stage the Mamba build wheels — see [setup/README.md](setup/README.md).

## Nemotron-Nano-v2 needs `mamba_ssm` + `causal_conv1d`

Nemotron-Nano-9B-v2 is a **Mamba-Transformer hybrid**, so loading it requires
`mamba_ssm` and `causal_conv1d` (CUDA extensions). These must match the runtime's
torch/CUDA/Python **exactly**. AI Runtime `databricks_ai_v5` ships
**torch 2.9.0+cu129 / py3.12**, so we use the prebuilt wheels tagged
`cu12torch2.9cxx11abiTRUE-cp312`, staged in a UC Volume (see `setup/`).

## Running (CLI)

```bash
# macOS: strip AppleDouble/xattrs or the tarball entrypoint path breaks.
export COPYFILE_DISABLE=1

cd cli/02_sft_lora_single
air run -f train.yaml -p <your-profile> --watch

# quick smoke test: cap the steps
air run -f train.yaml -p <your-profile> --override env_variables.MAX_STEPS=5
```

Outputs (LoRA adapters / full models) are copied to the `OUTPUT_VOL` UC Volume
and logged to the MLflow experiment under `mlflow_experiment_directory`.

## Gotchas learned during the port

- **Do not `mlflow.start_run()` and do not use `report_to=["mlflow"]`.** AI
  Runtime starts the MLflow run before your script; HF's built-in MLflow
  integration calls `start_run()` again, which raises *"Run ... is already
  active"* and then **hangs the job until timeout**. Use `report_to=[]` and the
  small `AIRMLflowCallback` included in each `train.py`.
- **macOS uploads:** set `COPYFILE_DISABLE=1` and strip xattrs, otherwise the
  `._*` AppleDouble files corrupt `$CODE_SOURCE_PATH`.
- **`serverless_gpu` (`@distributed`) is only in the notebook environment**, not
  in the CLI worker image — the CLI path uses `torchrun` instead.
- **`experiment_name`** allows only alphanumerics / `-` / `_`; the workspace
  path goes in `mlflow_experiment_directory`.
