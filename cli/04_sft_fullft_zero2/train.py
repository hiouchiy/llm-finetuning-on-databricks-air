#!/usr/bin/env python3
"""SFT full-parameter fine-tuning with DeepSpeed ZeRO-2 on a single 8xH100 node
(AI Runtime CLI, torchrun).

AIR-CLI version of `04.SFT Full-Parameter on Multi GPU with
DeepSpeedDistributer.py`. The classic version used pyspark
DeepspeedTorchDistributor; here torchrun launches one process per GPU and
HF Trainer drives DeepSpeed ZeRO-2 via the deepspeed config JSON.
"""

import json
import os

import mlflow
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


class AIRMLflowCallback(TrainerCallback):
    """Log HF Trainer metrics to the AIR-owned active MLflow run (rank 0 only)."""

    def __init__(self, is_rank0: bool = True):
        self.enabled = is_rank0 and mlflow.active_run() is not None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or not logs:
            return
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                try:
                    mlflow.log_metric(k, float(v), step=state.global_step)
                except Exception as e:
                    print(f"[AIRMLflow] log_metric failed for {k}: {e}", flush=True)


MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
DATASET_ID = os.environ.get("DATASET_ID", "bbz662bbz/databricks-dolly-15k-ja-gozaru")
JSONL_PATH = os.environ.get("JSONL_PATH", "")
OUTPUT_VOL = os.environ.get(
    "OUTPUT_VOL", "/Volumes/<catalog>/<schema>/model/nemotron_nano_gozaru_fullft_zero2"
)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def build_user_text(ex):
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    return f"{inst}\n\n[入力]\n{inp}" if inp else inst


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0 = global_rank == 0

    if is_rank0:
        print(f"🧭 global_rank={global_rank} world_size={world_size} "
              f"local_rank={local_rank} device_count={torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(ex):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": build_user_text(ex)},
            {"role": "assistant", "content": (ex.get("output") or "").strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    if JSONL_PATH:
        ds = load_dataset("json", data_files=JSONL_PATH, split="train")
    else:
        ds = load_dataset(DATASET_ID, split="train")
    ds = ds.map(to_text, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to(local_rank)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    output_dir = "/local_disk0/nemotron_nano_9b_gozaru_fullft_zero2"

    # DeepSpeed ZeRO-2 (no offload) — same config as the classic version.
    ds_config = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
    }
    ds_config_path = "/tmp/ds_zero2_no_offload.json"
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)

    max_steps = int(os.environ.get("MAX_STEPS", "-1"))
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        deepspeed=ds_config_path,
        optim="adamw_torch_fused",
        # report_to=[] on AIR (see AIRMLflowCallback); avoid HF start_run conflict.
        report_to=[],
        max_length=2048,
        packing=False,
        disable_tqdm=(not is_rank0),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # AIR already has an active MLflow run; attach (never start_run) on rank 0.
    active = mlflow.active_run() if is_rank0 else None
    if active is not None:
        mlflow.set_tag("base_model", MODEL_ID)
        mlflow.set_tag("dataset", DATASET_ID)
        mlflow.set_tag("task", "SFT full-parameter (DeepSpeed ZeRO-2, single 8xH100, AIR CLI)")
        mlflow.log_params({"world_size": world_size, "lr": 2e-4, "zero_stage": 2})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=args,
        callbacks=[AIRMLflowCallback(is_rank0=is_rank0)],
    )

    train_result = trainer.train()

    if is_rank0:
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("✅ Training done (ZeRO-2). model_dir:", output_dir)

        os.makedirs(OUTPUT_VOL, exist_ok=True)
        import shutil

        for name in os.listdir(output_dir):
            src = os.path.join(output_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(OUTPUT_VOL, name))
        print("✅ Copied model to UC Volume:", OUTPUT_VOL)

        if active is not None:
            mlflow.log_artifacts(output_dir, artifact_path="model")
            for k, v in (train_result.metrics or {}).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            # Do not end_run(): AIR owns the run lifecycle.

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
