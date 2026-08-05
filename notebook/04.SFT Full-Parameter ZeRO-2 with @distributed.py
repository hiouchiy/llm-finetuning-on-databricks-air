# Databricks notebook source
# MAGIC %md
# MAGIC # SFT Full-Parameter (DeepSpeed ZeRO-2) with AI Runtime `@distributed`
# MAGIC
# MAGIC AI Runtime notebook version of the classic
# MAGIC `04.SFT Full-Parameter on Multi GPU with DeepSpeedDistributer.py`.
# MAGIC
# MAGIC Replaces Spark's `DeepspeedTorchDistributor` with the `@distributed`
# MAGIC decorator on a single 8xH100 node. HF Trainer drives DeepSpeed ZeRO-2 via
# MAGIC the deepspeed JSON config.
# MAGIC
# MAGIC > Multi-node full-parameter training → use `cli/05_sft_fullft_multinode`
# MAGIC > (the `air` CLI, torchrun). Notebooks are single-node only.

# COMMAND ----------

# MAGIC %pip install transformers==4.57.3 tokenizers==0.22.1 trl peft deepspeed hf_transfer
# MAGIC %pip install /Volumes/hiroshi/tmp/wheels/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC %pip install /Volumes/hiroshi/tmp/wheels/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from serverless_gpu import distributed

NUM_GPUS = 8  # single 8xH100 node


@distributed(gpus=NUM_GPUS, gpu_type="H100", remote=True)
def train_nemotron_fullft_zero2():
    import json
    import os

    import mlflow
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    DATASET_ID = "bbz662bbz/databricks-dolly-15k-ja-gozaru"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    is_rank0 = dist.get_rank() == 0

    class AIRMLflowCallback(TrainerCallback):
        def __init__(self, enabled):
            self.enabled = enabled

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not self.enabled or not logs:
                return
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    try:
                        mlflow.log_metric(k, float(v), step=state.global_step)
                    except Exception:
                        pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_user_text(ex):
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        return f"{inst}\n\n[入力]\n{inp}" if inp else inst

    def to_text(ex):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": build_user_text(ex)},
            {"role": "assistant", "content": (ex.get("output") or "").strip()},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    ds = load_dataset(DATASET_ID, split="train").map(to_text)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(local_rank)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

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

    args = SFTConfig(
        output_dir="/local_disk0/nemotron_fullft_zero2",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        deepspeed=ds_config_path,
        optim="adamw_torch_fused",
        report_to=[],  # do NOT use "mlflow" on AIR (start_run conflict)
        max_length=2048,
        packing=False,
        disable_tqdm=(not is_rank0),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=args,
        callbacks=[AIRMLflowCallback(enabled=is_rank0)],
    )
    trainer.train()

    if is_rank0:
        out = "/local_disk0/nemotron_fullft_zero2"
        trainer.model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print("✅ Training done (ZeRO-2). model_dir:", out)

    dist.barrier()
    dist.destroy_process_group()


# COMMAND ----------

train_nemotron_fullft_zero2.distributed()
