# Databricks notebook source
# MAGIC %md
# MAGIC # SFT + LoRA on Multi-GPU with AI Runtime `@distributed`
# MAGIC
# MAGIC AI Runtime (Serverless GPU) notebook version of the classic
# MAGIC `03.SFT+LoRA on Multi GPU with TorchDistributer.ipynb`.
# MAGIC
# MAGIC Instead of Spark's `TorchDistributor`, we use the **`@distributed`
# MAGIC decorator** from the `serverless_gpu` library. Attach this notebook to
# MAGIC **Serverless GPU** compute and select the **H100** accelerator, then run.
# MAGIC The decorator launches one process per GPU on a single 8xH100 node.
# MAGIC
# MAGIC > Multi-node (gpus > 8) is not supported from a notebook — use the
# MAGIC > `cli/03_sft_lora_multi` (torchrun) or `cli/05_sft_fullft_multinode`
# MAGIC > path with the `air` CLI for that.

# COMMAND ----------

# MAGIC %pip install transformers==4.57.3 tokenizers==0.22.1 trl peft hf_transfer
# MAGIC # Nemotron-Nano-v2 (Mamba-Transformer hybrid) build deps — staged in a UC Volume.
# MAGIC %pip install /Volumes/<catalog>/<schema>/wheels/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC %pip install /Volumes/<catalog>/<schema>/wheels/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from serverless_gpu import distributed

NUM_GPUS = 8  # single 8xH100 node


@distributed(gpus=NUM_GPUS, gpu_type="H100", remote=True)
def train_nemotron_lora():
    import os

    import mlflow
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    DATASET_ID = "bbz662bbz/databricks-dolly-15k-ja-gozaru"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # serverless_gpu sets the standard distributed env vars.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    is_rank0 = dist.get_rank() == 0

    # AIR owns the MLflow run; attach, never start_run. Log via a small callback.
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

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.config.pad_token_id = tokenizer.pad_token_id

    args = SFTConfig(
        output_dir="/local_disk0/nemotron_lora",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        optim="adamw_torch_fused",
        report_to=[],  # do NOT use "mlflow" on AIR (start_run conflict)
        max_length=2048,
        packing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
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
        adapter_dir = "/local_disk0/nemotron_lora_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print("✅ Training done. adapter_dir:", adapter_dir)

    dist.destroy_process_group()


# COMMAND ----------

# Launch the multi-GPU job on Serverless GPU.
train_nemotron_lora.distributed()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC - `@distributed(gpus=8, gpu_type="H100", remote=True)` runs the function on
# MAGIC   a serverless 8xH100 node, one process per GPU.
# MAGIC - Set `remote=False` to run against a GPU your notebook is already attached
# MAGIC   to (useful for quick local iteration).
# MAGIC - For multi-node, this notebook path does not apply — use the `air` CLI
# MAGIC   (`cli/05_sft_fullft_multinode`).
