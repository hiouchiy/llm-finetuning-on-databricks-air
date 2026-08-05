# Databricks notebook source
# MAGIC %md
# MAGIC # SFT + LoRA on a single GPU (AI Runtime notebook)
# MAGIC
# MAGIC AI Runtime notebook version of `02.SFT+LoRA on Single GPU with HF TRL.ipynb`.
# MAGIC Attach to **Serverless GPU** compute with the **H100** accelerator and run.
# MAGIC Single-GPU training needs no `@distributed` decorator.

# COMMAND ----------

# MAGIC %pip install transformers==4.57.3 tokenizers==0.22.1 trl peft hf_transfer
# MAGIC %pip install /Volumes/hiroshi/tmp/wheels/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC %pip install /Volumes/hiroshi/tmp/wheels/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
DATASET_ID = "bbz662bbz/databricks-dolly-15k-ja-gozaru"

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


ds = load_dataset(DATASET_ID, split="train").map(to_text, remove_columns=None)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.config.use_cache = False

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules="all-linear",
)
model = get_peft_model(model, lora)
model.config.pad_token_id = tokenizer.pad_token_id

# COMMAND ----------

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
    # On AIR notebooks MLflow autologging works; report_to=["mlflow"] is fine
    # here because the notebook (not a pre-started AIR job run) owns the run.
    report_to=["mlflow"],
    max_length=2048,
    packing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=args,
)

# COMMAND ----------

with mlflow.start_run(run_name="nemotron_nano_9b_gozaru_lora_sft"):
    mlflow.set_tag("base_model", MODEL_ID)
    mlflow.set_tag("dataset", DATASET_ID)
    train_result = trainer.train()

    adapter_dir = "/local_disk0/nemotron_lora_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    mlflow.log_artifacts(adapter_dir, artifact_path="lora_adapter")

print("✅ done. adapter_dir:", adapter_dir)
