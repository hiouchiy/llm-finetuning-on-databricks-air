# Databricks notebook source
# MAGIC %md
# MAGIC # Run Nemotron-Nano-9B-v2 (inference) on AI Runtime
# MAGIC
# MAGIC AI Runtime notebook version of `01.Run the model.ipynb`.
# MAGIC Attach this notebook to **Serverless GPU** compute and select the
# MAGIC **H100** (or A10) accelerator, then run. Single-GPU inference needs no
# MAGIC `@distributed` decorator.

# COMMAND ----------

# MAGIC %pip install transformers==4.57.3 tokenizers==0.22.1 hf_transfer
# MAGIC # Nemotron-Nano-v2 (Mamba-Transformer hybrid) build deps — staged in a UC Volume.
# MAGIC %pip install /Volumes/hiroshi/tmp/wheels/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC %pip install /Volumes/hiroshi/tmp/wheels/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# COMMAND ----------

messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "Write a haiku about CPUs in Japanese"},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    tokenized_chat, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))
