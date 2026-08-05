#!/usr/bin/env python3
"""Load Nemotron-Nano-9B-v2 and run a quick generation (AI Runtime CLI).

AIR-CLI version of `01.Run the model.ipynb`. Single H100, no Spark/dbutils.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

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


if __name__ == "__main__":
    main()
