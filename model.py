"""Loading KokLLaMA (or the base model) under 4-bit NF4 quantization."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


def bnb_config():
    """QLoRA inference quantization config (paper Section 4.1)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(model_id=None):
    """Return (tokenizer, model) ready for left-padded batched generation."""
    model_id = model_id or config.MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Loaded {model_id}. Device map: {getattr(model, 'hf_device_map', None)}")
    return tokenizer, model
