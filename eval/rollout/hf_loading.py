"""One loader for every inference-only HF model (teacher, frontier references).

Gemma 3 >= 4B checkpoints (``google/gemma-3-27b-it``) are multimodal
``Gemma3ForConditionalGeneration`` models. ``AutoModelForCausalLM`` resolves
their config to the text-only class and fails or silently mis-loads weights on
transformers 4.51, so they are loaded with the conditional-generation class and
used text-only, which is how ``generate`` treats a prompt without images.
"""

from __future__ import annotations

from typing import Any


def load_causal_lm(
    model_id: str,
    *,
    use_4bit: bool = False,
    device_map: str | dict[str, Any] = "auto",
    attn_implementation: str = "sdpa",
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "model_type", "") == "gemma3" and hasattr(cfg, "vision_config"):
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer
