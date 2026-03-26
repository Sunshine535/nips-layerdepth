"""
Shared model loading utilities.

Handles flash-attention fallback and reliable device detection
for models loaded with device_map="auto" across multiple GPUs.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _best_attn_implementation() -> str:
    """Return the best available attention implementation, with runtime validation."""
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, k, v)
        return "flash_attention_2"
    except Exception as exc:
        logger.info("flash-attn unavailable or broken (%s), using sdpa", exc)
        return "sdpa"


def get_model_device(model) -> torch.device:
    """Reliably get the input device for a model (works with device_map='auto')."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        return torch.device("cuda:0")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_pretrained(model_path: str, torch_dtype, attn_impl: str):
    """Try loading with dtype= first (new API), fall back to torch_dtype= (old API)."""
    kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, **kwargs,
        )


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype=torch.bfloat16,
    attn_implementation: Optional[str] = None,
):
    """
    Load model + tokenizer with automatic flash-attention fallback.

    Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = attn_implementation or _best_attn_implementation()

    try:
        model = _load_pretrained(model_path, torch_dtype, attn_impl)
    except (ValueError, ImportError) as exc:
        logger.warning(
            "attn_implementation=%s failed (%s), retrying with sdpa",
            attn_impl, exc,
        )
        model = _load_pretrained(model_path, torch_dtype, "sdpa")

    model.eval()
    return model, tokenizer
