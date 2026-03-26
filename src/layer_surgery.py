"""
Layer surgery module: remove, reorder, and stitch transformer layers.

Supports Qwen-family models where decoder layers live under
model.model.layers (nn.ModuleList).
"""

import copy
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger("layer_surgery")


def get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return reference to the decoder layer ModuleList."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Cannot locate decoder layers. Expected model.model.layers or model.transformer.h"
    )


def set_decoder_layers(model: PreTrainedModel, new_layers: nn.ModuleList):
    """Replace the decoder layers in the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(new_layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = new_layers
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(new_layers)
    else:
        raise AttributeError("Cannot set decoder layers.")

    for new_idx, layer in enumerate(new_layers):
        for attn_name in ("self_attn", "linear_attn"):
            attn = getattr(layer, attn_name, None)
            if attn is not None and hasattr(attn, "layer_idx"):
                attn.layer_idx = new_idx


def remove_layers(
    model: PreTrainedModel,
    layer_indices_to_remove: List[int],
    inplace: bool = True,
) -> PreTrainedModel:
    """
    Remove specified layers from the model.

    Args:
        model: The transformer model
        layer_indices_to_remove: Indices of layers to remove (0-indexed)
        inplace: If True, modify model in place; otherwise deepcopy first

    Returns:
        Model with layers removed
    """
    if not inplace:
        model = copy.deepcopy(model)

    layers = get_decoder_layers(model)
    total = len(layers)
    keep = [i for i in range(total) if i not in layer_indices_to_remove]

    if not keep:
        raise ValueError("Cannot remove all layers")

    new_layers = nn.ModuleList([layers[i] for i in keep])
    set_decoder_layers(model, new_layers)

    logger.info(
        "Removed layers %s, kept %d/%d layers",
        sorted(layer_indices_to_remove), len(keep), total,
    )
    return model


def keep_prefix_layers(
    model: PreTrainedModel,
    k: int,
    inplace: bool = True,
) -> PreTrainedModel:
    """Keep only the first k decoder layers."""
    layers = get_decoder_layers(model)
    total = len(layers)
    if k >= total:
        logger.warning("k=%d >= total layers=%d, returning unchanged model", k, total)
        return model

    remove_indices = list(range(k, total))
    return remove_layers(model, remove_indices, inplace=inplace)


def compute_layer_importance(
    model: PreTrainedModel,
    dataloader,
    metric: str = "gradient_norm",
    device: str = "cuda",
) -> List[float]:
    """
    Compute importance score for each decoder layer.

    Args:
        model: The transformer model
        dataloader: DataLoader yielding input_ids tensors
        metric: One of 'gradient_norm', 'activation_norm', 'fisher'

    Returns:
        List of importance scores, one per layer
    """
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    importance = [0.0] * n_layers

    if metric == "gradient_norm":
        importance = _importance_gradient_norm(model, layers, dataloader, device)
    elif metric == "activation_norm":
        importance = _importance_activation_norm(model, layers, dataloader, device)
    elif metric == "fisher":
        importance = _importance_fisher(model, layers, dataloader, device)
    else:
        raise ValueError(f"Unknown importance metric: {metric}")

    total = sum(importance)
    if total > 0:
        importance = [s / total for s in importance]

    return importance


def _importance_gradient_norm(model, layers, dataloader, device) -> List[float]:
    """Layer importance by accumulated gradient norm."""
    n_layers = len(layers)
    grad_norms = [0.0] * n_layers

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 32:
            break
        input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
        model.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for i, layer in enumerate(layers):
            layer_norm = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    layer_norm += p.grad.data.norm(2).item() ** 2
            grad_norms[i] += layer_norm ** 0.5

    model.eval()
    return grad_norms


def _importance_activation_norm(model, layers, dataloader, device) -> List[float]:
    """Layer importance by activation norm (output hidden states)."""
    n_layers = len(layers)
    act_norms = [0.0] * n_layers
    counts = [0] * n_layers

    hooks = []
    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            act_norms[idx] += h.float().norm(2).item()
            counts[idx] += 1
        return hook_fn

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 32:
                break
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            model(input_ids)

    for h in hooks:
        h.remove()

    return [act_norms[i] / max(counts[i], 1) for i in range(n_layers)]


def _importance_fisher(model, layers, dataloader, device) -> List[float]:
    """Layer importance by Fisher information (squared gradient)."""
    n_layers = len(layers)
    fisher = [0.0] * n_layers

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 32:
            break
        input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
        model.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for i, layer in enumerate(layers):
            for p in layer.parameters():
                if p.grad is not None:
                    fisher[i] += (p.grad.data ** 2).sum().item()

    model.eval()
    return fisher


def importance_based_removal(
    model: PreTrainedModel,
    dataloader,
    fraction_to_remove: float,
    metric: str = "gradient_norm",
    device: str = "cuda",
    inplace: bool = True,
) -> PreTrainedModel:
    """Remove the least important fraction of layers."""
    importance = compute_layer_importance(model, dataloader, metric, device)
    n_layers = len(importance)
    n_remove = max(1, int(n_layers * fraction_to_remove))

    sorted_indices = sorted(range(n_layers), key=lambda i: importance[i])
    to_remove = sorted_indices[:n_remove]

    logger.info(
        "Removing %d least important layers (metric=%s): %s",
        n_remove, metric, sorted(to_remove),
    )
    return remove_layers(model, to_remove, inplace=inplace)
