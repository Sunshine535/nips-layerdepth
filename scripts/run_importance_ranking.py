#!/usr/bin/env python3
"""
Compute layer importance via three metrics and compare their rankings.

Metrics:  gradient_norm · activation_norm · fisher
Outputs:  importance_scores.json   — raw and normalised scores per layer
          importance_comparison.pdf — bar charts + rank correlation matrix
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layer_surgery import compute_layer_importance, get_decoder_layers
from src.model_utils import load_model_and_tokenizer, get_model_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("importance_ranking")

METRICS = ["gradient_norm", "activation_norm", "fisher"]


class CalibrationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return self.encodings.input_ids.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.encodings.input_ids[idx]}


def parse_args():
    parser = argparse.ArgumentParser(description="Layer Importance Ranking")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--output_dir", type=str, default="./results/importance")
    parser.add_argument("--cal_samples", type=int, default=100,
                        help="Number of GSM8K-train samples for calibration")
    parser.add_argument("--cal_batch_size", type=int, default=4)
    parser.add_argument("--cal_max_length", type=int, default=256)
    parser.add_argument("--metrics", nargs="+", default=METRICS,
                        choices=METRICS)
    return parser.parse_args()


def build_calibration_loader(tokenizer, n_samples, batch_size, max_length):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    texts = [ex["question"] for ex in ds.select(range(min(n_samples, len(ds))))]
    cal_ds = CalibrationDataset(texts, tokenizer, max_length)
    return DataLoader(cal_ds, batch_size=batch_size, shuffle=False)


def rank_layers(scores: list[float]) -> list[int]:
    """Return 0-based ranks (0 = least important)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def plot_importance(all_scores: dict, corr_matrix: dict, n_layers: int, output_path: str):
    n_metrics = len(all_scores)
    fig, axes = plt.subplots(n_metrics + 1, 1, figsize=(16, 4 * (n_metrics + 1)))

    layer_ids = list(range(n_layers))
    colors = {"gradient_norm": "#2196F3", "activation_norm": "#4CAF50", "fisher": "#FF9800"}

    for ax_idx, (metric, scores) in enumerate(all_scores.items()):
        ax = axes[ax_idx]
        ax.bar(layer_ids, scores, color=colors.get(metric, "#999"), alpha=0.85, width=0.8)
        ax.set_ylabel("Normalised Score", fontsize=11)
        ax.set_title(f"Layer Importance — {metric}", fontsize=13)
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.tick_params(axis="x", labelsize=7)

    # Rank correlation heatmap
    ax_corr = axes[-1]
    metric_names = list(corr_matrix.keys())
    n = len(metric_names)
    matrix = np.ones((n, n))
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            matrix[i, j] = corr_matrix[m1].get(m2, 1.0)

    im = ax_corr.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(n))
    ax_corr.set_yticks(range(n))
    ax_corr.set_xticklabels(metric_names, fontsize=10)
    ax_corr.set_yticklabels(metric_names, fontsize=10)
    ax_corr.set_title("Spearman Rank Correlation Between Metrics", fontsize=13)
    for i in range(n):
        for j in range(n):
            ax_corr.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax_corr, shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved to %s", output_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading model: %s", args.model_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    n_layers = len(get_decoder_layers(model))
    logger.info("Model loaded: %d layers", n_layers)

    cal_loader = build_calibration_loader(
        tokenizer, args.cal_samples, args.cal_batch_size, args.cal_max_length,
    )

    all_scores = {}
    all_ranks = {}

    for metric in args.metrics:
        logger.info("Computing importance: %s", metric)
        scores = compute_layer_importance(model, cal_loader, metric, device=str(get_model_device(model)))
        all_scores[metric] = scores
        all_ranks[metric] = rank_layers(scores)
        logger.info("  top-5 layers: %s",
                     sorted(range(n_layers), key=lambda i: scores[i], reverse=True)[:5])

    # Spearman rank correlation
    corr_matrix = {}
    for m1 in args.metrics:
        corr_matrix[m1] = {}
        for m2 in args.metrics:
            rho, pval = spearmanr(all_ranks[m1], all_ranks[m2])
            corr_matrix[m1][m2] = round(float(rho), 4)
            if m1 != m2:
                logger.info("  Spearman(%s, %s) = %.4f  (p=%.2e)", m1, m2, rho, pval)

    # Save scores
    output_data = {
        "model": args.model_path,
        "n_layers": n_layers,
        "cal_samples": args.cal_samples,
        "scores": {m: all_scores[m] for m in args.metrics},
        "ranks": {m: all_ranks[m] for m in args.metrics},
        "spearman_correlation": corr_matrix,
    }

    scores_path = os.path.join(args.output_dir, "importance_scores.json")
    with open(scores_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Scores saved to %s", scores_path)

    # Plot
    plot_importance(
        all_scores, corr_matrix, n_layers,
        os.path.join(args.output_dir, "importance_comparison.pdf"),
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
