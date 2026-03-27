#!/usr/bin/env python3
"""
Depth scaling law analysis.

For each task (GSM8K, MMLU subcategories), find the minimum depth that
achieves ≥95% of full-model accuracy.  Then fit:

    min_depth(complexity) = a · log(complexity) + b

Outputs:
    depth_scaling_law.pdf          — scatter + fitted curve
    minimum_depth_heatmap.pdf      — per-subcategory MVD heatmap
    scaling_law_analysis.json      — all numerical results
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.optimize import curve_fit
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layer_surgery import get_decoder_layers, set_decoder_layers
from src.model_utils import load_model_and_tokenizer, get_model_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scaling_law")


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Scaling Law Analysis")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--output_dir", type=str, default="./results/scaling_law")
    parser.add_argument("--max_samples_per_task", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Fraction of full-model accuracy required")
    parser.add_argument("--depth_candidates", nargs="+", type=int,
                        default=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mmlu_subjects", nargs="+", default=None,
                        help="Specific MMLU subjects; default: auto-select 10")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


# ── Evaluation helpers ───────────────────────────────────────────────────

def extract_answer_gsm8k(text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(get_model_device(model))
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def eval_gsm8k_at_depth(model, tokenizer, original_layers, depth, samples, max_new_tokens):
    new_layers = torch.nn.ModuleList(original_layers[:depth])
    set_decoder_layers(model, new_layers)

    correct, total = 0, 0
    for ex in samples:
        text = generate_answer(model, tokenizer, ex["prompt"], max_new_tokens)
        pred = extract_answer_gsm8k(text)
        try:
            correct += int(abs(float(pred) - float(ex["gold"])) < 1e-3) if pred and ex["gold"] else 0
        except ValueError:
            correct += int(pred.strip() == ex["gold"].strip())
        total += 1

    set_decoder_layers(model, torch.nn.ModuleList(original_layers))
    return correct / max(total, 1)


def eval_mmlu_at_depth(model, tokenizer, original_layers, depth, samples, _max_new_tokens):
    new_layers = torch.nn.ModuleList(original_layers[:depth])
    set_decoder_layers(model, new_layers)

    correct, total = 0, 0
    for ex in samples:
        answer = generate_answer(model, tokenizer, ex["prompt"], max_new_tokens=16)
        pred = answer.strip().upper()[:1]
        correct += int(pred == ex["gold"])
        total += 1

    set_decoder_layers(model, torch.nn.ModuleList(original_layers))
    return correct / max(total, 1)


# ── Task loaders ─────────────────────────────────────────────────────────

def load_gsm8k_samples(max_n):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(max_n, len(ds))))
    samples = []
    for ex in ds:
        prompt = (f"Solve step by step. Put final answer after ####.\n\n"
                  f"Question: {ex['question']}\n\nAnswer:")
        gold = extract_answer_gsm8k(ex["answer"])
        samples.append({"prompt": prompt, "gold": gold})
    return samples


def load_mmlu_subject_samples(subject, max_n):
    choices = ["A", "B", "C", "D"]
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception:
        try:
            ds = load_dataset("cais/mmlu", "all", split="test")
            ds = ds.filter(lambda x: x.get("subject", "") == subject)
        except Exception:
            return []

    ds = ds.select(range(min(max_n, len(ds))))
    samples = []
    for ex in ds:
        options = ex.get("choices", [ex.get(c, "") for c in choices])
        prompt = f"Question: {ex['question']}\n"
        for i, opt in enumerate(options if isinstance(options, list) else [options]):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer with just the letter (A, B, C, or D):\n"
        gold_idx = ex.get("answer", 0)
        gold = choices[gold_idx] if isinstance(gold_idx, int) and gold_idx < 4 else str(gold_idx).upper()[:1]
        samples.append({"prompt": prompt, "gold": gold})
    return samples


DEFAULT_MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_mathematics",
    "computer_security", "econometrics", "high_school_physics",
    "machine_learning", "moral_scenarios", "professional_law",
]


# ── Fitting functions ────────────────────────────────────────────────────

def log_model(x, a, b):
    return a * np.log(np.clip(x, 1e-8, None)) + b


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    n_total_layers = max(args.depth_candidates)

    logger.info("Loading model: %s", args.model_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    original_layers = list(get_decoder_layers(model))
    n_layers = len(original_layers)
    logger.info("Model loaded: %d layers", n_layers)

    # Adjust depth candidates to actual model
    depth_candidates = sorted([d for d in args.depth_candidates if d <= n_layers])
    if n_layers not in depth_candidates:
        depth_candidates.append(n_layers)

    results_path = os.path.join(args.output_dir, "depth_sweep_raw.json")
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        logger.info("Resumed %d tasks from cache", len(all_results))
    else:
        all_results = {}

    # Build task list
    tasks = {}
    gsm8k_samples = load_gsm8k_samples(args.max_samples_per_task)
    tasks["gsm8k"] = {"samples": gsm8k_samples, "eval_fn": eval_gsm8k_at_depth}

    subjects = args.mmlu_subjects or DEFAULT_MMLU_SUBJECTS
    for subj in subjects:
        samples = load_mmlu_subject_samples(subj, args.max_samples_per_task)
        if samples:
            tasks[f"mmlu_{subj}"] = {"samples": samples, "eval_fn": eval_mmlu_at_depth}
            logger.info("MMLU/%s: %d samples", subj, len(samples))
        else:
            logger.warning("No data for MMLU/%s, skipping", subj)

    # Sweep depths per task
    for task_name, task_info in tasks.items():
        if task_name in all_results and args.resume:
            logger.info("Skipping %s (cached)", task_name)
            continue

        logger.info("=== %s ===", task_name)
        depth_accs = {}
        for depth in tqdm(depth_candidates, desc=task_name):
            acc = task_info["eval_fn"](
                model, tokenizer, original_layers, depth,
                task_info["samples"], args.max_new_tokens,
            )
            depth_accs[str(depth)] = acc
            logger.info("  depth=%d: acc=%.4f", depth, acc)

        all_results[task_name] = depth_accs

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Compute MVD (minimum viable depth) per task
    mvd_data = {}
    for task_name, depth_accs in all_results.items():
        full_acc = depth_accs.get(str(n_layers), depth_accs.get(str(max(depth_candidates)), 0))
        target = args.threshold * full_acc
        mvd = n_layers
        for d in sorted(depth_candidates):
            if depth_accs.get(str(d), 0) >= target:
                mvd = d
                break
        complexity = 1.0 - full_acc if full_acc > 0 else 0.5
        mvd_data[task_name] = {
            "full_accuracy": full_acc,
            "target_accuracy": target,
            "mvd": mvd,
            "mvd_fraction": mvd / n_layers,
            "complexity": complexity,
        }
        logger.info("%s: full_acc=%.3f, MVD=%d/%d, complexity=%.3f",
                     task_name, full_acc, mvd, n_layers, complexity)

    # Fit scaling law: min_depth = a * log(complexity) + b
    complexities = np.array([v["complexity"] for v in mvd_data.values()])
    mvd_fracs = np.array([v["mvd_fraction"] for v in mvd_data.values()])

    fit_result = {}
    try:
        popt, pcov = curve_fit(log_model, complexities, mvd_fracs, maxfev=5000)
        y_pred = log_model(complexities, *popt)
        residuals = mvd_fracs - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((mvd_fracs - np.mean(mvd_fracs)) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-8)
        fit_result = {
            "a": float(popt[0]),
            "b": float(popt[1]),
            "r_squared": float(r_squared),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        }
        logger.info("Fit: min_depth_frac = %.4f * log(complexity) + %.4f  (R²=%.4f)",
                     popt[0], popt[1], r_squared)
    except Exception as e:
        logger.error("Curve fitting failed: %s", e)
        fit_result = {"error": str(e)}

    # ── Plots ────────────────────────────────────────────────────────────

    # 1. Depth scaling law scatter + curve
    fig, ax = plt.subplots(figsize=(10, 7))
    task_names = list(mvd_data.keys())
    cx = [mvd_data[t]["complexity"] for t in task_names]
    my = [mvd_data[t]["mvd"] for t in task_names]

    ax.scatter(cx, my, s=80, zorder=5, color="#2196F3", edgecolor="black", linewidth=0.5)
    for i, name in enumerate(task_names):
        short = name.replace("mmlu_", "").replace("_", " ")[:15]
        ax.annotate(short, (cx[i], my[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)

    if "a" in fit_result:
        x_fit = np.linspace(max(min(cx) * 0.5, 0.01), max(cx) * 1.2, 100)
        y_fit = fit_result["a"] * np.log(x_fit) + fit_result["b"]
        ax.plot(x_fit, y_fit * n_layers, "r--", linewidth=2,
                label=f'a·log(c)+b  (R²={fit_result["r_squared"]:.3f})')
        ax.legend(fontsize=11)

    ax.set_xlabel("Task Complexity (1 − full accuracy)", fontsize=13)
    ax.set_ylabel(f"Minimum Viable Depth (layers, {args.threshold:.0%} threshold)", fontsize=13)
    ax.set_title("Depth Scaling Law: Task Complexity → Required Depth", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "depth_scaling_law.pdf"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # 2. MVD heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, len(task_names) * 0.5 + 1)))
    sorted_tasks = sorted(task_names, key=lambda t: mvd_data[t]["mvd"])
    mvd_vals = [mvd_data[t]["mvd"] for t in sorted_tasks]
    labels = [t.replace("mmlu_", "MMLU/").replace("_", " ") for t in sorted_tasks]

    heatmap_data = np.array(mvd_vals).reshape(1, -1)
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"Minimum Viable Depth per Task ({args.threshold:.0%} threshold)", fontsize=14)
    for i, v in enumerate(mvd_vals):
        ax.text(i, 0, str(v), ha="center", va="center", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="MVD (layers)", shrink=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "minimum_depth_heatmap.pdf"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Save everything
    analysis = {
        "model": args.model_path,
        "n_layers": n_layers,
        "threshold": args.threshold,
        "depth_candidates": depth_candidates,
        "mvd_per_task": mvd_data,
        "scaling_law_fit": fit_result,
        "raw_depth_sweep": all_results,
    }
    with open(os.path.join(args.output_dir, "scaling_law_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info("Analysis complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
