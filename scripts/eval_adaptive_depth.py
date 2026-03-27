#!/usr/bin/env python3
"""
Evaluate adaptive depth selection: predict needed layers per query, use only those.

Uses MVD analysis to route queries to an appropriate depth (number of layers),
then evaluates accuracy vs. compute savings.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layer_surgery import get_decoder_layers, set_decoder_layers
from src.model_utils import load_model_and_tokenizer, get_model_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("adaptive_depth")


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Depth Evaluation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--mvd_results", type=str, required=True,
                        help="Path to mvd_analysis.json from fit_mvd.py")
    parser.add_argument("--output_dir", type=str, default="./results/adaptive_depth")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "mmlu"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--depth_candidates", nargs="+", type=int, default=None,
                        help="Layer counts to evaluate (default: auto from MVD)")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                        help="Confidence threshold for early exit")
    return parser.parse_args()


def extract_answer(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


@torch.no_grad()
def generate_with_depth(model, tokenizer, prompt, original_layers, depth,
                        max_new_tokens=512):
    """Generate using only the first `depth` layers."""
    new_layers = torch.nn.ModuleList(original_layers[:depth])
    set_decoder_layers(model, new_layers)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(get_model_device(model))
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=True,
    )
    gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    if outputs.scores:
        all_probs = [torch.softmax(s, dim=-1).max().item() for s in outputs.scores]
        confidence = float(np.mean(all_probs)) if all_probs else 0.5
    else:
        confidence = 0.5

    set_decoder_layers(model, torch.nn.ModuleList(original_layers))
    return text, confidence


def estimate_query_complexity(tokenizer, prompt: str) -> float:
    """Heuristic complexity estimate based on prompt features."""
    tokens = tokenizer.encode(prompt)
    num_tokens = len(tokens)
    num_numbers = len(re.findall(r'\d+', prompt))
    num_sentences = max(1, len(re.split(r'[.!?]', prompt)))

    complexity = (
        0.3 * min(num_tokens / 200.0, 1.0) +
        0.4 * min(num_numbers / 10.0, 1.0) +
        0.3 * min(num_sentences / 5.0, 1.0)
    )
    return min(complexity, 1.0)


def predict_depth(complexity: float, mvd_fit: dict, total_layers: int,
                  depth_candidates: list) -> int:
    """Predict the needed depth given query complexity."""
    if "params" in mvd_fit:
        params = mvd_fit["params"]
        model_type = mvd_fit.get("model_type", "linear")
        if model_type == "linear":
            frac = params["a"] * complexity + params["b"]
        elif model_type == "power_law":
            frac = params["a"] * (complexity ** params["b"]) + params["c"]
        elif model_type == "log":
            frac = params["a"] * np.log(complexity + 1e-8) + params["b"]
        else:
            frac = 0.8
        frac = np.clip(frac, 0.1, 1.0)
        predicted_layers = int(round(frac * total_layers))
    else:
        predicted_layers = total_layers

    best = min(depth_candidates, key=lambda d: abs(d - predicted_layers))
    return best


def adaptive_generate(model, tokenizer, prompt, original_layers,
                      depth_candidates, confidence_threshold, max_new_tokens=512):
    """Try generating at increasing depths until confidence is high enough."""
    sorted_depths = sorted(depth_candidates)

    for depth in sorted_depths:
        text, confidence = generate_with_depth(
            model, tokenizer, prompt, original_layers, depth, max_new_tokens
        )
        if confidence >= confidence_threshold:
            return text, depth, confidence

    text, confidence = generate_with_depth(
        model, tokenizer, prompt, original_layers, sorted_depths[-1], max_new_tokens
    )
    return text, sorted_depths[-1], confidence


def load_benchmark(benchmark: str, num_samples: int):
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if num_samples > 0:
            ds = ds.select(range(min(num_samples, len(ds))))
        items = []
        for ex in ds:
            prompt = f"Solve step by step. Put final answer after ####.\n\nQuestion: {ex['question']}\n\nAnswer:"
            gold = extract_answer(ex["answer"])
            items.append({"prompt": prompt, "gold": gold})
        return items
    elif benchmark == "math":
        ds = load_dataset("hendrycks/competition_math", split="test")
        if num_samples > 0:
            ds = ds.select(range(min(num_samples, len(ds))))
        items = []
        for ex in ds:
            prompt = f"Solve. Put answer in \\boxed{{}}.\n\nProblem: {ex['problem']}\n\nSolution:"
            gold = extract_answer(ex["solution"])
            items.append({"prompt": prompt, "gold": gold})
        return items
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.mvd_results) as f:
        mvd_data = json.load(f)

    total_layers = mvd_data.get("total_layers", 64)

    if args.depth_candidates:
        depth_candidates = args.depth_candidates
    else:
        depth_candidates = [8, 16, 24, 32, 40, 48, 56, total_layers]

    logger.info("Loading model: %s", args.model_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    original_layers = list(get_decoder_layers(model))
    logger.info("Model has %d layers. Depth candidates: %s", len(original_layers), depth_candidates)

    items = load_benchmark(args.benchmark, args.num_samples)
    logger.info("Evaluating %d samples from %s", len(items), args.benchmark)

    mvd_fit = mvd_data.get("fit", {})

    results_predicted = {"correct": 0, "total": 0, "layers_used": [], "details": []}
    results_adaptive = {"correct": 0, "total": 0, "layers_used": [], "details": []}
    results_full = {"correct": 0, "total": 0, "details": []}

    for item in tqdm(items, desc="Adaptive evaluation"):
        prompt = item["prompt"]
        gold = item["gold"]

        complexity = estimate_query_complexity(tokenizer, prompt)
        predicted_depth = predict_depth(complexity, mvd_fit, total_layers, depth_candidates)

        text_pred, conf_pred = generate_with_depth(
            model, tokenizer, prompt, original_layers, predicted_depth
        )
        pred_ans = extract_answer(text_pred)
        try:
            pred_correct = abs(float(pred_ans) - float(gold)) < 1e-3 if pred_ans and gold else False
        except ValueError:
            pred_correct = pred_ans.strip() == gold.strip()
        results_predicted["correct"] += int(pred_correct)
        results_predicted["total"] += 1
        results_predicted["layers_used"].append(predicted_depth)

        text_adap, used_depth, conf_adap = adaptive_generate(
            model, tokenizer, prompt, original_layers,
            depth_candidates, args.confidence_threshold
        )
        adap_ans = extract_answer(text_adap)
        try:
            adap_correct = abs(float(adap_ans) - float(gold)) < 1e-3 if adap_ans and gold else False
        except ValueError:
            adap_correct = adap_ans.strip() == gold.strip()
        results_adaptive["correct"] += int(adap_correct)
        results_adaptive["total"] += 1
        results_adaptive["layers_used"].append(used_depth)

        text_full, _ = generate_with_depth(
            model, tokenizer, prompt, original_layers, total_layers
        )
        full_ans = extract_answer(text_full)
        try:
            full_correct = abs(float(full_ans) - float(gold)) < 1e-3 if full_ans and gold else False
        except ValueError:
            full_correct = full_ans.strip() == gold.strip()
        results_full["correct"] += int(full_correct)
        results_full["total"] += 1

    def summarize(r, name, total_layers):
        acc = r["correct"] / max(r["total"], 1)
        if "layers_used" in r and r["layers_used"]:
            avg_layers = np.mean(r["layers_used"])
            compute_saving = 1.0 - (avg_layers / total_layers)
        else:
            avg_layers = total_layers
            compute_saving = 0.0
        return {
            "method": name,
            "accuracy": acc,
            "avg_layers": float(avg_layers),
            "compute_saving": float(compute_saving),
            "total": r["total"],
        }

    summary = {
        "full_model": summarize(results_full, "full_model", total_layers),
        "predicted_depth": summarize(results_predicted, "predicted_depth", total_layers),
        "adaptive_depth": summarize(results_adaptive, "adaptive_depth", total_layers),
    }

    out_path = os.path.join(args.output_dir, f"adaptive_{args.benchmark}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("ADAPTIVE DEPTH RESULTS (%s)", args.benchmark)
    logger.info("=" * 60)
    for method, s in summary.items():
        logger.info("  %s: acc=%.3f, avg_layers=%.1f, saving=%.1f%%",
                    method, s["accuracy"], s["avg_layers"], s["compute_saving"] * 100)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
