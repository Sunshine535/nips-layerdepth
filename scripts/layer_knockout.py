#!/usr/bin/env python3
"""
Layer knockout experiments on Qwen3.5-27B (64 layers).

Systematically removes layers and evaluates on MMLU/GSM8K/MATH/HumanEval.
Modes: single-layer removal, contiguous prefix, importance-based selection.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layer_surgery import (
    compute_layer_importance,
    get_decoder_layers,
    importance_based_removal,
    keep_prefix_layers,
    remove_layers,
    set_decoder_layers,
)
from src.model_utils import load_model_and_tokenizer, get_model_device

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layer_knockout")


def parse_args():
    parser = argparse.ArgumentParser(description="Layer Knockout Experiments")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        choices=["single", "prefix", "importance", "all"])
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt"
        )
    def __len__(self):
        return self.encodings.input_ids.shape[0]
    def __getitem__(self, idx):
        return {"input_ids": self.encodings.input_ids[idx]}


def extract_answer_gsm8k(text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def extract_answer_math(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    return extract_answer_gsm8k(text)


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(get_model_device(model))
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def eval_mmlu(model, tokenizer, cfg, max_samples=500) -> dict:
    logger.info("Evaluating MMLU")
    bench_cfg = cfg["benchmarks"]["mmlu"]
    try:
        dataset = load_dataset(bench_cfg["name"], bench_cfg["config"], split=bench_cfg["split"])
    except Exception:
        dataset = load_dataset("cais/mmlu", "all", split="test")

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    choices = ["A", "B", "C", "D"]
    correct, total = 0, 0

    for example in tqdm(dataset, desc="MMLU"):
        question = example["question"]
        options = example.get("choices", ["", "", "", ""])
        if "choices" not in example:
            options = [example.get(c, "") for c in ["A", "B", "C", "D"]]

        prompt = f"Question: {question}\n"
        for i, opt in enumerate(options if isinstance(options, list) else [options]):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer with just the letter (A, B, C, or D):\n"

        answer = generate_answer(model, tokenizer, prompt, max_new_tokens=16)
        pred = answer.strip().upper()[:1]

        gold_idx = example.get("answer", 0)
        if isinstance(gold_idx, int):
            gold = choices[gold_idx] if gold_idx < 4 else "A"
        else:
            gold = str(gold_idx).strip().upper()[:1]

        correct += int(pred == gold)
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_gsm8k(model, tokenizer, cfg, max_samples=500) -> dict:
    logger.info("Evaluating GSM8K")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct, total = 0, 0
    for example in tqdm(dataset, desc="GSM8K"):
        prompt = (
            f"Solve step by step. Put final answer after ####.\n\n"
            f"Question: {example['question']}\n\nAnswer:"
        )
        text = generate_answer(model, tokenizer, prompt)
        pred = extract_answer_gsm8k(text)
        gold = extract_answer_gsm8k(example["answer"])
        try:
            correct += int(abs(float(pred) - float(gold)) < 1e-3) if pred and gold else 0
        except ValueError:
            correct += int(pred.strip() == gold.strip())
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_math_bench(model, tokenizer, cfg, max_samples=500) -> dict:
    logger.info("Evaluating MATH")
    dataset = load_dataset("hendrycks/competition_math", split="test")
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct, total = 0, 0
    for example in tqdm(dataset, desc="MATH"):
        prompt = (
            f"Solve the problem. Put final answer in \\boxed{{}}.\n\n"
            f"Problem: {example['problem']}\n\nSolution:"
        )
        text = generate_answer(model, tokenizer, prompt)
        pred = extract_answer_math(text)
        gold = extract_answer_math(example["solution"])
        try:
            correct += int(abs(float(pred) - float(gold)) < 1e-3) if pred and gold else 0
        except ValueError:
            correct += int(pred.strip().lower() == gold.strip().lower())
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_humaneval(model, tokenizer, cfg, max_samples=-1) -> dict:
    logger.info("Evaluating HumanEval")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    completed, total = 0, 0
    for example in tqdm(dataset, desc="HumanEval"):
        prompt = example["prompt"]
        completion = generate_answer(model, tokenizer, prompt, max_new_tokens=512)

        full_code = prompt + completion
        has_return = "return " in completion
        no_syntax_error = True
        try:
            compile(full_code, "<string>", "exec")
        except SyntaxError:
            no_syntax_error = False

        completed += int(has_return and no_syntax_error)
        total += 1

    return {"pass_rate": completed / max(total, 1), "completed": completed, "total": total}


BENCH_FNS = {
    "mmlu": eval_mmlu,
    "gsm8k": eval_gsm8k,
    "math": eval_math_bench,
    "humaneval": eval_humaneval,
}


def evaluate_model(model, tokenizer, cfg, benchmarks):
    results = {}
    for bench in benchmarks:
        if bench in BENCH_FNS:
            max_s = cfg["benchmarks"].get(bench, {}).get("max_samples", 500)
            results[bench] = BENCH_FNS[bench](model, tokenizer, cfg, max_samples=max_s)
            logger.info("%s: %s", bench, json.dumps(results[bench]))
    return results


def run_single_layer_knockout(model, tokenizer, cfg, benchmarks, output_dir):
    """Remove one layer at a time and evaluate."""
    logger.info("=== Single Layer Knockout ===")
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    original_layers = list(layers)
    all_results = {}

    for layer_idx in range(n_layers):
        logger.info("Removing layer %d/%d", layer_idx, n_layers - 1)
        keep = [i for i in range(n_layers) if i != layer_idx]
        new_layers = torch.nn.ModuleList([original_layers[i] for i in keep])
        set_decoder_layers(model, new_layers)

        results = evaluate_model(model, tokenizer, cfg, benchmarks)
        all_results[f"remove_{layer_idx}"] = results

        set_decoder_layers(model, torch.nn.ModuleList(original_layers))

    out_path = os.path.join(output_dir, "single_knockout.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Single knockout results saved to %s", out_path)
    return all_results


def run_prefix_knockout(model, tokenizer, cfg, benchmarks, output_dir):
    """Keep only first k layers and evaluate."""
    logger.info("=== Prefix Knockout ===")
    layers = get_decoder_layers(model)
    original_layers = list(layers)
    n_layers = len(original_layers)

    k_values = cfg["knockout"]["prefix"]["k_values"]
    all_results = {}

    for k in k_values:
        if k >= n_layers:
            continue
        logger.info("Keeping first %d/%d layers", k, n_layers)
        new_layers = torch.nn.ModuleList(original_layers[:k])
        set_decoder_layers(model, new_layers)

        results = evaluate_model(model, tokenizer, cfg, benchmarks)
        all_results[f"prefix_{k}"] = results

        set_decoder_layers(model, torch.nn.ModuleList(original_layers))

    out_path = os.path.join(output_dir, "prefix_knockout.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Prefix knockout results saved to %s", out_path)
    return all_results


def run_importance_knockout(model, tokenizer, cfg, benchmarks, output_dir):
    """Remove layers by importance score."""
    logger.info("=== Importance-Based Knockout ===")
    layers = get_decoder_layers(model)
    original_layers = list(layers)
    n_layers = len(original_layers)

    imp_cfg = cfg["knockout"]["importance"]
    metric = imp_cfg["metric"]
    cal_samples = imp_cfg["calibration_samples"]
    fractions = imp_cfg["removal_fractions"]

    logger.info("Computing layer importance (metric=%s, samples=%d)", metric, cal_samples)
    cal_dataset = load_dataset("openai/gsm8k", "main", split="train")
    cal_texts = [ex["question"] for ex in cal_dataset.select(range(min(cal_samples, len(cal_dataset))))]
    cal_ds = SimpleTextDataset(cal_texts, tokenizer, max_length=256)
    cal_loader = DataLoader(cal_ds, batch_size=4, shuffle=False)

    importance = compute_layer_importance(model, cal_loader, metric, str(get_model_device(model)))
    importance_path = os.path.join(output_dir, "layer_importance.json")
    with open(importance_path, "w") as f:
        json.dump({"metric": metric, "importance": importance}, f, indent=2)

    all_results = {"importance_scores": importance}

    for frac in fractions:
        n_remove = max(1, int(n_layers * frac))
        sorted_idx = sorted(range(n_layers), key=lambda i: importance[i])
        to_remove = sorted_idx[:n_remove]
        keep = [i for i in range(n_layers) if i not in to_remove]

        logger.info("Removing %.0f%% (%d layers): %s", frac * 100, n_remove, sorted(to_remove))
        new_layers = torch.nn.ModuleList([original_layers[i] for i in keep])
        set_decoder_layers(model, new_layers)

        results = evaluate_model(model, tokenizer, cfg, benchmarks)
        all_results[f"remove_{frac:.1f}"] = {
            "removed_layers": sorted(to_remove),
            "kept_layers": keep,
            "results": results,
        }

        set_decoder_layers(model, torch.nn.ModuleList(original_layers))

    out_path = os.path.join(output_dir, "importance_knockout.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Importance knockout results saved to %s", out_path)
    return all_results


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    output_dir = args.output_dir or cfg["output"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model: %s", cfg["model"]["name_or_path"])
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name_or_path"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
    )

    n_layers = len(get_decoder_layers(model))
    logger.info("Model loaded with %d decoder layers", n_layers)

    benchmarks = args.benchmarks or list(cfg["benchmarks"].keys())
    modes = [args.mode] if args.mode and args.mode != "all" else cfg["knockout"]["modes"]

    logger.info("Running modes: %s on benchmarks: %s", modes, benchmarks)

    logger.info("Computing baseline (full model)")
    baseline = evaluate_model(model, tokenizer, cfg, benchmarks)
    with open(os.path.join(output_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Baseline: %s", json.dumps(baseline, indent=2))

    if "single" in modes:
        run_single_layer_knockout(model, tokenizer, cfg, benchmarks, output_dir)
    if "prefix" in modes:
        run_prefix_knockout(model, tokenizer, cfg, benchmarks, output_dir)
    if "importance" in modes:
        run_importance_knockout(model, tokenizer, cfg, benchmarks, output_dir)

    logger.info("All knockout experiments complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
