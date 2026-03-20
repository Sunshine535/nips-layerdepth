#!/usr/bin/env python3
"""
Block knockout: remove contiguous blocks of layers and evaluate.

Slides a window of size {2, 4, 8, 16} across all 64 layers of Qwen3.5-27B,
measures GSM8K and MMLU accuracy for each position to reveal which
contiguous regions are critical.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layer_surgery import get_decoder_layers, set_decoder_layers

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("block_knockout")


def parse_args():
    parser = argparse.ArgumentParser(description="Contiguous Block Knockout")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--output_dir", type=str, default="./results/block_knockout")
    parser.add_argument("--block_sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed block positions")
    return parser.parse_args()


# ── Evaluation helpers (kept consistent with layer_knockout.py) ──────────

def extract_answer_gsm8k(text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def eval_gsm8k(model, tokenizer, max_samples: int, max_new_tokens: int) -> dict:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    correct, total = 0, 0
    for ex in tqdm(dataset, desc="GSM8K", leave=False):
        prompt = (f"Solve step by step. Put final answer after ####.\n\n"
                  f"Question: {ex['question']}\n\nAnswer:")
        text = generate_answer(model, tokenizer, prompt, max_new_tokens)
        pred = extract_answer_gsm8k(text)
        gold = extract_answer_gsm8k(ex["answer"])
        try:
            correct += int(abs(float(pred) - float(gold)) < 1e-3) if pred and gold else 0
        except ValueError:
            correct += int(pred.strip() == gold.strip())
        total += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_mmlu(model, tokenizer, max_samples: int, _max_new_tokens: int) -> dict:
    try:
        dataset = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        dataset = load_dataset("cais/mmlu", "all", split="validation")
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    choices = ["A", "B", "C", "D"]
    correct, total = 0, 0
    for ex in tqdm(dataset, desc="MMLU", leave=False):
        question = ex["question"]
        options = ex.get("choices", [ex.get(c, "") for c in choices])
        prompt = f"Question: {question}\n"
        for i, opt in enumerate(options if isinstance(options, list) else [options]):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer with just the letter (A, B, C, or D):\n"

        answer = generate_answer(model, tokenizer, prompt, max_new_tokens=16)
        pred = answer.strip().upper()[:1]
        gold_idx = ex.get("answer", 0)
        gold = choices[gold_idx] if isinstance(gold_idx, int) and gold_idx < 4 else str(gold_idx).strip().upper()[:1]
        correct += int(pred == gold)
        total += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


BENCH_FNS = {"gsm8k": eval_gsm8k, "mmlu": eval_mmlu}


# ── Main experiment ──────────────────────────────────────────────────────

def run_block_knockout(model, tokenizer, original_layers, block_size, n_layers,
                       benchmarks, max_samples, max_new_tokens, output_dir, resume):
    """Slide a block of `block_size` layers across all positions and evaluate."""
    block_results = {}
    out_path = os.path.join(output_dir, f"block_size_{block_size}.json")

    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            block_results = json.load(f)
        logger.info("Resumed %d existing positions for block_size=%d",
                     len(block_results), block_size)

    num_positions = n_layers - block_size + 1
    for start in range(num_positions):
        key = f"start_{start}_end_{start + block_size - 1}"
        if resume and key in block_results:
            continue

        removed = list(range(start, start + block_size))
        kept = [i for i in range(n_layers) if i not in removed]

        logger.info("Block [%d, %d] (size %d) — keeping %d/%d layers",
                     start, start + block_size - 1, block_size, len(kept), n_layers)

        new_layers = torch.nn.ModuleList([original_layers[i] for i in kept])
        set_decoder_layers(model, new_layers)

        results = {}
        for bench in benchmarks:
            if bench in BENCH_FNS:
                results[bench] = BENCH_FNS[bench](model, tokenizer, max_samples, max_new_tokens)
                logger.info("  %s: %.4f", bench, results[bench]["accuracy"])

        block_results[key] = {
            "start": start,
            "end": start + block_size - 1,
            "removed_layers": removed,
            "n_kept": len(kept),
            "results": results,
        }

        set_decoder_layers(model, torch.nn.ModuleList(original_layers))

        with open(out_path, "w") as f:
            json.dump(block_results, f, indent=2)

    return block_results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading model: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    original_layers = list(get_decoder_layers(model))
    n_layers = len(original_layers)
    logger.info("Model loaded: %d layers", n_layers)

    # Baseline
    baseline_path = os.path.join(args.output_dir, "baseline.json")
    if args.resume and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        logger.info("Loaded cached baseline: %s", baseline)
    else:
        logger.info("Computing baseline (full model)...")
        baseline = {}
        for bench in args.benchmarks:
            if bench in BENCH_FNS:
                baseline[bench] = BENCH_FNS[bench](model, tokenizer, args.max_samples, args.max_new_tokens)
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info("Baseline: %s", json.dumps(baseline, indent=2))

    for block_size in args.block_sizes:
        if block_size >= n_layers:
            logger.warning("block_size=%d >= n_layers=%d, skipping", block_size, n_layers)
            continue
        logger.info("=== Block size %d ===", block_size)
        run_block_knockout(
            model, tokenizer, original_layers, block_size, n_layers,
            args.benchmarks, args.max_samples, args.max_new_tokens,
            args.output_dir, args.resume,
        )

    # Summary across all block sizes
    summary = {"baseline": baseline, "block_sizes": {}}
    for bs in args.block_sizes:
        path = os.path.join(args.output_dir, f"block_size_{bs}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            accs = {}
            for bench in args.benchmarks:
                bench_accs = [v["results"].get(bench, {}).get("accuracy", 0)
                              for v in data.values()]
                if bench_accs:
                    accs[bench] = {
                        "min": min(bench_accs),
                        "max": max(bench_accs),
                        "mean": sum(bench_accs) / len(bench_accs),
                    }
            summary["block_sizes"][str(bs)] = {
                "n_positions": len(data),
                "accuracy_stats": accs,
            }

    summary_path = os.path.join(args.output_dir, "block_knockout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
