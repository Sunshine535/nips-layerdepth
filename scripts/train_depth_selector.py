#!/usr/bin/env python3
"""
Train an adaptive depth selector via REINFORCE.

A small MLP observes the CLS embedding (first-token hidden state) and
predicts a per-layer keep/skip mask. The reward trades accuracy against
the number of layers used:

    reward = accuracy - λ * (layers_used / total_layers)

Training data: GSM8K + MMLU prompts evaluated through the frozen LLM.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.layer_surgery import get_decoder_layers, set_decoder_layers

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("depth_selector")


# ── Depth selector network ───────────────────────────────────────────────

class DepthSelector(nn.Module):
    """MLP: first-token embedding → per-layer keep probability."""

    def __init__(self, hidden_dim: int, n_layers: int, mlp_hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, n_layers),
        )
        self.n_layers = n_layers

    def forward(self, cls_embedding: torch.Tensor):
        """
        Args:
            cls_embedding: (B, hidden_dim)
        Returns:
            keep_probs: (B, n_layers) in (0, 1)
        """
        return torch.sigmoid(self.net(cls_embedding))

    def sample_mask(self, keep_probs: torch.Tensor):
        """Gumbel-sigmoid sampling for differentiable hard masks."""
        dist = torch.distributions.Bernoulli(probs=keep_probs)
        mask = dist.sample()
        log_probs = dist.log_prob(mask).sum(dim=-1)
        return mask, log_probs


# ── Evaluation helpers ───────────────────────────────────────────────────

def extract_answer_gsm8k(text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else ""


@torch.no_grad()
def get_cls_embedding(model, tokenizer, prompt: str):
    """Get the first-token hidden state from the full model (no generation)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    outputs = model(input_ids=inputs.input_ids, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    return last_hidden[:, 0, :]  # (1, hidden_dim)


@torch.no_grad()
def generate_with_mask(model, tokenizer, prompt: str, original_layers: list,
                       mask: torch.Tensor, max_new_tokens: int = 256) -> str:
    """Generate using only the layers where mask == 1."""
    kept_indices = mask.nonzero(as_tuple=True)[0].tolist()
    if not kept_indices:
        kept_indices = [0, len(original_layers) - 1]

    new_layers = torch.nn.ModuleList([original_layers[i] for i in kept_indices])
    set_decoder_layers(model, new_layers)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    set_decoder_layers(model, torch.nn.ModuleList(original_layers))
    return text


def check_gsm8k_answer(generated: str, gold_answer: str) -> float:
    pred = extract_answer_gsm8k(generated)
    try:
        return 1.0 if pred and gold_answer and abs(float(pred) - float(gold_answer)) < 1e-3 else 0.0
    except ValueError:
        return 1.0 if pred.strip() == gold_answer.strip() else 0.0


def check_mmlu_answer(generated: str, gold_letter: str) -> float:
    pred = generated.strip().upper()[:1]
    return 1.0 if pred == gold_letter else 0.0


# ── Data loading ─────────────────────────────────────────────────────────

def load_training_data(max_gsm8k: int = 300, max_mmlu: int = 300):
    """Build mixed training set of (prompt, gold, check_fn, task_name) tuples."""
    items = []

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    for ex in gsm8k.select(range(min(max_gsm8k, len(gsm8k)))):
        prompt = (f"Solve step by step. Put final answer after ####.\n\n"
                  f"Question: {ex['question']}\n\nAnswer:")
        gold = extract_answer_gsm8k(ex["answer"])
        items.append({"prompt": prompt, "gold": gold,
                      "check_fn": "gsm8k", "task": "gsm8k"})

    choices = ["A", "B", "C", "D"]
    try:
        mmlu = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        mmlu = load_dataset("cais/mmlu", "all", split="validation")
    for ex in mmlu.select(range(min(max_mmlu, len(mmlu)))):
        options = ex.get("choices", [ex.get(c, "") for c in choices])
        prompt = f"Question: {ex['question']}\n"
        for i, opt in enumerate(options if isinstance(options, list) else [options]):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer with just the letter (A, B, C, or D):\n"
        gold_idx = ex.get("answer", 0)
        gold = choices[gold_idx] if isinstance(gold_idx, int) and gold_idx < 4 else str(gold_idx).upper()[:1]
        items.append({"prompt": prompt, "gold": gold,
                      "check_fn": "mmlu", "task": "mmlu"})

    random.shuffle(items)
    return items


# ── REINFORCE training loop ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Adaptive Depth Selector")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--output_dir", type=str, default="./results/depth_selector")
    parser.add_argument("--train_gsm8k", type=int, default=300)
    parser.add_argument("--train_mmlu", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lam", type=float, default=0.3,
                        help="Lambda: penalty coefficient for layers used")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Held-out samples for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    hidden_dim = model.config.hidden_size
    logger.info("Model: %d layers, hidden_dim=%d", n_layers, hidden_dim)

    # Build depth selector
    selector = DepthSelector(hidden_dim, n_layers).to(model.device).float()
    optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr)

    # Load data and split train / eval
    all_items = load_training_data(args.train_gsm8k, args.train_mmlu)
    n_eval = min(args.eval_samples, len(all_items) // 5)
    eval_items = all_items[:n_eval]
    train_items = all_items[n_eval:]
    logger.info("Train: %d samples, Eval: %d samples", len(train_items), len(eval_items))

    check_fns = {
        "gsm8k": check_gsm8k_answer,
        "mmlu": check_mmlu_answer,
    }

    training_log = []
    best_eval_reward = -float("inf")

    for epoch in range(args.epochs):
        random.shuffle(train_items)
        selector.train()
        epoch_rewards = []
        epoch_layers = []
        epoch_correct = []

        for step, item in enumerate(tqdm(train_items, desc=f"Epoch {epoch + 1}")):
            cls_emb = get_cls_embedding(model, tokenizer, item["prompt"])  # (1, H)
            cls_emb_float = cls_emb.float()

            keep_probs = selector(cls_emb_float)  # (1, n_layers)
            mask, log_prob = selector.sample_mask(keep_probs)  # (1, n_layers), (1,)

            mask_1d = mask[0].long()
            n_used = mask_1d.sum().item()

            generated = generate_with_mask(
                model, tokenizer, item["prompt"], original_layers,
                mask_1d, args.max_new_tokens,
            )

            check_fn = check_fns[item["check_fn"]]
            accuracy = check_fn(generated, item["gold"])
            reward = accuracy - args.lam * (n_used / n_layers)

            # REINFORCE: minimize -reward * log_prob
            loss = -(reward * log_prob).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()

            epoch_rewards.append(reward)
            epoch_layers.append(n_used)
            epoch_correct.append(accuracy)

            if (step + 1) % 50 == 0:
                logger.info(
                    "  step %d: reward=%.3f, acc=%.1f%%, layers=%d/%d",
                    step + 1,
                    sum(epoch_rewards[-50:]) / 50,
                    100 * sum(epoch_correct[-50:]) / 50,
                    int(sum(epoch_layers[-50:]) / 50),
                    n_layers,
                )

        # Epoch summary
        mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        mean_acc = sum(epoch_correct) / len(epoch_correct)
        mean_layers = sum(epoch_layers) / len(epoch_layers)

        logger.info(
            "Epoch %d: reward=%.4f, acc=%.3f, avg_layers=%.1f/%d",
            epoch + 1, mean_reward, mean_acc, mean_layers, n_layers,
        )

        # Evaluation
        selector.eval()
        eval_rewards, eval_correct, eval_layers = [], [], []
        with torch.no_grad():
            for item in tqdm(eval_items, desc="Eval", leave=False):
                cls_emb = get_cls_embedding(model, tokenizer, item["prompt"]).float()
                keep_probs = selector(cls_emb)
                mask_1d = (keep_probs[0] > 0.5).long()
                n_used = mask_1d.sum().item()

                generated = generate_with_mask(
                    model, tokenizer, item["prompt"], original_layers,
                    mask_1d, args.max_new_tokens,
                )
                check_fn = check_fns[item["check_fn"]]
                acc = check_fn(generated, item["gold"])
                reward = acc - args.lam * (n_used / n_layers)

                eval_rewards.append(reward)
                eval_correct.append(acc)
                eval_layers.append(n_used)

        eval_mean_reward = sum(eval_rewards) / max(len(eval_rewards), 1)
        eval_mean_acc = sum(eval_correct) / max(len(eval_correct), 1)
        eval_mean_layers = sum(eval_layers) / max(len(eval_layers), 1)

        logger.info(
            "Eval: reward=%.4f, acc=%.3f, avg_layers=%.1f/%d",
            eval_mean_reward, eval_mean_acc, eval_mean_layers, n_layers,
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_reward": mean_reward,
            "train_accuracy": mean_acc,
            "train_avg_layers": mean_layers,
            "eval_reward": eval_mean_reward,
            "eval_accuracy": eval_mean_acc,
            "eval_avg_layers": eval_mean_layers,
        }
        training_log.append(epoch_record)

        if eval_mean_reward > best_eval_reward:
            best_eval_reward = eval_mean_reward
            ckpt_path = os.path.join(args.output_dir, "best_selector.pt")
            torch.save({
                "state_dict": selector.state_dict(),
                "config": {"hidden_dim": hidden_dim, "n_layers": n_layers},
                "epoch": epoch + 1,
                "eval_reward": eval_mean_reward,
            }, ckpt_path)
            logger.info("Saved best selector (reward=%.4f)", eval_mean_reward)

    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # Final summary
    summary = {
        "model": args.model_path,
        "n_layers": n_layers,
        "lambda": args.lam,
        "epochs": args.epochs,
        "train_samples": len(train_items),
        "eval_samples": len(eval_items),
        "best_eval_reward": best_eval_reward,
        "final_epoch": training_log[-1],
    }
    with open(os.path.join(args.output_dir, "depth_selector_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
