#!/usr/bin/env python3
"""
Fit MVD(task) = g(complexity) from knockout results.

Minimum Viable Depth (MVD) is the smallest number of layers
that achieves >= threshold fraction of full-model accuracy.
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fit_mvd")


def parse_args():
    parser = argparse.ArgumentParser(description="Fit MVD(task) relationship")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with knockout JSON results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Fraction of baseline accuracy to consider viable")
    parser.add_argument("--model_type", type=str, default="power_law",
                        choices=["linear", "power_law", "log"])
    parser.add_argument("--total_layers", type=int, default=64)
    return parser.parse_args()


def load_results(results_dir: str) -> dict:
    data = {}
    for fname in ["baseline.json", "prefix_knockout.json", "single_knockout.json",
                   "importance_knockout.json"]:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data[fname.replace(".json", "")] = json.load(f)
    return data


def compute_mvd_from_prefix(prefix_results: dict, baseline: dict,
                            threshold: float, total_layers: int) -> dict:
    """
    For each benchmark, find minimum k (prefix layers) where
    accuracy >= threshold * baseline_accuracy.
    """
    mvd_per_bench = {}

    benchmarks = set()
    for key, results in prefix_results.items():
        if key.startswith("prefix_"):
            benchmarks.update(results.keys())

    for bench in benchmarks:
        baseline_acc = baseline.get(bench, {}).get("accuracy",
                       baseline.get(bench, {}).get("pass_rate", 0))
        target = threshold * baseline_acc

        k_acc_pairs = []
        for key, results in sorted(prefix_results.items()):
            if not key.startswith("prefix_"):
                continue
            k = int(key.split("_")[1])
            acc = results.get(bench, {}).get("accuracy",
                  results.get(bench, {}).get("pass_rate", 0))
            k_acc_pairs.append((k, acc))

        k_acc_pairs.append((total_layers, baseline_acc))
        k_acc_pairs.sort()

        mvd = total_layers
        for k, acc in k_acc_pairs:
            if acc >= target:
                mvd = k
                break

        mvd_per_bench[bench] = {
            "mvd": mvd,
            "baseline_accuracy": baseline_acc,
            "target_accuracy": target,
            "k_accuracy_curve": k_acc_pairs,
        }

    return mvd_per_bench


def compute_complexity_proxy(baseline: dict) -> dict:
    """Use baseline accuracy as proxy for task complexity (lower acc = harder)."""
    complexity = {}
    for bench, results in baseline.items():
        acc = results.get("accuracy", results.get("pass_rate", 0))
        complexity[bench] = 1.0 - acc
    return complexity


def fit_linear(x, a, b):
    return a * x + b


def fit_power_law(x, a, b, c):
    return a * np.power(x + 1e-8, b) + c


def fit_log(x, a, b):
    return a * np.log(x + 1e-8) + b


def fit_mvd_complexity(mvd_dict: dict, complexity: dict,
                       model_type: str, total_layers: int) -> dict:
    """Fit MVD = g(complexity) and return fit parameters + predictions."""
    benchmarks = sorted(set(mvd_dict.keys()) & set(complexity.keys()))

    if len(benchmarks) < 2:
        logger.warning("Not enough benchmarks for fitting (need >= 2, got %d)", len(benchmarks))
        return {"error": "insufficient data", "benchmarks": benchmarks}

    x = np.array([complexity[b] for b in benchmarks])
    y = np.array([mvd_dict[b]["mvd"] / total_layers for b in benchmarks])

    try:
        if model_type == "linear":
            popt, pcov = curve_fit(fit_linear, x, y, maxfev=5000)
            y_pred = fit_linear(x, *popt)
            params = {"a": float(popt[0]), "b": float(popt[1])}
        elif model_type == "power_law":
            popt, pcov = curve_fit(fit_power_law, x, y, p0=[1.0, 0.5, 0.1], maxfev=5000)
            y_pred = fit_power_law(x, *popt)
            params = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
        elif model_type == "log":
            popt, pcov = curve_fit(fit_log, x, y, maxfev=5000)
            y_pred = fit_log(x, *popt)
            params = {"a": float(popt[0]), "b": float(popt[1])}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / max(ss_tot, 1e-8))

        return {
            "model_type": model_type,
            "params": params,
            "r_squared": float(r_squared),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "per_benchmark": {
                b: {
                    "complexity": float(x[i]),
                    "mvd_fraction": float(y[i]),
                    "mvd_layers": mvd_dict[b]["mvd"],
                    "predicted_fraction": float(y_pred[i]),
                    "predicted_layers": int(round(y_pred[i] * total_layers)),
                }
                for i, b in enumerate(benchmarks)
            },
        }
    except Exception as e:
        logger.error("Curve fitting failed: %s", e)
        return {
            "error": str(e),
            "raw_data": {
                b: {"complexity": float(complexity[b]),
                    "mvd": mvd_dict[b]["mvd"]}
                for b in benchmarks
            },
        }


def main():
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    data = load_results(args.results_dir)

    if "baseline" not in data:
        logger.error("baseline.json not found in %s", args.results_dir)
        return
    if "prefix_knockout" not in data:
        logger.error("prefix_knockout.json not found in %s", args.results_dir)
        return

    baseline = data["baseline"]
    prefix_results = data["prefix_knockout"]

    logger.info("Computing MVD per benchmark (threshold=%.2f)", args.threshold)
    mvd_dict = compute_mvd_from_prefix(
        prefix_results, baseline, args.threshold, args.total_layers
    )

    for bench, info in mvd_dict.items():
        logger.info("  %s: MVD = %d/%d layers (baseline_acc=%.3f)",
                    bench, info["mvd"], args.total_layers, info["baseline_accuracy"])

    complexity = compute_complexity_proxy(baseline)
    logger.info("Complexity proxies: %s",
                {k: f"{v:.3f}" for k, v in complexity.items()})

    logger.info("Fitting MVD = g(complexity) with model=%s", args.model_type)
    fit_result = fit_mvd_complexity(
        mvd_dict, complexity, args.model_type, args.total_layers
    )

    full_results = {
        "threshold": args.threshold,
        "total_layers": args.total_layers,
        "mvd_per_benchmark": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "k_accuracy_curve"}
                              for k, v in mvd_dict.items()},
        "complexity_proxy": complexity,
        "fit": fit_result,
        "accuracy_curves": {k: v["k_accuracy_curve"] for k, v in mvd_dict.items()},
    }

    out_path = os.path.join(output_dir, "mvd_analysis.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("MVD ANALYSIS RESULTS")
    logger.info("=" * 60)
    if "r_squared" in fit_result:
        logger.info("Fit R²: %.4f, RMSE: %.4f", fit_result["r_squared"], fit_result["rmse"])
        logger.info("Model: %s, Params: %s", fit_result["model_type"], fit_result["params"])
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
