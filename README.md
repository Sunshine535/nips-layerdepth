# Minimum Viable Depth: How Many Transformer Layers Do LLMs Really Need?

**NeurIPS 2026 — `nips-layerdepth`**

## Abstract

Large language models run a fixed stack of transformer layers for every input, even when shallow computation would suffice. This work defines and measures **Minimum Viable Depth (MVD)**—the smallest number of layers (or layer subsets) needed to preserve benchmark-level performance—and connects MVD to task difficulty via empirical **depth scaling laws**. Using **Qwen/Qwen3.5-27B** (64 layers) as the primary subject, we run **single-layer knockout**, **contiguous block knockout**, **multi-metric importance ranking**, **depth scaling-law fitting**, and **adaptive depth selection** (lightweight router trained with REINFORCE-style signals). The full replication suite is designed for **≈3200 GPU-hours** on modern datacenter GPUs; scripts auto-detect **4–8× NVIDIA A100-class** nodes and set `NUM_GPUS` / `CUDA_VISIBLE_DEVICES` accordingly.

## Quick Start

```bash
git clone https://github.com/<your-org>/nips-layerdepth.git
cd nips-layerdepth
chmod +x setup.sh scripts/run_all_experiments.sh
./setup.sh

./scripts/run_all_experiments.sh
```

Set `HF_TOKEN` if gated models require authentication. Override cache with `HF_HOME` / `TRANSFORMERS_CACHE` as needed.

## Hardware

- **Recommended:** 4–8× **NVIDIA A100** (40GB or 80GB) on a single node; GPU count is **auto-detected** at runtime (`scripts/gpu_utils.sh` → `NUM_GPUS`, `CUDA_VISIBLE_DEVICES`).
- **Primary model:** `Qwen/Qwen3.5-27B` — plan for **multi-GPU** memory and throughput; smaller smoke configs can be added in `configs/` for debugging.
- **Budget:** approximately **3200 GPU-hours** for the full paper-scale pipeline (knockout sweeps + selector + analysis).

## Project Structure

```
nips-layerdepth/
├── README.md
├── setup.sh
├── requirements.txt
├── PROPOSAL.md / PLAN.md / PAPERS.md / EXPERIMENTS.md   # research docs
├── configs/          # YAML configs (e.g. knockout_config.yaml)
├── scripts/          # runnable pipeline + analysis entrypoints
├── src/              # library / modules
├── results/          # experiment outputs (generated)
└── logs/             # run logs (generated)
```

## Experiments Overview

| Track | Description |
|--------|-------------|
| Single-layer knockout | Ablate one layer at a time; measure GSM8K / MMLU (and extensions). |
| Block knockout | Contiguous blocks (e.g. sizes 2, 4, 8, 16) with resume support. |
| Importance ranking | Aggregate ranking from gradient / activation / Fisher-style proxies. |
| Depth scaling law | Fit MVD vs. complexity; thresholded recovery analysis (e.g. 0.95). |
| Adaptive depth selector | Train small router (REINFORCE); evaluate adaptive depth policies. |
| MVD + adaptive eval | Fit MVD curves and run adaptive-depth evaluation on held-out tasks. |

Orchestration: `scripts/run_all_experiments.sh` (calls `layer_knockout.py`, `run_block_knockout.py`, `run_importance_ranking.py`, `run_scaling_law_analysis.py`, `train_depth_selector.py`, `fit_mvd.py`, `eval_adaptive_depth.py`).

## Timeline (indicative)

| Phase | Focus |
|--------|--------|
| Months 1–2 | Knockout infrastructure, benchmarks, statistical harness |
| Months 3–4 | Block + importance metrics; scaling-law data collection |
| Months 5–6 | Adaptive selector training, MVD fitting, paper figures |
| Month 7 | Ablations, robustness, writing & internal review |

## BibTeX

```bibtex
@inproceedings{layerdepth2026,
  title     = {Minimum Viable Depth: How Many Transformer Layers Do {LLMs} Really Need?},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) if present in the repository root, or include the standard MIT text in your distribution.
