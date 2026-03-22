# Minimum Viable Depth: How Many Transformer Layers Do LLMs Really Need?

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-layerdepth.git
cd nips-layerdepth

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-layerdepth_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

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
