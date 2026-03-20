# Execution Plan: Minimum Viable Depth

## Overview

- **Duration**: 12 weeks (March 20 – June 12, 2026)
- **NeurIPS 2026 deadline**: Late May 2026
- **Hardware**: 8× A100-80GB
- **Primary model**: Qwen3.5-27B (64 layers)

## Phase 1: Infrastructure & Layer Knockout (Weeks 1–3)

### Week 1: Infrastructure Setup

- [ ] **Day 1–2**: Build layer knockout engine
  - Hook into Qwen3.5-27B forward pass
  - Support three modes: single-layer, contiguous, selected-subset
  - Layer removal = replace layer output with identity (skip residual)
  - Verify: removing 0 layers = original performance (sanity check)

- [ ] **Day 3–4**: Evaluation harness
  - Integrate lm-evaluation-harness for all 12 benchmark categories
  - Benchmarks: GSM8K, MATH-500, MMLU (57 sub), ARC-C, TriviaQA,
    BBH (27 sub), GPQA, HumanEval, MBPP, HellaSwag, WinoGrande, PIQA
  - Add timing instrumentation (per-layer latency, total latency)
  - Batch evaluation with result caching

- [ ] **Day 5**: Smoke test on Qwen3-8B
  - Single-layer knockout for all 36 layers
  - Quick eval on GSM8K + MMLU (5-shot, 100 samples)
  - Verify infrastructure correctness, estimate time budget

**Deliverable**: Working knockout + eval pipeline, smoke test results

### Week 2: Full Single-Layer Knockout

- [ ] **Day 1–3**: Qwen3.5-27B single-layer knockout sweep
  - 64 layers × 12 benchmarks = 768 evaluations
  - Parallelize across 8 GPUs (8 layers per GPU simultaneously)
  - Estimated time: ~48 hours with parallelization
  - Record: accuracy, perplexity, per-layer latency impact

- [ ] **Day 4–5**: Analysis and visualization
  - Layer importance heatmap (layer × benchmark)
  - Identify: critical layers, redundant layers, task-specific layers
  - Compute Block Influence (BI) score for comparison with ShortGPT
  - Statistical analysis: variance across random seeds (3 seeds)

**Deliverable**: Single-layer importance matrix, initial redundancy map

### Week 3: Contiguous & Selected Knockout

- [ ] **Day 1–2**: Contiguous knockout sweep
  - Start positions: {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60}
  - Block lengths: {2, 4, 8, 12, 16, 24}
  - Skip combinations where start + length > 64
  - Focus on 6 key benchmarks: GSM8K, MMLU, BBH, HumanEval, HellaSwag, GPQA
  - Estimated: ~300 configurations × 6 benchmarks

- [ ] **Day 3–4**: Selected-subset knockout (greedy elimination)
  - Algorithm: Start with all 64 layers, iteratively remove least important
  - Importance = average performance drop across all benchmarks
  - Record performance at each removal step → Pareto frontier
  - Run on 3 benchmark subsets: easy, medium, hard (self-defined clusters)

- [ ] **Day 5**: Cross-model validation
  - Repeat contiguous knockout on Qwen3.5-9B (40 layers)
  - Compare layer importance patterns with Qwen3.5-27B
  - Compute rank correlation of layer importance orderings

**Deliverable**: MVD measurements for all knockout types, cross-model comparison

## Phase 2: Task Complexity & Scaling Law (Weeks 4–5)

### Week 4: Task Complexity Measurement

- [ ] **Day 1–2**: Compute complexity features for all benchmark items
  - **Reasoning depth**: Run Qwen3.5-27B CoT, count reasoning steps
  - **Knowledge breadth**: NER on questions + answers, count unique entities
  - **Compositional complexity**: Parse question structure, count sub-questions
  - **Output entropy**: Compute answer variability from 10 sample completions

- [ ] **Day 3**: Complexity score calibration
  - Manual annotation of 500 items across difficulty spectrum
  - Fit feature weights w₁...w₄ to match human-judged difficulty
  - Validate on held-out 200 items (Spearman ρ > 0.7 target)

- [ ] **Day 4–5**: Task-level complexity aggregation
  - Aggregate item-level scores to task-level (mean, p90)
  - Create complexity ranking of all benchmark categories
  - Expected ordering: PIQA < HellaSwag < MMLU < GSM8K < BBH < GPQA < MATH

**Deliverable**: Calibrated complexity scores for all benchmarks and items

### Week 5: Scaling Law Fitting

- [ ] **Day 1–2**: MVD extraction from knockout data
  - For each benchmark × ε threshold, find minimum depth retaining performance
  - ε thresholds: {0.01, 0.02, 0.05, 0.10}
  - Result: MVD_ε(task) for all benchmarks at all thresholds

- [ ] **Day 2–3**: Scaling law fitting
  - Model 1: Power law — MVD = α · C^β + γ
  - Model 2: Logarithmic — MVD = α · log(C) + γ
  - Model 3: Piecewise linear — breakpoints at complexity thresholds
  - Fit via least squares, report R² and AIC for model selection
  - 5-fold cross-validation on benchmark splits

- [ ] **Day 4–5**: Cross-model transfer
  - Apply Qwen3.5-27B scaling law to predict Qwen3.5-9B MVD
  - Re-fit only constants (α, β, γ), keep functional form
  - Measure prediction error: |predicted MVD - actual MVD|
  - Do same for Qwen3-8B

**Deliverable**: Task-complexity → depth scaling law with cross-model validation

## Phase 3: Adaptive Depth Selector (Weeks 6–7)

### Week 6: Selector Design & Training

- [ ] **Day 1**: Generate training labels
  - For 50K samples from mixed benchmarks, compute per-sample MVD
  - Label = optimal exit layer (discretized to {8, 12, 16, 20, ..., 64})
  - Balance: ~equal samples per exit layer bin

- [ ] **Day 2–3**: Train selector network
  - Architecture: MLP(d_model → 512 → 256 → num_exit_points)
  - Attach at layer 4 of Qwen3.5-27B
  - Loss: Cross-entropy on exit layer prediction
  - Augmentation: Label smoothing (adjacent layers are nearly equivalent)
  - Training: 10K steps, lr=1e-3, batch_size=256
  - Validation: 5K held-out samples, accuracy + oracle-gap metric

- [ ] **Day 4–5**: Selector variants
  - Variant A: Attach at layer 2 (faster but less informed)
  - Variant B: Attach at layer 8 (more accurate but slower)
  - Variant C: Cascaded (check at layers 4, 8, 16 — multi-stage)
  - Compare accuracy–latency tradeoff across variants

**Deliverable**: Trained depth selectors (3 variants), accuracy reports

### Week 7: End-to-End Evaluation

- [ ] **Day 1–2**: Inference pipeline integration
  - Implement: Input → Layer 4 → Selector → Skip to predicted layer → Continue
  - Handle KV cache correctly for skipped layers
  - Batched inference with mixed exit layers (pad + mask strategy)

- [ ] **Day 3–4**: Speedup benchmarking
  - Test configurations:
    - Full model (64 layers) — baseline
    - Fixed exit at layer 48 (75%)
    - Fixed exit at layer 32 (50%)
    - Oracle MVD (per-input, look-up table)
    - Adaptive selector (variants A, B, C)
  - Metrics: tokens/sec, latency (p50/p95/p99), quality on all benchmarks
  - Batch sizes: {1, 4, 16, 64} on single A100

- [ ] **Day 5**: Statistical significance
  - 5 random seeds for selector training
  - Paired bootstrap test (10K resamples) for quality comparisons
  - Report confidence intervals for all speedup numbers

**Deliverable**: Full speedup × quality results table with significance

## Phase 4: Layer Stitching Adapters (Weeks 8–9)

### Week 8: Adapter Training

- [ ] **Day 1**: Identify top skip patterns
  - From Phase 3 results, find the 5 most common skip patterns
  - Example patterns: skip[20-40], skip[16-32], skip[24-48], etc.
  - Each pattern needs a dedicated stitching adapter

- [ ] **Day 2–3**: Train stitching adapters
  - Architecture: LoRA rank-4 at skip boundary (entry + exit layers)
  - Training data: 10K samples from RedPajama-v2 (calibration set)
  - Loss: MSE on hidden states — ||h_adapted - h_original||²
  - Training: 2000 steps, lr=5e-4 per adapter
  - Total: 5 patterns × 2K steps = ~10K training steps

- [ ] **Day 4–5**: Adapter effectiveness measurement
  - Compare with and without stitching adapters for each skip pattern
  - Metric: Quality recovery (% of quality gap closed)
  - Expected: 50–80% of quality gap recovered by adapters
  - Measure adapter overhead (parameter count, latency impact)

**Deliverable**: 5 trained stitching adapters, effectiveness measurements

### Week 9: Combined System & Ablations

- [ ] **Day 1–2**: Full pipeline assembly
  - Selector + skip + stitch, end-to-end
  - Selector routes input → skip pattern → matching adapter
  - Handle fallback: if selector uncertain, use full depth

- [ ] **Day 2–3**: Comprehensive ablation study
  - Ablation A: Selector only (no stitching) vs. full pipeline
  - Ablation B: Random selector vs. trained selector
  - Ablation C: Stitching adapter rank {2, 4, 8, 16}
  - Ablation D: Selector attachment layer {2, 4, 8, 16}
  - Ablation E: Training data size {1K, 5K, 10K, 50K}

- [ ] **Day 4–5**: Comparison with published baselines
  - Implement LayerSkip early-exit (or use their released code)
  - Implement ShortGPT layer removal (BI score method)
  - Implement CALM confidence-based exit
  - Fair comparison on identical benchmarks and hardware

**Deliverable**: Complete results table, ablation matrix, baseline comparisons

## Phase 5: Paper Writing (Weeks 10–12)

### Week 10: Draft Writing

- [ ] Introduction + related work (2 pages)
- [ ] Method section (3 pages): MVD, complexity, scaling law, selector, stitching
- [ ] Experimental setup (1 page): models, benchmarks, baselines, metrics
- [ ] Start results section: knockout results + scaling law

### Week 11: Results & Analysis

- [ ] Complete results section (2 pages): selector speedup, stitching, ablations
- [ ] Analysis section (1 page): what does the scaling law tell us about transformers?
- [ ] Create all figures: heatmaps, scaling curves, speedup plots, Pareto fronts
- [ ] Create all tables: main results, ablation, comparison

### Week 12: Polish & Submit

- [ ] Abstract and conclusion
- [ ] Appendix: full benchmark results, implementation details
- [ ] Internal review and revision
- [ ] Format check, reference check, anonymization
- [ ] Submit to NeurIPS 2026

## GPU Budget

| Phase | Duration | GPUs | GPU-Hours |
|---|---|---|---|
| Phase 1: Knockout | 3 weeks | 8 | ~1,200 |
| Phase 2: Complexity + Scaling | 2 weeks | 4 | ~300 |
| Phase 3: Selector | 2 weeks | 8 | ~800 |
| Phase 4: Stitching + Ablation | 2 weeks | 8 | ~800 |
| Phase 5: Paper | 3 weeks | 2 | ~100 |
| **Total** | **12 weeks** | — | **~3,200** |

## Risk Mitigation Checkpoints

| Week | Checkpoint | Go/No-Go Criteria |
|---|---|---|
| 2 | Knockout sanity | >20% of layers show <1% drop on at least 3 benchmarks |
| 4 | Complexity scores | Human correlation ρ > 0.6 |
| 5 | Scaling law fit | R² > 0.7 for at least one functional form |
| 7 | Selector speedup | >1.3× speedup with <3% quality loss |
| 9 | Full pipeline | Beats at least 2 baselines on speedup–quality Pareto |

## Key Files to Produce

| File | Purpose | Estimated Size |
|---|---|---|
| `src/knockout.py` | Layer knockout engine | ~500 lines |
| `src/complexity.py` | Task complexity scoring | ~300 lines |
| `src/fit_scaling_law.py` | Curve fitting + validation | ~400 lines |
| `src/train_selector.py` | Depth selector training | ~400 lines |
| `src/stitching.py` | Layer stitching adapters | ~300 lines |
| `src/model_surgery.py` | Model manipulation utilities | ~200 lines |
| `src/eval_harness.py` | Evaluation wrapper | ~300 lines |
