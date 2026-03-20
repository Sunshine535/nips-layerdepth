# Research Proposal: Minimum Viable Depth

## 1. Problem Statement

Modern LLMs process every input through all transformer layers regardless of difficulty.
A 64-layer model uses identical compute for "What is 2+2?" and "Prove the Riemann
hypothesis." This wastes 30–70% of inference compute on easy inputs.

**Research Question**: For a given task complexity level, what is the minimum number
of transformer layers (Minimum Viable Depth, MVD) required to maintain performance
within ε of the full model? Can we derive a closed-form scaling law?

## 2. Hypothesis

**H1 (Depth Redundancy)**: For any task category with bounded complexity C, there
exists an MVD significantly less than full depth L such that removing layers beyond
MVD causes < ε degradation.

**H2 (Scaling Law)**: MVD follows a power-law relationship with task complexity:
`MVD(C) = α · C^β + γ`, where α, β, γ are architecture-dependent constants.

**H3 (Composability)**: The MVD scaling law learned on one model transfers across
model sizes within the same architecture family (with re-fitted constants).

## 3. Core Idea

### 3.1 Minimum Viable Depth (MVD)

Define MVD_ε(task, model) as the smallest k such that a model using only k of its L
layers achieves performance within ε of the full model on the target task.

Formally: MVD_ε(t, M) = min{k : |P(M_k, t) - P(M_L, t)| ≤ ε}

where M_k is the model truncated/pruned to k effective layers.

### 3.2 Task Complexity Score

We propose a multi-dimensional complexity metric C(task) combining:
- **Reasoning depth**: Number of logical steps required (measured via chain-of-thought)
- **Knowledge breadth**: Diversity of facts needed (measured via entity count)
- **Compositional complexity**: Degree of sub-problem nesting
- **Output entropy**: Variability of correct answers

Each dimension is normalized to [0,1] and combined: C = w₁·R + w₂·K + w₃·Comp + w₄·H

### 3.3 Layer Knockout Protocol

Three knockout strategies to measure MVD:

**Single-layer knockout**: Remove layer i, measure performance. Identifies critical
vs. redundant individual layers.

**Contiguous knockout**: Remove layers [i, i+k). Measures how many consecutive layers
can be dropped from each position (early/middle/late).

**Selected-subset knockout**: Remove an optimized subset of layers. Uses greedy
search: iteratively remove the layer with least impact until performance drops
below threshold.

### 3.4 Adaptive Depth Selector

A lightweight classifier (2-layer MLP on hidden states at layer 4) that predicts
the optimal exit layer per input:

- Input: hidden state h_4 (after 4 initial layers)
- Output: predicted optimal layer k ∈ {8, 12, 16, ..., L}
- Training: supervised on MVD labels from knockout experiments
- Overhead: < 0.1% additional parameters

### 3.5 Layer Stitching Adapters

When skipping layers [i, j), the representation shifts. Small residual adapters
(rank-4 LoRA) bridge the gap:

- Placed at skip boundaries
- Trained to minimize ||h_j^adapted - h_j^original|| on calibration data
- Shared across inputs, specific to skip pattern

## 4. Theoretical Grounding

### 4.1 Connection to Log-Depth Circuits

Merrill & Sabharwal (2023) proved transformers can simulate log-depth circuits.
This implies that for bounded-complexity tasks, O(log n) layers suffice in theory.
Our work empirically measures the gap between theoretical minimum and practical MVD.

### 4.2 Curvature-Weighted Capacity

We connect MVD to the Minimum Description Length (MDL) framework. Layers with low
curvature in the loss landscape contribute less task-relevant information. The MVD
corresponds to the subset of layers carrying high-curvature capacity.

### 4.3 Residual Stream Perspective

In the residual stream view, each layer adds a perturbation to the representation.
Layers with small perturbation norm ||Δh_i|| / ||h_i|| are candidates for removal.
We validate that perturbation norm correlates with knockout impact (r > 0.7).

## 5. Experimental Design

### 5.1 Phase 1: Layer Knockout Survey (Weeks 1–3)

**Experiment 1.1**: Full single-layer knockout on Qwen3.5-27B (64 layers × 12 benchmarks)
- Metric: Per-layer importance score = average performance drop across benchmarks
- Expected: ~20% of layers have near-zero importance on most tasks

**Experiment 1.2**: Contiguous knockout sweep on Qwen3.5-27B
- Sweep: Start position ∈ {4, 8, 12, ..., 60}, length ∈ {2, 4, 8, 16}
- Metric: Performance vs. number of removed layers, by position
- Expected: Middle layers more removable than early/late

**Experiment 1.3**: Selected-subset knockout (greedy elimination)
- Algorithm: Iteratively remove least-important layer, re-evaluate
- Metric: Pareto frontier of (layers removed, performance retained)
- Expected: 25–40% of layers removable with <2% quality loss

### 5.2 Phase 2: Task Complexity & Scaling Law (Weeks 3–5)

**Experiment 2.1**: Compute complexity scores for all benchmark items
- Use Qwen3.5-27B chain-of-thought to measure reasoning depth
- Count unique entities for knowledge breadth
- Manual annotation of 500 items for calibration

**Experiment 2.2**: Fit MVD scaling law
- X-axis: Task complexity C(task)
- Y-axis: Measured MVD_ε at ε = {0.01, 0.02, 0.05}
- Fit: Power law, logarithmic, and piecewise linear models
- Validation: Hold-out tasks and cross-model transfer

### 5.3 Phase 3: Adaptive Selector (Weeks 5–7)

**Experiment 3.1**: Train depth selector on Qwen3.5-27B
- Training data: 50K samples with MVD labels from Phase 1
- Architecture: MLP(d_model → 256 → num_exit_points)
- Attached at layer 4 (early enough for speedup, late enough for signal)

**Experiment 3.2**: End-to-end inference benchmark
- Compare: Full model, fixed exit, oracle MVD, adaptive selector
- Metrics: Throughput (tokens/sec), latency (ms/query), quality retention
- Hardware: Single A100, batch sizes {1, 4, 16, 64}

### 5.4 Phase 4: Layer Stitching (Weeks 7–9)

**Experiment 4.1**: Train stitching adapters for top-5 skip patterns
- Calibration data: 10K samples from RedPajama
- Adapter: LoRA rank-4 at skip boundary layers
- Training: 1000 steps, MSE loss on hidden states

**Experiment 4.2**: Combined system evaluation
- Full pipeline: Input → Selector → Skip + Stitch → Output
- Comparison with LayerSkip, FlexiDepth, standard early exit
- Statistical significance: 5 random seeds, paired bootstrap

## 6. Baselines

| Method | Description | Source |
|---|---|---|
| Full model | All 64 layers | — |
| Fixed early exit | Exit at layer k for all inputs | Standard |
| LayerSkip | Self-speculative decoding with early exit | Meta, 2024 |
| FlexiDepth | Skip 8/32 layers with adapter | 2025 |
| ShortGPT | Layer removal by BI score | 2024 |
| Random knockout | Random layer subset | Ablation |

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| MVD varies too much per-instance | Medium | Use distribution over MVD, not point estimate |
| Scaling law doesn't fit cleanly | Medium | Try piecewise models, cluster tasks first |
| Selector overhead negates savings | Low | Selector is tiny (<0.1% params), attached early |
| Stitching adapters don't help | Medium | Fall back to contiguous-only skipping |
| Results don't transfer across models | Medium | Test on 3 models; report per-model constants |

## 8. Expected Contributions

1. First systematic MVD measurement across 12 benchmark categories
2. Empirical task-complexity → depth scaling law with theoretical motivation
3. Practical adaptive depth system with demonstrated speedup
4. Layer stitching technique for non-contiguous depth reduction
5. Open-source toolkit for depth analysis of any transformer model

## 9. Novelty Argument

LayerSkip uses a trained early-exit head; we derive *when* to exit from task complexity.
FlexiDepth skips a fixed set of layers; we adapt the skip set per-input.
ShortGPT removes layers permanently; we dynamically select layers at inference time.
None derive a task → depth scaling law or provide a theoretical framework connecting
transformer depth to task complexity.

## 10. Timeline

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | Knockout infrastructure | Single-layer knockout results for Qwen3.5-27B |
| 3 | Full knockout sweep | Contiguous + selected knockout results |
| 3–4 | Complexity measurement | Task complexity scores, initial scaling law fit |
| 5 | Scaling law validation | Cross-model transfer results |
| 5–6 | Selector training | Trained adaptive selector |
| 7 | Selector evaluation | End-to-end speedup benchmarks |
| 7–8 | Stitching adapters | Trained adapters, combined system |
| 9 | Full evaluation | All baselines, ablations, significance tests |
| 10–11 | Paper writing | Full NeurIPS draft |
| 12 | Revision | Camera-ready |
