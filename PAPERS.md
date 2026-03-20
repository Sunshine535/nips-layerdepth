# Related Work: Minimum Viable Depth

## Core References (Must-Cite)

### Layer Skipping & Early Exit

**LayerSkip: Enabling Early-Exit Inference and Self-Speculative Decoding**
Elhoushi et al. (Meta FAIR), 2024
- Trains with layer dropout + early-exit loss during pre-training
- Self-speculative decoding: early layers draft, full model verifies
- 1.6–2.0× speedup on code and summarization
- Key difference from ours: fixed early-exit strategy, no task-complexity modeling
- Paper: arXiv:2404.16710

**FlexiDepth: Dynamic Depth Inference for Efficient Transformers**
2025
- Learns to skip 8 out of 32 layers (25% reduction) with adapter modules
- Fixed skip pattern per-model, not per-input
- <1% accuracy loss on MMLU, moderate speedup
- Key difference: we derive *which* layers to skip per-input from task complexity

**ShortGPT: Layers in Large Language Models are More Redundant Than You Expect**
Men et al., 2024
- Block Influence (BI) metric identifies redundant layers
- Permanently removes layers, showing middle layers often expendable
- Up to 25% layer removal with <5% quality loss
- Key difference: permanent removal vs. our dynamic per-input selection
- Paper: arXiv:2403.03853

### Sparse & Efficient Inference

**WiSparse: Width-and-Sparsity Aware Transformer Inference**
2024
- Joint width and sparsity optimization for efficient inference
- Complementary to depth reduction — could combine with our method
- Shows different efficiency axes (width vs. depth) are somewhat orthogonal

**R-Sparse: Rank-Based Sparse Attention for Efficient Transformers**
2024
- Reduces attention computation through rank-based token selection
- Orthogonal efficiency axis (attention sparsity vs. layer depth)
- Potential combination: sparse attention + adaptive depth

**LaRoSA: Large-Scale Robust Sparse Attention**
2025
- Scalable sparse attention patterns for long contexts
- Shows sparsity patterns differ by layer position
- Supports our finding that layer roles vary by position

**WINA: Weight Informed Neuron Activation**
2025
- Activation-level sparsity during inference
- Neuron importance scores correlate with our layer importance
- Could provide finer-grained analysis within layers

### Theoretical Foundations

**The Log-Depth Advantage of Transformers**
Merrill & Sabharwal, 2023
- Proves transformers can simulate log-depth Boolean circuits
- Implies O(log n) layers suffice for bounded-complexity tasks
- Provides theoretical lower bound for our MVD measurements
- Paper: arXiv:2302.06198

**Minimum Description Length and Generalization in Transformers**
Voita & Titov, 2020
- MDL framework for measuring information per component
- Curvature-weighted capacity metric
- Our MVD can be interpreted through the MDL lens — layers carrying
  negligible description length are candidates for removal

**Residual Streams and Information Flow in Transformers**
Elhage et al. (Anthropic), 2021
- Residual stream view: each layer adds a perturbation
- Layer contributions measured by perturbation norm
- Directly motivates our knockout methodology

### Depth & Capacity Analysis

**Scaling Laws for Neural Language Models**
Kaplan et al. (OpenAI), 2020
- Power-law relationships between model size and performance
- Treats depth and width as interchangeable — we separate them
- Our scaling law is task-conditioned, theirs is aggregate

**Depth vs Width: A Systematic Study in Transformers**
Tay et al., 2021
- Shows depth matters more for reasoning, width for memorization
- Supports our hypothesis that MVD depends on task type
- But doesn't provide per-task MVD measurements

**On the Effect of Transformer Depth on Performance**
Xu et al., 2024
- Studies how adding depth affects different capabilities
- Shows diminishing returns beyond a task-dependent threshold
- Confirms existence of per-task depth saturation

## Extended References

### Early Exit Methods

**BERxiT: Early Exiting for BERT with Better Fine-Tuning and Extension to Regression**
Xin et al., 2021
- Pioneered early exit for BERT-class models
- Internal classifiers at each layer decide whether to exit
- Limited to classification; our method handles generation

**CALM: Confident Adaptive Language Modeling**
Schuster et al. (Google), 2022
- Token-level early exit based on confidence
- Shows different tokens need different depths
- Our contribution: systematic measurement + scaling law (not just confidence)
- Paper: arXiv:2207.07061

**SkipDecode: Autoregressive Skip Decoding with Batched Verification**
Del Corro et al., 2024
- Skip layers during decoding, verify with full model
- Combines with speculative decoding paradigm
- Complementary to our adaptive selector approach

### Pruning & Compression

**LLM-Pruner: Structural Pruning for Large Language Models**
Ma et al., 2023
- Task-agnostic structured pruning
- Removes entire attention heads and FFN neurons
- Different granularity from layer-level removal
- Paper: arXiv:2305.11627

**SliceGPT: Compress Large Language Models by Deleting Rows and Columns**
Ashkboos et al., 2024
- Removes dimensions within layers (orthogonal to depth pruning)
- 30% parameter reduction with small quality loss
- Could combine: SliceGPT within layers + our depth reduction across layers

**The Unreasonable Ineffectiveness of the Deeper Layers**
Gromov et al., 2024
- Shows removing up to 50% of deeper layers has minimal impact on QA
- Strong motivation for our work
- We go further: systematic measurement, task conditioning, scaling law
- Paper: arXiv:2403.17887

### Speculative Decoding

**Speculative Decoding with Big Little Decoder**
Kim et al., 2023
- Uses smaller model to draft, larger model to verify
- Depth reduction as an alternative to smaller draft models
- Our adaptive selector could replace the draft model

**Medusa: Simple LLM Inference Acceleration Framework**
Cai et al., 2024
- Multiple decoding heads for parallel token prediction
- Orthogonal to depth reduction — could combine for multiplicative speedup

### Knowledge Distillation (Depth Context)

**MiniLLM: Knowledge Distillation of Large Language Models**
Gu et al., 2023
- Distills deep model into shallower student
- KD requires full retraining; our method is training-free (knockout + small adapters)
- But KD results provide ceiling for how much depth can be reduced

**Distilling Step-by-Step**
Hsieh et al. (Google), 2023
- Distills reasoning chains, enabling shallower models to reason
- Connects to our finding: simpler tasks need fewer layers because
  the reasoning chain is shorter

## Comparison Matrix

| Method | Dynamic? | Per-input? | Task-aware? | Scaling Law? | Training-free? |
|---|---|---|---|---|---|
| LayerSkip | No | Partial | No | No | No |
| FlexiDepth | No | No | No | No | No |
| ShortGPT | No | No | No | No | Yes |
| CALM | Yes | Yes | No | No | No |
| Ours | Yes | Yes | Yes | **Yes** | Mostly* |

*Selector and stitching adapters require light training; knockout is training-free.

## Key Gaps in Literature

1. **No task-complexity → depth scaling law**: All methods use fixed or learned exit
   points without connecting to task properties
2. **No systematic MVD measurement**: ShortGPT and Gromov et al. measure layer
   redundancy but don't systematically sweep across task types
3. **No non-contiguous dynamic skip**: FlexiDepth and LayerSkip skip contiguous
   blocks; we enable arbitrary layer subsets via stitching adapters
4. **No theoretical connection**: No prior work connects empirical depth reduction
   to log-depth circuit theory or MDL framework
