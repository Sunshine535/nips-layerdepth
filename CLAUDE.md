# Project: nips-layerdepth

## Project goal

Minimum Viable Depth: How Many Transformer Layers Do LLMs Really Need? — 通过层/块 knockout、重要性排序研究最小有效深度（MVD）与 scaling law，训练 REINFORCE depth selector 实现自适应深度推理。

## Key models

- `Qwen/Qwen3.5-27B` — 主实验模型

## Key datasets

- GSM8K — 数学推理
- MMLU — 知识评测

## Repo map

- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排（Block A/B）
  - `layer_knockout.py` — 单层 knockout
  - `run_block_knockout.py` — 块 knockout
  - `run_importance_ranking.py` — 层重要性排序
  - `run_scaling_law_analysis.py` — Scaling law 分析
  - `train_depth_selector.py` — REINFORCE depth selector 训练
  - `fit_mvd.py` — MVD 拟合
  - `eval_adaptive_depth.py` — 自适应深度评测
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
- `configs/` — 配置（含 `knockout_config.yaml`）
- `results/` — 实验输出

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# 一键全流程
bash run.sh

# 后台运行
nohup bash run.sh > run.log 2>&1 &

# 强制重跑
FORCE_RERUN=1 bash run.sh
```

## Experiment phases

| Block | Step | 内容 |
|-------|------|------|
| A (并行) | 1 | 层 knockout |
| A (并行) | 2 | 块 knockout |
| A (并行) | 3 | 重要性排序 |
| A (并行) | 4 | Scaling law 分析 |
| B (并行) | 5 | Depth selector 训练 |
| B (并行) | 6 | MVD 拟合 + 自适应深度评测 |

## Data and outputs

- 层 knockout: `results/layer_knockout/`
- 块 knockout: `results/block_knockout/`
- 重要性排序: `results/importance/`
- Scaling law: `results/scaling_law/`
- Depth selector: `results/depth_selector/`
- MVD 分析: `results/mvd_analysis/`
- 日志: `logs/`

## Environment

- Python 3.12（优先）或 3.11/3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, datasets, accelerate, scikit-learn
- 可选: flash-attn, flash-linear-attention, causal-conv1d
- 不使用 wandb
- 每任务默认分配 2 GPU（`CUDA_VISIBLE_DEVICES=$(gpu_pair ...)`）

## Project-specific rules

- Block A 的 4 个 step 并行执行
- Block B 的 2 个 step 并行执行，但依赖 Block A 完成
- 使用 `gpu_pair` 函数为每个任务分配 2 张 GPU

## Remote server

<!-- TODO: 请补充此项目的主服务器信息 -->

- SSH: `ssh YOUR_SERVER`
- GPU: 待确认
- Activate: `source .venv/bin/activate`
- Code dir: 待确认
- Background: `screen -dmS layerdepth bash -c '...'`
