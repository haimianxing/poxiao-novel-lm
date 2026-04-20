# Poxiao (破晓) — Long-Form Novel Language Model

A complete **pretrain → SFT → deployment** pipeline for training a long-form novel generation model from open-source Chinese novel data. Designed as a local, privacy-preserving creative writing productivity tool.

**破晓** (*pò xiǎo*) means "dawn" in Chinese — the moment before a new story begins.

---

## Features

| Component | Details |
|-----------|---------|
| **Architecture** | LLaMA-style decoder-only (RMSNorm + GQA + RoPE + SwiGLU) |
| **Scale** | 64M parameters (configurable), bf16 mixed precision |
| **Pretraining** | Multi-phase curriculum: short context (768) → long context (2048) with multi-task learning |
| **SFT** | Dialog-style fine-tuning with LoRA (rank 8, targeting QKV projections) |
| **MoE** | Optional Mixture-of-Experts (4 experts, top-1 routing) |
| **Distributed** | DDP training with gradient accumulation and early stopping |
| **Inference** | Interactive generation with auto-evaluation on novel continuation prompts |

---

## Quick Start

### Installation

```bash
pip install torch transformers datasets
```

### Pretrain

```python
from train import pretrain

# Phase A: short context (seq_len=768)
# Phase B/C: long context (seq_len=2048) with multi-task
python train.py
```

### SFT (Supervised Fine-Tuning)

```python
# Fine-tune with LoRA on dialog-formatted novel data
python train_sft.py
```

### Inference

```python
# Interactive novel generation
python inference.py

# Simple text generation
python generate.py --prompt "夜色笼罩着古城，" --max_tokens 512
```

---

## Architecture

```
PoxiaoForCausalLM
├── Embedding (6400 vocab, 768 dim, weight-tied)
├── Transformer Block × 12
│   ├── RMSNorm → Grouped Query Attention (RoPE)
│   ├── RMSNorm → SwiGLU FFN (or MoE: 4 experts, top-1)
│   └── Residual Connection
└── RMSNorm → Linear (vocab projection)
```

### Key Design Choices

- **Grouped Query Attention (GQA)**: Reduces KV-cache memory for long sequences
- **RoPE Positional Encoding**: Better length extrapolation for novel-length generation
- **SwiGLU Activation**: Superior to standard ReLU/GELU in language tasks
- **Weight Tying**: Shared embedding and LM head weights (parameter efficient)
- **LoRA Fine-Tuning**: Only QKV projections are adapted during SFT, keeping base model frozen

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────┐
│                  Pretraining Pipeline                │
│                                                     │
│  Phase A (seq=768)  →  Phase B (seq=2048)          │
│       ↓                      ↓                      │
│  Standard CLM Loss    CLM + Multi-Task Loss         │
│       ↓                      ↓                      │
│  LR: 1.5e-3 → 1e-5 (cosine decay)                  │
│  Early stop: patience=10, target PPL < 15           │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│                    SFT Pipeline                      │
│                                                      │
│  Dialog-formatted data  →  LoRA fine-tuning         │
│  Loss masking (only assistant tokens)                │
│  Rank=8, Alpha=16, Target=QKV                       │
└──────────────────────┬───────────────────────────────┘
                       ↓
              ┌────────────────┐
              │   Inference    │
              │  (generation)  │
              └────────────────┘
```

---

## Project Structure

```
poxiao-novel-lm/
├── config.py        # Model & training configuration (64M params)
├── model.py         # LLaMA-style architecture (RMSNorm, GQA, SwiGLU, MoE)
├── tokenizer.py     # BPE tokenizer wrapper (6400 vocab)
├── dataset.py       # Pretrain & SFT dataset loaders
├── train.py         # Pretraining with DDP, phased curriculum, early stopping
├── train_sft.py     # SFT with LoRA on dialog data
├── lora.py          # Pure PyTorch LoRA implementation
├── generate.py      # Simple text generation CLI
├── inference.py     # Interactive inference & auto-evaluation
└── README.md
```

---

## Configuration

Default configuration (64M parameters):

| Parameter | Value |
|-----------|-------|
| Vocab Size | 6,400 |
| Hidden Size | 768 |
| Num Layers | 12 |
| Num Attention Heads | 12 |
| Num KV Groups (GQA) | 4 |
| Intermediate Size | 1,536 |
| Max Sequence Length | 2,048 |
| Parameters | ~64M |

All hyperparameters are configurable in `config.py`.

---

## Roadmap

- [ ] Integrate with OpenClaw / Cursor for interactive novel writing workflows
- [ ] Local deployment with llama.cpp / ONNX Runtime
- [ ] Web UI for non-technical users
- [ ] Multi-chapter coherence via retrieval-augmented generation
- [ ] Larger model variants (200M, 500M)

---

## License

MIT License

---

# 破晓 (Poxiao) — 长篇小说语言模型

一个从开源中文小说数据到可部署的小说创作工具的**完整预训练 → SFT → 部署**流水线。

## 项目简介

**破晓**旨在提供一个端到端的长篇小说生成模型训练框架。从原始小说文本出发，经过预训练和多轮对话式微调，最终实现本地化部署的小说创作生产力工具。

### 核心特性

- **LLaMA架构**：RMSNorm + GQA + RoPE + SwiGLU，专为长文本生成优化
- **64M参数**：轻量级，可在消费级GPU上训练和推理
- **三阶段训练**：短上下文预训练 → 长上下文多任务 → LoRA对话微调
- **MoE可选**：支持4专家混合专家系统，提升模型容量
- **LoRA微调**：仅训练QKV投影层，显存友好
- **DDP分布式**：支持多卡训练，梯度累积和早停机制

### 训练流水线

```
原始小说数据
    ↓
Phase A: 短序列预训练 (seq=768, 标准CLM损失)
    ↓
Phase B/C: 长序列预训练 (seq=2048, CLM+多任务损失)
    ↓
LoRA SFT: 对话式微调 (仅assistant部分计算loss)
    ↓
推理部署: 交互式小说续写
```

### 快速开始

```bash
# 安装依赖
pip install torch transformers datasets

# 预训练
python train.py

# LoRA微调
python train_sft.py

# 交互式推理
python inference.py
```

### 后续规划

- 结合 OpenClaw / Cursor 等工具，实现交互式小说创作工作流
- llama.cpp / ONNX Runtime 本地部署
- 非技术用户的 Web UI 界面
- 基于检索增强生成的多章节连贯性
- 更大规模模型变体 (200M, 500M)

---

## Citation

```bibtex
@software{zhang2026poxiao,
  title={Poxiao: A Complete Pretrain-SFT Pipeline for Long-Form Novel Generation},
  author={Zhang, Chengzhe},
  year={2026},
  url={https://github.com/haimianxing/poxiao-novel-lm}
}
```
