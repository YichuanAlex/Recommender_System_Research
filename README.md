# Recommender System Research - 推荐系统研究

<div align="center">

**Advanced Recommender Systems with Multi-Modal Feature Fusion and Generative Retrieval | 多模态特征融合与生成式检索的先进推荐系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Field-Recommend%20System-red.svg)]()

**Author | 作者**: Zixi Jiang (YichuanAlex)  
**Email | 邮箱**: jiangzixi1527435659@gmail.com  
**Location | 地点**: Shanghai, Shanghai, China  
**Last Updated | 最后更新**: 2026-03-30

</div>

---

## Table of Contents | 目录

- [Overview | 项目概述](#overview--项目概述)
- [Research Background | 研究背景](#research-background--研究背景)
- [Research Objectives | 研究目标](#research-objectives--研究目标)
- [Key Innovations | 核心创新点](#key-innovations--核心创新点)
- [Methodology | 方法论](#methodology--方法论)
- [System Architecture | 系统架构](#system-architecture--系统架构)
- [Project Structure | 项目结构](#project-structure--项目结构)
- [Installation and Setup | 安装与配置](#installation-and-setup--安装与配置)
- [Usage Guide | 使用指南](#usage-guide--使用指南)
- [Model Details | 模型详情](#model-details--模型详情)
- [Technical Features | 技术特性](#technical-features--技术特性)
- [Citation | 引用建议](#citation--引用建议)
- [License | 许可证](#license--许可证)
- [Contact | 联系方式](#contact--联系方式)

---

## Overview | 项目概述

**English:**  
This repository presents a comprehensive collection of advanced recommender system implementations and research experiments. The project focuses on multi-modal feature fusion, generative retrieval, and sequence modeling techniques for large-scale industrial recommendation scenarios. Key research directions include:

1. **Multi-Modal Feature Integration**: Fusing sparse features, dense embeddings, array features, and continuous features
2. **Generative Retrieval**: Implementing RQ-VAE (Residual Quantized Variational Autoencoder) for semantic ID generation
3. **Sequence Modeling**: Using Transformer-based architectures with Flash Attention for user behavior modeling
4. **GAN-based Approaches**: Exploring generative adversarial networks for feature representation learning
5. **Large-Scale Deployment**: Optimized inference pipelines with FAISS-based approximate nearest neighbor search

**中文:**  
本仓库展示了先进推荐系统实现和研究实验的综合集合。项目专注于大规模工业推荐场景中的多模态特征融合、生成式检索和序列建模技术。主要研究方向包括：

1. **多模态特征融合**: 融合稀疏特征、稠密 embedding、数组特征和连续特征
2. **生成式检索**: 实现 RQ-VAE（残差量化变分自编码器）用于语义 ID 生成
3. **序列建模**: 使用基于 Transformer 的架构和 Flash Attention 进行用户行为建模
4. **GAN 方法**: 探索生成对抗网络用于特征表示学习
5. **大规模部署**: 基于 FAISS 的近似最近邻搜索优化推理流程

---

## Research Background | 研究背景

**English:**  
Modern recommender systems face several critical challenges:

1. **Feature Heterogeneity**: User and item features come in various formats (sparse IDs, dense vectors, sequences, images, text)
2. **Cold Start Problem**: New users and items lack historical interaction data
3. **Scalability**: Industrial systems need to handle millions of users and items with low latency
4. **Representation Learning**: Effectively capturing complex patterns in user behavior sequences
5. **Multi-Modal Integration**: Leveraging visual and textual information alongside traditional features

This research addresses these challenges through a unified framework that combines:
- Deep learning-based feature representation
- Attention mechanisms for sequence modeling
- Generative models for semantic ID learning
- Efficient retrieval algorithms

**中文:**  
现代推荐系统面临几个关键挑战：

1. **特征异构性**: 用户和物品特征以各种格式存在（稀疏 ID、稠密向量、序列、图像、文本）
2. **冷启动问题**: 新用户和物品缺乏历史交互数据
3. **可扩展性**: 工业系统需要处理数百万用户和物品，同时保持低延迟
4. **表示学习**: 有效捕捉用户行为序列中的复杂模式
5. **多模态融合**: 在传统特征之外利用视觉和文本信息

本研究通过统一框架解决这些挑战，结合了：
- 基于深度学习的特征表示
- 用于序列建模的注意力机制
- 用于语义 ID 学习的生成模型
- 高效检索算法

---

## Research Objectives | 研究目标

**English:**
1. Develop a flexible and extensible recommendation framework supporting multiple model architectures
2. Implement efficient multi-modal feature fusion strategies
3. Explore generative retrieval methods using VQ-VAE and RQ-VAE
4. Design scalable inference pipelines with approximate nearest neighbor search
5. Investigate GAN-based approaches for feature representation learning
6. Build robust training and evaluation pipelines for industrial-scale datasets

**中文:**
1. 开发支持多种模型架构的灵活可扩展推荐框架
2. 实现高效的多模态特征融合策略
3. 探索使用 VQ-VAE 和 RQ-VAE 的生成式检索方法
4. 设计具有近似最近邻搜索的可扩展推理流程
5. 研究基于 GAN 的特征表示学习方法
6. 为工业级数据集构建稳健的训练和评估流程

---

## Key Innovations | 核心创新点

**English:**

### 1. Multi-Modal Feature Fusion Architecture
- Unified handling of sparse features, array features, continuous features, and multi-modal embeddings
- N-ary encoding for numerical features using hybrid encoding schemes
- Cross-network architectures for explicit feature interaction

### 2. RQ-VAE for Semantic ID Generation
- Residual quantization for multi-level semantic abstraction
- Codebook learning with KMeans/Balanced KMeans initialization
- End-to-end training with reconstruction and commitment losses

### 3. Flash Attention Implementation
- Optimized multi-head attention using PyTorch 2.0 Flash Attention
- Memory-efficient sequence modeling for long user behavior histories
- Support for causal masking in autoregressive settings

### 4. GAN-Based Feature Quantization
- Generator-Discriminator architecture for embedding quantization
- Gumbel-Softmax trick for differentiable discrete output
- Adversarial training for improved representation quality

### 5. Efficient Inference Pipeline
- FAISS-based approximate nearest neighbor search
- Binary embedding storage for memory efficiency
- Batch processing for high-throughput inference

**中文:**

### 1. 多模态特征融合架构
- 统一处理稀疏特征、数组特征、连续特征和多模态 embedding
- 使用混合编码方案的数值特征 N 元编码
- 用于显式特征交互的交叉网络架构

### 2. 用于语义 ID 生成的 RQ-VAE
- 用于多级语义抽象的残差量化
- 使用 KMeans/Balanced KMeans 初始化的码本学习
- 具有重建和承诺损失的端到端训练

### 3. Flash Attention 实现
- 使用 PyTorch 2.0 Flash Attention 优化多头注意力
- 针对长用户行为历史的内存高效序列建模
- 支持自回归设置中的因果掩码

### 4. 基于 GAN 的特征量化
- 用于 embedding 量化的生成器 - 判别器架构
- 用于可微分离散输出的 Gumbel-Softmax 技巧
- 用于改进表示质量的对抗训练

### 5. 高效推理流程
- 基于 FAISS 的近似最近邻搜索
- 用于内存效率的二进制 embedding 存储
- 用于高吞吐量推理的批处理

---

## Methodology | 方法论

### 1. Feature Representation | 特征表示

**English:**
- **Sparse Features**: Categorical features embedded via lookup tables
  - User features: gender, age_group, city_level, etc.
  - Item features: category, brand, price_range, etc.
  - Dimension: 16-128 per feature

- **Array Features**: Variable-length sequences with positional encoding
  - User behavior sequences (click, like, share, etc.)
  - Max length: 100, padded/truncated with positional encoding
  - Attention-based aggregation

- **Continuous Features**: Normalized numerical values
  - Direct projection or N-ary hybrid encoding
  - Batch normalization for stability

- **Multi-Modal Features**: Pre-computed embeddings
  - Image embeddings (ViT, ResNet): 3584-4096 dimensions
  - Text embeddings (BERT, etc.): 1024 dimensions
  - Fused via RQ-VAE or linear projection

**中文:**
- **稀疏特征**: 通过查找表嵌入的类别特征
  - 用户特征：性别、年龄段、城市等级等
  - 物品特征：类别、品牌、价格区间等
  - 维度：每个特征 16-128

- **数组特征**: 具有位置编码的可变长度序列
  - 用户行为序列（点击、点赞、分享等）
  - 最大长度：100，填充/截断并带位置编码
  - 基于注意力的聚合

- **连续特征**: 归一化的数值
  - 直接投影或 N 元混合编码
  - 批归一化保证稳定性

- **多模态特征**: 预计算的 embedding
  - 图像 embedding（ViT、ResNet）：3584-4096 维
  - 文本 embedding（BERT 等）：1024 维
  - 通过 RQ-VAE 或线性投影融合

### 2. Model Architecture | 模型架构

**English:**
```
User/Item Features → Feature Fusion → Sequence Modeling → Prediction Head
     ↓                    ↓                  ↓                  ↓
Sparse/Array/Cont   Concat/Attention   Transformer/Cross   Dot Product/MLP
Multi-Modal         Fusion             Flash Attention     Sigmoid/Softmax
```

**Base Model Components:**
1. **Embedding Layer**: Feature-specific embedding tables
2. **Fusion Layer**: Concatenation, attention, or cross-network
3. **Sequence Encoder**: Multi-block Transformer with layer normalization
4. **Prediction Head**: Point-wise scoring or list-wise ranking

**中文:**
```
用户/物品特征 → 特征融合 → 序列建模 → 预测头
     ↓            ↓           ↓          ↓
稀疏/数组/连续  拼接/注意力  Transformer/交叉  点积/MLP
多模态          融合         Flash Attention  Sigmoid/Softmax
```

**基础模型组件:**
1. **嵌入层**: 特定特征的嵌入表
2. **融合层**: 拼接、注意力或交叉网络
3. **序列编码器**: 多层 Transformer 带层归一化
4. **预测头**: 点对点评分或列表排序

### 3. RQ-VAE Implementation | RQ-VAE 实现

**English:**
- **Encoder**: Maps multi-modal embeddings to latent space
- **Residual Quantizer**: Multi-level VQ with shared/independent codebooks
- **Decoder**: Reconstructs embeddings from quantized representations
- **Loss Function**: 
  - Reconstruction loss (MSE)
  - Codebook loss (commitment)
  - Perplexity regularization

**中文:**
- **编码器**: 将多模态 embedding 映射到潜在空间
- **残差量化器**: 多级 VQ，支持共享/独立码本
- **解码器**: 从量化表示重建 embedding
- **损失函数**: 
  - 重建损失（MSE）
  - 码本损失（承诺）
  - 困惑度正则化

### 4. Training Strategy | 训练策略

**English:**
- **Loss Functions**:
  - Binary Cross-Entropy for point-wise ranking
  - InfoNCE loss for contrastive learning
  - Pairwise hinge loss for list-wise ranking

- **Optimization**:
  - AdamW optimizer with weight decay
  - Learning rate scheduling with warmup
  - Gradient clipping for stability

- **Regularization**:
  - Dropout (0.1-0.3)
  - L2 regularization on embeddings
  - Label smoothing

**中文:**
- **损失函数**:
  - 点对点排序的二元交叉熵
  - 对比学习的 InfoNCE 损失
  - 列表排序的成对铰链损失

- **优化**:
  - 带有权重衰减的 AdamW 优化器
  - 带预热的学习率调度
  - 梯度裁剪保证稳定性

- **正则化**:
  - Dropout（0.1-0.3）
  - Embedding 的 L2 正则化
  - 标签平滑

---

## System Architecture | 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Input Layer                        │
│                     数据输入层                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  User/Item Feature Data (Sparse, Array, Continuous) │    │
│  │  用户/物品特征数据（稀疏、数组、连续）              │    │
│  │  Multi-Modal Embeddings (Image, Text)               │    │
│  │  多模态 Embedding（图像、文本）                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Feature Processing Layer                   │
│                   特征处理层                                │
│  • Feature Statistics Computation    特征统计计算           │
│  • Missing Value Imputation          缺失值填充             │
│  • Multi-Modal Embedding Loading     多模态 Embedding 加载  │
│  • N-ary Encoding for Numerical      数值特征 N 元编码      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Representation Learning Layer                 │
│                 表示学习层                                  │
│  • Embedding Tables                    嵌入表               │
│  • RQ-VAE / GAN Quantization           RQ-VAE/GAN 量化      │
│  • Feature Fusion (Concat/Attention)   特征融合             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Sequence Modeling Layer                    │
│                   序列建模层                                │
│  • Flash Multi-Head Attention        Flash 多头注意力       │
│  • Transformer Blocks                Transformer 块         │
│  • Cross Network                     交叉网络               │
│  • Layer Normalization & Dropout     层归一化和 Dropout     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Prediction Layer                        │
│                     预测层                                  │
│  • Point-wise Scoring                点对点评分             │
│  • Pairwise Ranking                  成对排序               │
│  • InfoNCE Contrastive Loss          InfoNCE 对比损失       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Inference & Retrieval                      │
│                   推理与检索                                │
│  • Item Embedding Generation         物品 Embedding 生成    │
│  • FAISS Approximate Nearest Neighbor FAISS 近似最近邻     │
│  • Binary Format Storage             二进制格式存储         │
│  • Batch Processing                  批处理                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure | 项目结构

```
rec/
│
├── README.md                              # Project documentation | 项目文档
│
├── GAN+/                                  # GAN-based quantization | GAN 量化
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # Main model | 主模型
│   ├── model_rqvae.py                     # RQ-VAE/GAN quantizer | RQ-VAE/GAN 量化器
│   ├── dataset.py                         # Data loading | 数据加载
│   ├── infer.py                           # Inference script | 推理脚本
│   └── record.md                          # Development notes | 开发笔记
│
├── GAN0/                                  # Baseline GAN implementation | 基础 GAN
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # Main model | 主模型
│   ├── model_rqvae.py                     # Quantizer module | 量化器模块
│   ├── dataset.py                         # Data loading | 数据加载
│   └── infer.py                           # Inference script | 推理脚本
│
├── HSTU+/                                 # Hierarchical Sequential Transfer Unit | 分层序列迁移单元
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # HSTU model | HSTU 模型
│   ├── dataset.py                         # Data loading | 数据加载
│   ├── infer.py                           # Inference script | 推理脚本
│   ├── run.sh                             # Run script | 运行脚本
│   └── 2402.17152v3.pdf                   # Reference paper | 参考论文
│
├── RQVAE/                                 # Residual Quantized VAE | 残差量化 VAE
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # RQ-VAE model | RQ-VAE 模型
│   ├── model_rqvae.py                     # RQ-VAE implementation | RQ-VAE 实现
│   ├── dataset.py                         # Data loading | 数据加载
│   ├── infer.py                           # Inference script | 推理脚本
│   ├── run.sh                             # Run script | 运行脚本
│   └── NeurIPS-2023-...pdf                # Reference paper | 参考论文
│
├── TencentGR_1k_不知名副本/                # Tencent Generative Retrieval | 腾讯生成式检索
│   ├── README.md                          # Project guide | 项目指南
│   ├── requirements.txt                   # Dependencies | 依赖
│   ├── main.py                            # Training main | 训练主流程
│   ├── model.py                           # Model definition | 模型定义
│   ├── dataset.py                         # Dataset loader | 数据集加载
│   ├── infer.py                           # Inference | 推理
│   ├── run_inference.py                   # Inference runner | 推理运行器
│   ├── start_training.py                  # Training starter | 训练启动器
│   ├── run.sh                             # Run script | 运行脚本
│   ├── TencentGR_1k/                      # Data directory | 数据目录
│   ├── checkpoints/                       # Model checkpoints | 模型检查点
│   ├── inference_results/                 # Inference output | 推理输出
│   ├── logs/                              # Training logs | 训练日志
│   └── tf_events/                         # TensorBoard events | TensorBoard 事件
│
├── emb 数据测试/                            # Embedding data testing | Embedding 数据测试
│   ├── main.py                            # Test script | 测试脚本
│   ├── model.py                           # Test model | 测试模型
│   ├── model_rqvae.py                     # RQ-VAE module | RQ-VAE 模块
│   └── dataset.py                         # Test dataset | 测试数据集
│
├── on the emb/                            # Embedding research | Embedding 研究
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # SFG Model | SFG 模型
│   ├── dataset.py                         # Data loading | 数据加载
│   ├── infer.py                           # Inference script | 推理脚本
│   └── *.pdf                              # Research papers | 研究论文
│
├── origin/                                # Original baseline | 原始基线
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # Baseline model | 基线模型
│   ├── model_rqvae.py                     # RQ-VAE module | RQ-VAE 模块
│   ├── dataset.py                         # Data loading | 数据加载
│   ├── infer.py                           # Inference script | 推理脚本
│   ├── requirements.txt                   # Dependencies | 依赖
│   └── run.sh                             # Run script | 运行脚本
│
├── 官网脚本/                               # Official website scripts | 官网脚本
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # Model definition | 模型定义
│   ├── dataset.py                         # Dataset loader | 数据集加载
│   └── infer.py                           # Inference script | 推理脚本
│
├── 数据字段渗透测试/                        # Feature field ablation | 特征字段消融
│   ├── main.py                            # Training script | 训练脚本
│   ├── model.py                           # Model definition | 模型定义
│   ├── model_rqvae.py                     # RQ-VAE module | RQ-VAE 模块
│   ├── dataset.py                         # Data loading | 数据加载
│   ├── infer.py                           # Inference script | 推理脚本
│   └── run.sh                             # Run script | 运行脚本
│
└── 调研/                                   # Literature survey | 文献调研
    ├── 乱砍/                              # Random cutting experiments | 随机切割实验
    │   └── *.pdf                          # Research papers | 研究论文
    ├── 判别性分析/                        # Discriminative analysis | 判别性分析
    │   └── *.pdf                          # Research papers | 研究论文
    ├── 国内厂研/                          # Domestic research | 国内研究
    │   └── *.pdf                          # Research papers | 研究论文
    ├── 奇异谱分析/                        # Singular spectrum analysis | 奇异谱分析
    │   └── *.pdf                          # Research papers | 研究论文
    └── *.pdf                              # General papers | 通用论文
```

---

## Installation and Setup | 安装与配置

### Prerequisites | 前置条件

**English:**
- Python 3.10 or higher
- CUDA 11.7+ (for GPU acceleration)
- pip or conda package manager
- Git (for cloning the repository)

**中文:**
- Python 3.10 或更高版本
- CUDA 11.7+（用于 GPU 加速）
- pip 或 conda 包管理器
- Git（用于克隆仓库）

### Installation Steps | 安装步骤

**English:**
```bash
# 1. Clone the repository
git clone https://github.com/YichuanAlex/Recommender_System_Research.git

# 2. Navigate to project directory
cd rec

# 3. Create conda environment (recommended)
conda create -n rec python=3.10
conda activate rec

# 4. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install dependencies
pip install -r origin/requirements.txt

# 6. Install additional packages
pip install faiss-gpu tensorboard tqdm scipy
```

**中文:**
```bash
# 1. 克隆仓库
git clone https://github.com/YichuanAlex/Recommender_System_Research.git

# 2. 进入项目目录
cd rec

# 3. 创建 conda 环境（推荐）
conda create -n rec python=3.10
conda activate rec

# 4. 安装带 CUDA 支持的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装依赖
pip install -r origin/requirements.txt

# 6. 安装额外包
pip install faiss-gpu tensorboard tqdm scipy
```

### Key Dependencies | 关键依赖

```txt
# Core
torch>=2.0.0           # Deep learning framework | 深度学习框架
numpy>=1.24.0          # Numerical computing | 数值计算
pandas>=2.0.0          # Data manipulation | 数据处理

# Model
transformers>=4.30.0   # Transformer models | Transformer 模型
accelerate>=0.20.0     # Model acceleration | 模型加速

# Retrieval
faiss-gpu>=1.7.4       # Approximate nearest neighbor | 近似最近邻

# Visualization
matplotlib>=3.7.0      # Plotting library | 绘图库
tensorboard>=2.13.0    # Training visualization | 训练可视化

# Utilities
tqdm>=4.65.0           # Progress bars | 进度条
scipy>=1.10.0          # Scientific computing | 科学计算
```

---

## Usage Guide | 使用指南

### Quick Start | 快速开始

**English:**
```bash
# 1. Prepare data directory structure
mkdir -p data/TencentGR_1k
# Place your data files in the appropriate directories

# 2. Set environment variables
export TRAIN_DATA_PATH="data/TencentGR_1k"
export TRAIN_LOG_PATH="logs"
export TRAIN_CKPT_PATH="checkpoints"

# 3. Run training (using TencentGR as example)
cd TencentGR_1k_不知名副本
python start_training.py

# 4. Run inference
python run_inference.py
```

**中文:**
```bash
# 1. 准备数据目录结构
mkdir -p data/TencentGR_1k
# 将数据文件放置在相应目录中

# 2. 设置环境变量
export TRAIN_DATA_PATH="data/TencentGR_1k"
export TRAIN_LOG_PATH="logs"
export TRAIN_CKPT_PATH="checkpoints"

# 3. 运行训练（以 TencentGR 为例）
cd TencentGR_1k_不知名副本
python start_training.py

# 4. 运行推理
python run_inference.py
```

### Training Configuration | 训练配置

**English:**
```bash
# Example training command with custom parameters
python main.py \
    --batch_size=256 \
    --lr=0.0005 \
    --maxlen=101 \
    --hidden_units=64 \
    --num_blocks=2 \
    --num_heads=4 \
    --dropout_rate=0.2 \
    --num_epochs=100 \
    --device=cuda \
    --mm_emb_id 81 82 83 84 85 86
```

**中文:**
```bash
# 带自定义参数的训练命令示例
python main.py \
    --batch_size=256 \
    --lr=0.0005 \
    --maxlen=101 \
    --hidden_units=64 \
    --num_blocks=2 \
    --num_heads=4 \
    --dropout_rate=0.2 \
    --num_epochs=100 \
    --device=cuda \
    --mm_emb_id 81 82 83 84 85 86
```

### Inference Pipeline | 推理流程

**English:**
```bash
# 1. Generate item embeddings
python infer.py --mode=save_item_emb

# 2. Run approximate nearest neighbor search
python infer.py --mode=retrieve

# 3. Generate final recommendations
python infer.py --mode=predict
```

**中文:**
```bash
# 1. 生成物品 embedding
python infer.py --mode=save_item_emb

# 2. 运行近似最近邻搜索
python infer.py --mode=retrieve

# 3. 生成最终推荐
python infer.py --mode=predict
```

---

## Model Details | 模型详情

### BaselineModel Architecture | 基础模型架构

**English:**
The `BaselineModel` is a Transformer-based sequence model with the following components:

1. **Input Layer**: 
   - User/item sparse features (embedding tables)
   - Array features (behavior sequences with positional encoding)
   - Continuous features (normalized and projected)
   - Multi-modal features (image/text embeddings)

2. **Sequence Encoder**:
   - Multi-block Transformer encoder
   - Flash Multi-Head Attention (optimized for long sequences)
   - Layer normalization (pre-norm or post-norm)
   - Dropout regularization

3. **Prediction Head**:
   - Point-wise scoring: `score = dot(user_emb, item_emb)`
   - Pairwise ranking: InfoNCE contrastive loss
   - Output: sigmoid/softmax probability

**中文:**
`BaselineModel` 是基于 Transformer 的序列模型，包含以下组件：

1. **输入层**: 
   - 用户/物品稀疏特征（嵌入表）
   - 数组特征（带位置编码的行为序列）
   - 连续特征（归一化并投影）
   - 多模态特征（图像/文本 embedding）

2. **序列编码器**:
   - 多块 Transformer 编码器
   - Flash 多头注意力（针对长序列优化）
   - 层归一化（前归一化或后归一化）
   - Dropout 正则化

3. **预测头**:
   - 点对点评分：`score = dot(user_emb, item_emb)`
   - 成对排序：InfoNCE 对比损失
   - 输出：sigmoid/softmax 概率

### RQ-VAE Module | RQ-VAE 模块

**English:**
The RQ-VAE module provides semantic ID generation for multi-modal features:

```python
# Example usage
from model_rqvae import RQVAE

# Initialize RQ-VAE
rqvae = RQVAE(
    input_dim=4096,      # Input embedding dimension
    latent_dim=256,      # Latent space dimension
    num_quantizers=8,    # Number of residual quantizers
    num_codes=1024,      # Codebook size
    commitment_cost=0.25 # Commitment loss weight
)

# Forward pass
reconstructed, semantic_ids, loss = rqvae(multi_modal_emb)
```

**中文:**
RQ-VAE 模块为多模态特征提供语义 ID 生成：

```python
# 使用示例
from model_rqvae import RQVAE

# 初始化 RQ-VAE
rqvae = RQVAE(
    input_dim=4096,      # 输入 embedding 维度
    latent_dim=256,      # 潜在空间维度
    num_quantizers=8,    # 残差量化器数量
    num_codes=1024,      # 码本大小
    commitment_cost=0.25 # 承诺损失权重
)

# 前向传播
reconstructed, semantic_ids, loss = rqvae(multi_modal_emb)
```

---

## Technical Features | 技术特性

**English:**

### 1. Flash Attention Integration
- Automatic detection and usage of PyTorch 2.0 Flash Attention
- 2-3x speedup for long sequence modeling
- Memory efficiency for sequences up to 1000+ items

### 2. Multi-Modal Feature Support
- Image embeddings: ViT, ResNet, CLIP (3584-4096 dimensions)
- Text embeddings: BERT, RoBERTa (768-1024 dimensions)
- Automatic loading and caching from JSON/Pickle files
- Lazy loading for memory efficiency

### 3. Efficient Data Loading
- Memory-mapped file access for large datasets
- Batch processing with custom collate functions
- Multi-worker DataLoader support
- Feature statistics pre-computation

### 4. Checkpoint Management
- Automatic saving with best validation metric
- Resume training from checkpoints
- Model versioning and metadata tracking
- TensorBoard integration for visualization

### 5. Inference Optimization
- FAISS GPU index for fast retrieval
- Binary embedding format (.fbin, .u64bin)
- Batch processing for high throughput
- Cold start feature handling

**中文:**

### 1. Flash Attention 集成
- 自动检测和使用 PyTorch 2.0 Flash Attention
- 长序列建模速度提升 2-3 倍
- 支持 1000+ 物品序列的内存效率

### 2. 多模态特征支持
- 图像 embedding：ViT、ResNet、CLIP（3584-4096 维）
- 文本 embedding：BERT、RoBERTa（768-1024 维）
- 从 JSON/Pickle 文件自动加载和缓存
- 懒加载提高内存效率

### 3. 高效数据加载
- 大型数据集的内存映射文件访问
- 带自定义 collate 函数的批处理
- 支持多 worker DataLoader
- 特征统计预计算

### 4. 检查点管理
- 自动保存最佳验证指标
- 从检查点恢复训练
- 模型版本控制和元数据跟踪
- TensorBoard 可视化集成

### 5. 推理优化
- FAISS GPU 索引用于快速检索
- 二进制 embedding 格式（.fbin、.u64bin）
- 高吞吐量批处理
- 冷启动特征处理

---

## Citation | 引用建议

**English:**
If you use this code in your research, please cite:

```bibtex
@misc{jiang2026recommender,
  author = {Jiang, Zixi},
  title = {Recommender System Research: Multi-Modal Feature Fusion and Generative Retrieval},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/YichuanAlex/Recommender_System_Research}},
  note = {Accessed: 2026-03-30}
}
```

**中文:**
如果您在研究中使用了此代码，请引用：

```bibtex
@misc{jiang2026recommender,
  author = {江子曦},
  title = {推荐系统研究：多模态特征融合与生成式检索},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub 仓库},
  howpublished = {\url{https://github.com/YichuanAlex/Recommender_System_Research}},
  note = {访问日期：2026-03-30}
}
```

---

## License | 许可证

**English:**
This project is licensed under the MIT License - see the LICENSE file for details.

**中文:**
本项目采用 MIT 许可证 - 详见 LICENSE 文件。

---

## Contact | 联系方式

**English:**
For questions, suggestions, or collaborations, please contact:

- **Author**: Zixi Jiang (YichuanAlex)
- **Email**: jiangzixi1527435659@gmail.com
- **GitHub**: https://github.com/YichuanAlex
- **Location**: Shanghai, Shanghai, China

**中文:**
如有问题、建议或合作意向，请联系：

- **作者**: 江子曦 (YichuanAlex)
- **邮箱**: jiangzixi1527435659@gmail.com
- **GitHub**: https://github.com/YichuanAlex
- **地点**: 中国上海

---

<div align="center">

**🎯 Advanced Recommender Systems · Multi-Modal Fusion · Generative Retrieval 🚀**

**先进推荐系统 · 多模态融合 · 生成式检索**

**Made with ❤️ by YichuanAlex**

</div>
