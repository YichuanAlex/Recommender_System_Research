# Sensitivity Analysis of Confounding Factors in Mediation Analysis

<div align="center">

**混杂因素敏感分析 - 中介效应稳健性评估**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Area](https://img.shields.io/badge/Research-Causal%20Inference%20%26%20Statistics-green.svg)]()

**Author | 作者**: YichuanAlex (Zixi Jiang)  
**Email**: jiangzixi1527435659@gmail.com  
**Last Updated | 最后更新**: 2026-03-30

</div>

---

## 目录 | Table of Contents

- [研究概述](#研究概述)
- [研究背景与意义](#研究背景与意义)
- [研究目标](#研究目标)
- [理论基础](#理论基础)
- [方法论](#方法论)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [安装与配置](#安装与配置)
- [使用指南](#使用指南)
- [主要研究结果](#主要研究结果)
- [敏感性分析方法](#敏感性分析方法)
- [可视化成果](#可视化成果)
- [图表使用指南](#图表使用指南)
- [引用建议](#引用建议)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 研究概述

**English:**  
This project presents a comprehensive sensitivity analysis framework for unobserved confounding in causal mediation analysis. Using the Rho-parameterized approach (Imai et al., 2010), the system evaluates the robustness of mediation effects against potential unmeasured confounders. The research focuses on three continuous mediators (SORT, CS, BaselineNIH) and provides complete statistical inference, visualization, and interpretation tools.

**中文:**  
本项目提出了一个综合的敏感性分析框架，用于评估因果中介分析中未观测混杂的影响。采用 Rho 参数化方法 (Imai et al., 2010)，系统评估了中介效应对潜在未测量混杂因素的稳健性。研究聚焦于三个连续型中介变量 (SORT, CS, BaselineNIH)，提供了完整的统计推断、可视化和解释工具。

---

## 研究背景与意义

### 研究背景

**English:**  
Mediation analysis is widely used in social sciences, epidemiology, and biomedical research to understand the mechanisms through which an exposure affects an outcome. However, a fundamental assumption in mediation analysis is the absence of unobserved confounding, which is often untestable with observed data.

Key challenges include:
1. **Untestable Assumption**: No unmeasured confounding cannot be verified from data
2. **Sensitivity to Violations**: Small violations can lead to biased estimates
3. **Lack of Robustness Assessment**: Many studies don't report sensitivity analyses
4. **Interpretation Difficulty**: Understanding how much confounding is needed to nullify effects

**中文:**  
中介分析广泛应用于社会科学、流行病学和生物医学研究，以理解暴露因素如何影响结果变量的机制。然而，中介分析的一个基本假设是不存在未观测混杂，而这通常无法从观测数据中检验。

主要挑战包括：
1. **不可检验的假设**: 无法从数据中验证无未测量混杂
2. **对违背的敏感性**: 小的违背可能导致估计偏倚
3. **缺乏稳健性评估**: 许多研究未报告敏感性分析
4. **解释困难**: 理解需要多大的混杂才能使效应消失

### 研究意义

**English:**
- **Theoretical Significance**: Provides a formal framework for assessing robustness of mediation effects
- **Methodological Contribution**: Implements Imai's sensitivity analysis with practical tools
- **Practical Value**: Helps researchers evaluate credibility of causal claims
- **Policy Implications**: Supports evidence-based decision making with uncertainty quantification

**中文:**
- **理论意义**: 提供了评估中介效应稳健性的正式框架
- **方法学贡献**: 实现了 Imai 敏感性分析的实用工具
- **实用价值**: 帮助研究者评估因果推断的可信度
- **政策含义**: 支持带有不确定性量化的循证决策

---

## 研究目标

**English:**
1. Implement Rho-parameterized sensitivity analysis for mediation models
2. Evaluate robustness of indirect effects for three continuous mediators (SORT, CS, BaselineNIH)
3. Generate publication-quality visualization of sensitivity analysis results
4. Provide comprehensive statistical metrics and interpretation guidelines
5. Create an automated analysis pipeline for reproducibility

**中文:**
1. 实现中介模型的 Rho 参数化敏感性分析
2. 评估三个连续型中介变量 (SORT, CS, BaselineNIH) 间接效应的稳健性
3. 生成出版级的敏感性分析可视化图表
4. 提供综合的统计指标和解释指南
5. 创建可复现的自动化分析流程

---

## 理论基础

### 中介分析框架

**English:**

#### Causal Mediation Model

Let:
- **X** = Exposure/Treatment variable (A1 in our study)
- **M** = Mediator variable (SORT, CS, or BaselineNIH)
- **Y** = Outcome variable (HT in our study)
- **C** = Observed confounders (covariates)

The mediation model decomposes the total effect into:
- **Natural Direct Effect (NDE)**: Effect of X on Y not through M
- **Natural Indirect Effect (NIE)**: Effect of X on Y through M
- **Total Effect (TE)**: NDE + NIE

**中文:**

#### 因果中介模型

定义：
- **X** = 暴露/处理变量（本研究中为 A1）
- **M** = 中介变量（SORT, CS 或 BaselineNIH）
- **Y** = 结果变量（HT）
- **C** = 观测混杂因素（协变量）

中介模型将总效应分解为：
- **自然直接效应 (NDE)**: X 不通过 M 对 Y 的效应
- **自然间接效应 (NIE)**: X 通过 M 对 Y 的效应
- **总效应 (TE)**: NDE + NIE

### 敏感性分析原理

**English:**

#### Rho-Parameterized Sensitivity Analysis

**Key Parameter**: ρ (rho)
- Definition: Residual correlation between mediator and outcome
- Interpretation: Degree of unobserved confounding
- Range: Typically -0.5 to 0.5

**Assumptions**:
1. No unobserved confounding when ρ = 0
2. Unobserved confounding exists when ρ ≠ 0
3. ρ captures the correlation between error terms in mediator and outcome models

**Method**:
1. Estimate ACME (Average Causal Mediation Effect) for different ρ values
2. Plot ACME as a function of ρ
3. Find critical ρ where ACME = 0
4. Assess robustness based on critical ρ magnitude

**中文:**

#### Rho 参数化敏感性分析

**关键参数**: ρ (rho)
- 定义：中介变量和结果变量的残差相关性
- 解释：未观测混杂的程度
- 范围：通常为 -0.5 到 0.5

**假设**:
1. 当 ρ = 0 时无未观测混杂
2. 当 ρ ≠ 0 时存在未观测混杂
3. ρ 捕捉中介模型和结果模型误差项之间的相关性

**方法**:
1. 估计不同 ρ 值下的 ACME（平均因果中介效应）
2. 绘制 ACME 随 ρ 变化的曲线
3. 找到使 ACME = 0 的临界 ρ 值
4. 基于临界 ρ 值的大小评估稳健性

---

## 方法论

### 1. 数据预处理

**English:**
- **Data Source**: Clinical dataset with exposure, mediators, outcome, and covariates
- **Variables**:
  - Exposure: A1 (binary/continuous)
  - Mediators: SORT, CS, BaselineNIH (continuous)
  - Outcome: HT (binary)
  - Covariates: Demographic and clinical characteristics
- **Quality Control**: Missing data handling, outlier detection, normality checks

**中文:**
- **数据来源**: 包含暴露、中介、结果和协变量的临床数据集
- **变量**:
  - 暴露：A1（二分类/连续）
  - 中介：SORT, CS, BaselineNIH（连续）
  - 结果：HT（二分类）
  - 协变量：人口学和临床特征
- **质量控制**: 缺失数据处理、异常值检测、正态性检验

### 2. 中介效应估计

**English:**

#### Counterfactual Imputation Approach (Binary Mediator)

For binary mediator FIV:
1. Fit mediator model: `M ~ X + C`
2. Fit outcome model: `Y ~ X + M + C`
3. Use counterfactual imputation to estimate:
   - ACME (Average Causal Mediation Effect)
   - ADE (Average Direct Effect)
   - Total Effect
4. Bootstrap for confidence intervals (1000 resamples)

#### Product-of-Coefficients Approach (Continuous Mediator)

For continuous mediators (SORT, CS, BaselineNIH):
1. Fit mediator model: `M = α₀ + α₁X + α₂C + ε₁`
2. Fit outcome model: `Y = β₀ + β₁X + β₂M + β₃C + ε₂`
3. Indirect effect: `ACME = α₁ × β₂`
4. Sobel test for significance
5. Bootstrap for confidence intervals

**中文:**

#### 反事实插补法（二分类中介）

对于二分类中介 FIV：
1. 拟合中介模型：`M ~ X + C`
2. 拟合结果模型：`Y ~ X + M + C`
3. 使用反事实插补估计：
   - ACME（平均因果中介效应）
   - ADE（平均直接效应）
   - 总效应
4. Bootstrap 法计算置信区间（1000 次重抽样）

#### 系数乘积法（连续中介）

对于连续中介（SORT, CS, BaselineNIH）：
1. 拟合中介模型：`M = α₀ + α₁X + α₂C + ε₁`
2. 拟合结果模型：`Y = β₀ + β₁X + β₂M + β₃C + ε₂`
3. 间接效应：`ACME = α₁ × β₂`
4. Sobel 检验评估显著性
5. Bootstrap 法计算置信区间

### 3. 敏感性分析

**English:**

#### Rho-Parameterized Sensitivity Analysis

**Procedure**:
1. Define ρ grid: -0.5 to 0.5 with step 0.01
2. For each ρ value:
   - Adjust mediator-outcome correlation
   - Re-estimate ACME using modified covariance
   - Calculate confidence intervals
3. Plot ACME(ρ) curve
4. Find critical ρ where ACME crosses zero
5. Calculate R² = (critical ρ)²

**Key Metrics**:
- **Observed ACME (ρ=0)**: Indirect effect under no unobserved confounding
- **95% CI (ρ=0)**: Confidence interval for observed ACME
- **Critical ρ**: Value of ρ where ACME = 0
- **R² for critical ρ**: Variance proportion explained by unobserved confounding

**中文:**

#### Rho 参数化敏感性分析

**步骤**:
1. 定义 ρ 网格：-0.5 到 0.5，步长 0.01
2. 对每个 ρ 值：
   - 调整中介 - 结果相关性
   - 使用修正协方差重新估计 ACME
   - 计算置信区间
3. 绘制 ACME(ρ) 曲线
4. 找到 ACME 穿过零点的临界 ρ 值
5. 计算 R² = (临界 ρ)²

**关键指标**:
- **观测 ACME (ρ=0)**: 无未观测混杂下的间接效应
- **95% CI (ρ=0)**: 观测 ACME 的置信区间
- **临界 ρ**: ACME = 0 时的 ρ 值
- **临界 ρ 的 R²**: 未观测混杂需要解释的方差比例

### 4. 模型比较与验证

**English:**
- **Bootstrap ROC Analysis**: Compare discriminative performance of models with/without mediator
- **AUC Comparison**: Model 1 (without mediator) vs Model 2 (with mediator) vs Model 3 (full model)
- **Calibration Assessment**: Hosmer-Lemeshow test for model fit
- **Sensitivity Analysis**: Multiple mediators for robustness check

**中文:**
- **Bootstrap ROC 分析**: 比较有/无中介模型的区分度
- **AUC 比较**: 模型 1（无中介）vs 模型 2（有中介）vs 模型 3（全模型）
- **校准度评估**: Hosmer-Lemeshow 检验评估模型拟合
- **敏感性分析**: 多个中介变量的稳健性检验

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   数据输入层                                │
│                Data Input Layer                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  总数据.xlsx                                         │    │
│  │  • 暴露变量 (A1)                                     │    │
│  │  • 中介变量 (SORT, CS, BaselineNIH, FIV)            │    │
│  │  • 结果变量 (HT)                                     │    │
│  │  • 协变量 (人口学、临床特征)                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  数据预处理层                               │
│              Data Preprocessing Layer                       │
│  • 数据读取与检查                                            │
│  • 缺失值处理                                                │
│  • 变量类型转换                                              │
│  • 描述性统计                                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  中介效应分析层                             │
│              Mediation Analysis Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ 二分类中介 (FIV) │  │ 连续中介 (SORT) │                   │
│  │ 反事实插补法     │  │ 系数乘积法       │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              敏感性分析层                                   │
│          Sensitivity Analysis Layer                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Rho 参数化敏感性分析                                │    │
│  │  • ρ 网格定义 (-0.5 到 0.5)                           │    │
│  │  • ACME(ρ) 曲线估计                                  │    │
│  │  • 临界 ρ 值计算                                     │    │
│  │  • 置信区间计算                                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  可视化层                                   │
│              Visualization Layer                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Figure 1: 中介效应示意图                          │    │
│  │  • Figure 2: Bootstrap ROC 曲线                      │    │
│  │  • Figure 3: 敏感性分析图 (ρ 参数化)                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  输出层                                     │
│               Output Layer                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • PNG 图表 (300 DPI)                                │    │
│  │  • CSV 统计表格                                      │    │
│  │  • Markdown 分析报告                                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心功能

**English:**
1. **Multi-Mediator Analysis**: Support for binary (FIV) and continuous (SORT, CS, BaselineNIH) mediators
2. **Rho-Parameterized Sensitivity**: Complete implementation of Imai's sensitivity analysis framework
3. **Automated Pipeline**: End-to-end analysis from raw data to publication-ready figures
4. **Bootstrap ROC Comparison**: Model performance evaluation with 1000 bootstrap resamples
5. **Dual-Version Visualization**: Complete version (with legends) and clean version (for publication)
6. **Comprehensive Metrics**: ACME, ADE, TE, critical ρ, R², confidence intervals, p-values
7. **Reproducibility**: Fully automated and documented analysis workflow

**中文:**
1. **多中介分析**: 支持二分类 (FIV) 和连续型 (SORT, CS, BaselineNIH) 中介变量
2. **Rho 参数化敏感性**: 完整实现 Imai 敏感性分析框架
3. **自动化流程**: 从原始数据到出版级图表的端到端分析
4. **Bootstrap ROC 比较**: 1000 次 Bootstrap 重抽样的模型性能评估
5. **双版本可视化**: 完整版（带图例）和清洁版（用于出版）
6. **综合指标**: ACME, ADE, TE, 临界 ρ, R², 置信区间，p 值
7. **可复现性**: 完全自动化和文档化的分析流程

---

## 项目结构

```
混杂因素敏感分析/
│
├── README.md                              # 项目文档（本文件）
├── 总数据.xlsx                            # 原始数据集
│
├── analysis_pipeline.py                   # 主分析流程脚本
│   • 数据读取与预处理                     │
│   • 中介效应估计                         │
│   • 敏感性分析                           │
│   • 图表生成                             │
│
├── compare_all_mediators.py               # 多中介变量比较分析
├── extract_key_metrics.py                 # 关键指标提取
├── extract_sensitivity_metrics.py         # 敏感性分析指标提取
├── generate_simple_roc.py                 # 简化 ROC 曲线生成
├── verify_results.py                      # 结果验证
│
├── debug_glm.py                           # GLM 模型调试
├── debug_mediation.py                     # 中介分析调试
├── check_figures.py                       # 图表检查
├── check_tables.py                        # 表格检查
│
├── view_figure3.py                        # Figure 3 查看
├── view_roc_curves.py                     # ROC 曲线查看
├── view_all_mediators.py                  # 所有中介变量查看
│
├── outputs/                               # 输出目录
│   ├── figures/                           # 生成的图表
│   │   ├── figure1_mediation.png          # 中介效应示意图
│   │   ├── figure2_bootstrap_roc.png      # Bootstrap ROC 曲线
│   │   ├── figure2_simple_roc.png         # 简化 ROC 曲线
│   │   ├── figure3_rho_sensitivity.png        # SORT 敏感性分析（完整版）
│   │   ├── figure3_rho_sensitivity_clean.png  # SORT 敏感性分析（清洁版）
│   │   ├── figure3_CS_sensitivity.png         # CS 敏感性分析（完整版）
│   │   ├── figure3_CS_sensitivity_clean.png   # CS 敏感性分析（清洁版）
│   │   ├── figure3_BaselineNIH_sensitivity.png      # BaselineNIH 敏感性分析（完整版）
│   │   └── figure3_BaselineNIH_sensitivity_clean.png# BaselineNIH 敏感性分析（清洁版）
│   │
│   └── tables/                            # 生成的表格
│       ├── table1_baseline.csv            # 基线特征表
│       ├── table2_panelA_univ_A1_to_FIV.csv   # 单因素 Logistic 回归
│       ├── table2_panelA_multiv_A1_to_FIV.csv # 多因素 Logistic 回归
│       ├── table2_panelB_ht_predictors.csv    # HT 预测因子
│       └── table3_mediation.csv           # 中介分析结果
│
├── outputs/report.md                      # 分析报告
├── CS_BaselineNIH_summary.md              # CS 和 BaselineNIH 汇总
├── complete_answers_summary.md            # 完整问题解答
├── figure2_final.md                       # Figure 2 说明
├── figure3_complete_guide.md              # Figure 3 完整指南
├── figures_improvements_summary.md        # 图表改进总结
├── roc_improvements.md                    # ROC 改进说明
├── roc_simplified.md                      # ROC 简化版说明
└── three_mediators_complete_report.md     # 三个中介变量完整报告
```

---

## 安装与配置

### 前置条件

**English:**
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

**中文:**
- Python 3.8 或更高版本
- pip 包管理器
- Git（用于克隆仓库）

### 安装步骤

**English:**
```bash
# 1. Clone the repository
git clone https://github.com/YichuanAlex/Sensitivity_analysis_of_confounding_factors.git

# 2. Navigate to project directory
cd "混杂因素敏感分析"

# 3. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn openpyxl
```

**中文:**
```bash
# 1. 克隆仓库
git clone https://github.com/YichuanAlex/Sensitivity_analysis_of_confounding_factors.git

# 2. 进入项目目录
cd "混杂因素敏感分析"

# 3. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. 安装依赖
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn openpyxl
```

### 依赖包

**English:**
```txt
numpy>=1.20.0            # Numerical computing
pandas>=1.3.0            # Data manipulation and analysis
scipy>=1.7.0             # Scientific computing
statsmodels>=0.12.0      # Statistical models (GLM, mediation)
matplotlib>=3.4.0        # Visualization
seaborn>=0.11.0          # Statistical visualization
scikit-learn>=0.24.0     # Machine learning (ROC, AUC)
openpyxl>=3.0.0          # Excel file reading
```

**中文:**
```txt
numpy>=1.20.0            # 数值计算
pandas>=1.3.0            # 数据处理和分析
scipy>=1.7.0             # 科学计算
statsmodels>=0.12.0      # 统计模型（GLM, 中介分析）
matplotlib>=3.4.0        # 可视化
seaborn>=0.11.0          # 统计可视化
scikit-learn>=0.24.0     # 机器学习（ROC, AUC）
openpyxl>=3.0.0          # Excel 文件读取
```

---

## 使用指南

### 快速开始

**English:**
```bash
# 1. Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Run complete analysis pipeline
python analysis_pipeline.py

# 3. View results in outputs/ directory
ls outputs/figures/
ls outputs/tables/
cat outputs/report.md
```

**中文:**
```bash
# 1. 激活虚拟环境
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 运行完整分析流程
python analysis_pipeline.py

# 3. 在 outputs/ 目录查看结果
ls outputs/figures/
ls outputs/tables/
cat outputs/report.md
```

### 分步执行

**English:**
```bash
# Step 1: Data preprocessing and descriptive statistics
python -c "import pandas as pd; df = pd.read_excel('总数据.xlsx'); print(df.describe())"

# Step 2: Mediation analysis for continuous mediators
python compare_all_mediators.py

# Step 3: Sensitivity analysis for all three mediators
python extract_sensitivity_metrics.py

# Step 4: Generate ROC curves
python generate_simple_roc.py

# Step 5: View and verify figures
python view_figure3.py
python view_roc_curves.py
```

**中文:**
```bash
# 步骤 1: 数据预处理和描述性统计
python -c "import pandas as pd; df = pd.read_excel('总数据.xlsx'); print(df.describe())"

# 步骤 2: 连续中介变量的中介分析
python compare_all_mediators.py

# 步骤 3: 三个中介变量的敏感性分析
python extract_sensitivity_metrics.py

# 步骤 4: 生成 ROC 曲线
python generate_simple_roc.py

# 步骤 5: 查看和验证图表
python view_figure3.py
python view_roc_curves.py
```

### 自定义分析

**English:**
```python
# 在 analysis_pipeline.py 中修改参数

# 修改 Bootstrap 次数
n_bootstraps = 1000  # 改为 5000 以获得更精确的结果

# 修改 ρ 网格范围
rho_grid = np.arange(-0.5, 0.51, 0.01)  # 更精细的网格

# 添加新的中介变量
mediators = ['SORT', 'CS', 'BaselineNIH', 'YourNewMediator']
```

**中文:**
```python
# 在 analysis_pipeline.py 中修改参数

# 修改 Bootstrap 次数
n_bootstraps = 1000  # 改为 5000 以获得更精确的结果

# 修改 ρ 网格范围
rho_grid = np.arange(-0.5, 0.51, 0.01)  # 更精细的网格

# 添加新的中介变量
mediators = ['SORT', 'CS', 'BaselineNIH', 'YourNewMediator']
```

---

## 主要研究结果

### 基线特征

**English:**
Table 1 presents baseline characteristics of the study population, including demographic variables, clinical characteristics, and distributions of exposure, mediators, and outcome.

**中文:**
表 1 展示了研究人群的基线特征，包括人口学变量、临床特征以及暴露、中介和结果变量的分布。

### 中介效应分析结果

**English:**

#### Table 2 Panel A: A1 → FIV 关联

**Univariable Analysis**:
- A1 shows significant association with FIV (OR = X.XX, 95% CI: [X.XX, X.XX], P < 0.05)

**Multivariable Analysis** (adjusted for covariates):
- A1 remains independently associated with FIV (adjusted OR = X.XX, 95% CI: [X.XX, X.XX], P < 0.05)

#### Table 2 Panel B: HT 预测因子

**Full Model**:
- A1: Significant predictor of HT (P < 0.05)
- FIV: Significant predictor of HT (P < 0.05)
- postAS: Significant predictor of HT (P < 0.05)
- Model AUC: 0.831 [0.808–0.906]

**中文:**

#### 表 2 Panel A: A1 → FIV 关联

**单因素分析**:
- A1 与 FIV 显著相关 (OR = X.XX, 95% CI: [X.XX, X.XX], P < 0.05)

**多因素分析**（调整协变量）:
- A1 与 FIV 独立相关 (校正 OR = X.XX, 95% CI: [X.XX, X.XX], P < 0.05)

#### 表 2 Panel B: HT 预测因子

**全模型**:
- A1: HT 的显著预测因子 (P < 0.05)
- FIV: HT 的显著预测因子 (P < 0.05)
- postAS: HT 的显著预测因子 (P < 0.05)
- 模型 AUC: 0.831 [0.808–0.906]

### 中介分析结果

**English:**

#### Table 3: 中介效应估计

**FIV (Binary Mediator)**:
- ACME: 1.1462 [0.4940–2.1650], P = 0.0013
- Proportion Mediated: 2.640
- Interpretation: FIV partially mediates the effect of A1 on HT

**SORT (Continuous Mediator)**:
- ACME (ρ=0): 1.146180 [0.494, 2.165]
- Sobel Test: P < 0.05
- Interpretation: SORT shows significant mediation effect

**中文:**

#### 表 3: 中介效应估计

**FIV (二分类中介)**:
- ACME: 1.1462 [0.4940–2.1650], P = 0.0013
- 中介比例：2.640
- 解释：FIV 部分中介 A1 对 HT 的效应

**SORT (连续中介)**:
- ACME (ρ=0): 1.146180 [0.494, 2.165]
- Sobel 检验：P < 0.05
- 解释：SORT 显示显著的中介效应

### 敏感性分析结果

**English:**

#### 三个中介变量的 Critical ρ 值比较

| 中介变量 | Observed ACME (ρ=0) | 95% CI | Critical ρ | R² for critical ρ | 稳健性 |
|---------|---------------------|--------|------------|-------------------|--------|
| **SORT** | 1.146180 | [0.494, 2.165] | **0.005805** | **0.0000337** | ⭐⭐⭐⭐⭐ |
| **CS** | 见图表 | 见图表 | 见图表 | 见图表 | 见图表 |
| **BaselineNIH** | 见图表 | 见图表 | 见图表 | 见图表 | 见图表 |

**稳健性评估标准**:
- |Critical ρ| < 0.1: ⭐⭐⭐⭐⭐ 非常稳健
- 0.1 ≤ |Critical ρ| < 0.2: ⭐⭐⭐⭐ 较为稳健
- 0.2 ≤ |Critical ρ| < 0.3: ⭐⭐⭐ 中等稳健
- 0.3 ≤ |Critical ρ| < 0.4: ⭐⭐ 较不稳健
- |Critical ρ| ≥ 0.4: ⭐ 不稳健

**SORT 结果解读**:
- **Critical ρ = 0.005805**（非常小）
- **R² = 0.0000337**（只需要解释 0.00337% 的方差）
- **结论**: 结果非常稳健，几乎不可能存在这么大的未观测混杂

**中文:**

#### 三个中介变量的临界 ρ 值比较

| 中介变量 | 观测 ACME (ρ=0) | 95% CI | 临界 ρ | 临界 ρ 的 R² | 稳健性 |
|---------|-----------------|--------|--------|-------------|--------|
| **SORT** | 1.146180 | [0.494, 2.165] | **0.005805** | **0.0000337** | ⭐⭐⭐⭐⭐ |
| **CS** | 见图表 | 见图表 | 见图表 | 见图表 | 见图表 |
| **BaselineNIH** | 见图表 | 见图表 | 见图表 | 见图表 | 见图表 |

**稳健性评估标准**:
- |临界 ρ| < 0.1: ⭐⭐⭐⭐⭐ 非常稳健
- 0.1 ≤ |临界 ρ| < 0.2: ⭐⭐⭐⭐ 较为稳健
- 0.2 ≤ |临界 ρ| < 0.3: ⭐⭐⭐ 中等稳健
- 0.3 ≤ |临界 ρ| < 0.4: ⭐⭐ 较不稳健
- |临界 ρ| ≥ 0.4: ⭐ 不稳健

**SORT 结果解读**:
- **临界 ρ = 0.005805**（非常小）
- **R² = 0.0000337**（只需要解释 0.00337% 的方差）
- **结论**: 结果非常稳健，几乎不可能存在这么大的未观测混杂

---

## 敏感性分析方法

### Rho 参数化敏感性分析

**English:**

#### 方法来源

**学术引用**:
```
Imai, K., Keele, L., & Tingley, D. (2010). 
A general approach to causal mediation analysis. 
Psychological Methods, 15(4), 309-334.
```

#### 核心原理

1. **参数**: ρ (Mediator-Outcome 残差相关性)
   - 表示未观测混杂对中介变量和结果变量的共同影响

2. **基本假设**:
   - 无未观测混杂时：ρ = 0
   - 存在未观测混杂时：ρ ≠ 0

3. **分析方法**:
   - 通过改变 ρ 值（从 -0.5 到 0.5）
   - 观察 ACME 如何随 ρ 变化
   - 找到使 ACME = 0 的临界 ρ 值（Critical ρ）

4. **解读**:
   - Critical ρ 越小 → 结果越稳健
   - Critical ρ 越大 → 结果越容易被推翻

#### 与其他方法的区别

| 方法 | 用途 | 参数 |
|------|------|------|
| **Rho 参数化** | 中介分析的未观测混杂 | ρ (残差相关性) |
| **E-value** | 暴露 - 结局关联 | OR/RR 比值 |
| **Cornfield 条件** | 二值暴露的混杂 | 相对风险比 |

**中文:**

#### 方法来源

**学术引用**:
```
Imai, K., Keele, L., & Tingley, D. (2010). 
A general approach to causal mediation analysis. 
Psychological Methods, 15(4), 309-334.
```

#### 核心原理

1. **参数**: ρ (Mediator-Outcome 残差相关性)
   - 表示未观测混杂对中介变量和结果变量的共同影响

2. **基本假设**:
   - 无未观测混杂时：ρ = 0
   - 存在未观测混杂时：ρ ≠ 0

3. **分析方法**:
   - 通过改变 ρ 值（从 -0.5 到 0.5）
   - 观察 ACME 如何随 ρ 变化
   - 找到使 ACME = 0 的临界 ρ 值（Critical ρ）

4. **解读**:
   - Critical ρ 越小 → 结果越稳健
   - Critical ρ 越大 → 结果越容易被推翻

#### 与其他方法的区别

| 方法 | 用途 | 参数 |
|------|------|------|
| **Rho 参数化** | 中介分析的未观测混杂 | ρ (残差相关性) |
| **E-value** | 暴露 - 结局关联 | OR/RR 比值 |
| **Cornfield 条件** | 二值暴露的混杂 | 相对风险比 |

---

## 可视化成果

### 图表类型

**English:**

#### Figure 1: Mediation Diagram
- Shows causal mediation model with paths
- Displays ACME, ADE, and Total Effect
- Suitable for introduction/methods section

#### Figure 2: Bootstrap ROC Curves
- Compares Model 1 (without mediator) vs Model 2 (with mediator)
- Shows AUC with 95% CI from 1000 bootstrap samples
- Demonstrates improvement in discrimination

#### Figure 3: Sensitivity Analysis Plot
- **完整版**: With legends and annotations
- **清洁版**: Without text labels (for publication)
- Shows ACME as function of ρ
- Indicates critical ρ where ACME = 0
- Displays 95% confidence interval as shaded area

**中文:**

#### 图 1: 中介效应示意图
- 展示因果中介模型和路径
- 显示 ACME, ADE 和总效应
- 适用于引言/方法部分

#### 图 2: Bootstrap ROC 曲线
- 比较模型 1（无中介）vs 模型 2（有中介）
- 显示 1000 次 Bootstrap 样本的 AUC 和 95% CI
- 展示区分度的改善

#### 图 3: 敏感性分析图
- **完整版**: 带图例和标注
- **清洁版**: 无文字标注（用于出版）
- 显示 ACME 随 ρ 变化的曲线
- 标注 ACME = 0 的临界 ρ 值
- 以阴影带显示 95% 置信区间

### 图表特征

**English:**

#### Figure 3 Specifications

| Element | Style |
|---------|-------|
| ACME Curve | Blue solid line (#1f77b4), linewidth 2.0 |
| Confidence Interval | Gray shaded area (alpha=0.3) |
| Null Effect | Black dashed line, linewidth 1.5 |
| Critical ρ | Red vertical solid line, linewidth 2.0 |
| Observed ACME | Green scatter point (s=150) |
| Font | Times New Roman, bold |
| Axis Labels | 14pt bold |
| Tick Labels | 12pt bold |
| Title | 16pt bold |
| Background | Pure white |
| Size | 2951 × 2354 pixels (300 DPI) |

**中文:**

#### 图 3 规格

| 元素 | 样式 |
|------|------|
| ACME 曲线 | 蓝色实线 (#1f77b4)，线宽 2.0 |
| 置信区间 | 灰色半透明阴影 (alpha=0.3) |
| Null 效应 | 黑色虚线，线宽 1.5 |
| 临界 ρ | 红色垂直实线，线宽 2.0 |
| 观测 ACME | 绿色散点 (s=150) |
| 字体 | Times New Roman，加粗 |
| 轴标签 | 14 号加粗 |
| 刻度标签 | 12 号加粗 |
| 标题 | 16 号加粗 |
| 背景 | 纯白色 |
| 尺寸 | 2951 × 2354 像素 (300 DPI) |

---

## 图表使用指南

### 两个版本的使用

**English:**

#### Version 1: Complete Version (with legends)
- **Filename**: `figure3_X_sensitivity.png`
- **Features**:
  - Complete legends and annotations
  - All information included
  - Easy to understand
- **Usage**:
  - PPT presentations
  - Academic reports
  - Supplementary materials
  - Poster presentations

#### Version 2: Clean Version (without text)
- **Filename**: `figure3_X_sensitivity_clean.png`
- **Features**:
  - No text labels (only ticks and curves)
  - Pure white background
  - Suitable for publication
- **Usage**:
  - Journal submission
  - Can add labels later in Photoshop/Illustrator

**中文:**

#### 版本 1: 完整版（带图例）
- **文件名**: `figure3_X_sensitivity.png`
- **特点**:
  - 完整的图例和标注
  - 包含所有信息
  - 易于理解
- **用途**:
  - PPT 演示
  - 学术报告
  - 补充材料
  - 海报展示

#### 版本 2: 清洁版（无文字）
- **文件名**: `figure3_X_sensitivity_clean.png`
- **特点**:
  - 无文字标注（只有刻度和曲线）
  - 纯白色背景
  - 适合出版
- **用途**:
  - 期刊投稿
  - 可在 Photoshop/Illustrator 中后期添加标注

### 论文中的图注示例

**English:**
```
Figure 3. Sensitivity analysis for unobserved confounding in mediation 
analysis. The blue line represents the Average Causal Mediation Effect 
(ACME) across different values of ρ (mediator-outcome residual correlation). 
The vertical red line indicates the critical value of ρ at which the ACME 
becomes zero. The green dot shows the observed ACME when ρ=0 (no unobserved 
confounding). The gray shaded area represents the 95% confidence interval.
```

**中文:**
```
图 3. 中介分析中未观测混杂的敏感性分析。蓝线表示不同 ρ 值（中介 - 结果
残差相关性）下的平均因果中介效应（ACME）。垂直红线表示 ACME 变为零时的
临界 ρ 值。绿点显示 ρ=0 时（无未观测混杂）观测到的 ACME。灰色阴影区域
表示 95% 置信区间。
```

### 方法部分描述

**English:**
```
We conducted sensitivity analysis for unobserved confounding using the 
rho-parameterized approach (Imai et al., 2010). This method assesses how 
strong the unobserved confounding (parameterized as ρ, the mediator-outcome 
residual correlation) would need to be to nullify the observed indirect 
effect. A small critical value of ρ indicates that the results are robust 
to unobserved confounding.
```

**中文:**
```
我们使用 Rho 参数化方法 (Imai et al., 2010) 进行了未观测混杂的敏感性
分析。该方法评估需要多大的未观测混杂（参数化为 ρ，即中介 - 结果残差相
关性）才能使观测到的间接效应消失。临界 ρ 值越小，表明结果对未观测混杂
越稳健。
```

---

## 引用建议

**English:**
```bibtex
@mastersthesis{jiang2026sensitivity,
  title={Sensitivity Analysis of Confounding Factors in Mediation Analysis},
  author={Jiang, Zixi},
  school={Liaoning Normal University},
  year={2026},
  address={Dalian, Liaoning, China},
  language={Chinese}
}

@article{imai2010general,
  title={A general approach to causal mediation analysis},
  author={Imai, Kosuke and Keele, Luke and Tingley, Dustin},
  journal={Psychological Methods},
  volume={15},
  number={4},
  pages={309--334},
  year={2010},
  publisher={American Psychological Association}
}
```

**中文:**
```
姜子西。(2026). 中介分析中混杂因素的敏感性研究 (硕士学位论文). 辽宁师范大学，大连.

Imai, K., Keele, L., & Tingley, D. (2010). A general approach to causal 
mediation analysis. Psychological Methods, 15(4), 309-334.
```

---

## 许可证

**English:**
This project is licensed under the MIT License. You are free to use, modify, and distribute this work for academic and non-commercial purposes. Please cite the original author when using this research.

**中文:**
本项目采用 MIT 许可证。您可以自由地使用、修改和分发本作品用于学术和非商业目的。使用本研究时请注明原作者。

---

## 联系方式

**English:**
For questions, suggestions, or collaborations, please contact:

- **Author**: Zixi Jiang (YichuanAlex)
- **Email**: jiangzixi1527435659@gmail.com
- **GitHub**: https://github.com/YichuanAlex
- **Institution**: Liaoning Normal University, Dalian, China

**中文:**
如有问题、建议或合作意向，请联系：

- **作者**: 姜子西 (YichuanAlex)
- **邮箱**: jiangzixi1527435659@gmail.com
- **GitHub**: https://github.com/YichuanAlex
- **单位**: 辽宁师范大学，中国大连

---

<div align="center">

**🔬 因果推断 · 中介分析 · 敏感性评估 📊**

**Causal Inference · Mediation Analysis · Sensitivity Assessment**

[![Robust Statistics](https://img.shields.io/badge/Statistics-Robust%20%26%20Reliable-blue.svg)]()
[![Reproducible Research](https://img.shields.io/badge/Research-Reproducible-green.svg)]()

**感谢使用本研究项目！**

**Thank you for using this research project!**

</div>
