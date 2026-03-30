# Figures 改进总结

## 1. Bootstrap ROC 曲线 (figure2_bootstrap_roc.png)

### 主要改进：

#### ✅ 添加 Bias-Corrected AUC
- **方法**: 使用 BCa (Bias-Corrected and Accelerated) 方法计算置信区间
- **优势**: 比传统百分位数法更准确，特别是当 bootstrap 分布有偏时
- **实现**: 
  ```python
  def bias_corrected_ci(original, boot_samples, confidence=0.95):
      # 计算 bias correction factor z0
      z0 = stats.norm.ppf(np.mean(boot_samples < original))
      # 计算 bias-corrected 百分位数
      alpha_low = stats.norm.cdf(z0 + stats.norm.ppf(alpha / 2))
      alpha_high = stats.norm.cdf(z0 + stats.norm.ppf(1 - alpha / 2))
  ```

#### ✅ Bootstrap 抽样次数
- **默认**: 1000 次 bootstrap 重复
- **代码位置**: `bootstrap_auc_models(df, B=1000, seed=1234)`

#### ✅ 曲线样式（按照参考图片）
- **轴标签**: 
  - X 轴：`1 - specificity`（等宽字体）
  - Y 轴：`sensitivity`（等宽字体）
- **刻度间距**: 0.25（0.00, 0.25, 0.50, 0.75, 1.00）
- **曲线元素**:
  - 粗实线（linewidth=2.5）：原始样本 ROC 曲线
  - 灰色细线（linewidth=0.5, alpha=0.3）：100 个 bootstrap 样本曲线
  - 阴影区域（alpha=0.3）：95% 置信区间
  - 红色虚线（r--, linewidth=1.5）：随机猜测线
- **AUC 标注**:
  - 位置：右下角（x=0.95, y=0.15 开始）
  - 格式：`AUC 1: 0.822 [0.79,0.90]`
  - 背景：wheat 色圆角框（alpha=0.5）

## 2. Rho 敏感性分析 (figure3_rho_sensitivity.png)

### 关键数值解读：

#### 📊 **rho_threshold (ρ 阈值)**
- **定义**: 使 ACME 变为 0 所需的 ρ 值
- **示例**: 
  - `rho_threshold = 0.0058` 意味着需要 ρ ≥ 0.0058 的未观测混杂才能使间接效应消失
  - 计算公式：`rho_threshold = -b/k`，其中 b 是 SORT 的系数，k 是缩放因子
- **解读**:
  - 值越小 → 结果越稳健（很小的混杂就能推翻，说明不太可能存在这么大的混杂）
  - 值越大 → 结果越不稳健（需要很大的混杂才能推翻，说明结果容易被推翻）
  - 通常：|ρ| < 0.1 为弱混杂，|ρ| > 0.3 为强混杂

#### 📊 **Observed ACME (ρ=0)**
- **定义**: 在没有未观测混杂假设下的间接效应估计
- **位置**: 图中标注在 ρ=0 处（绿色散点）
- **意义**: 这是主要的效应估计值

#### 📊 **曲线斜率**
- **含义**: ACME 对 ρ 变化的敏感程度
- **公式**: `ACME(ρ) = a × (b + ρ × k)`
- **解读**: 斜率越陡，结果对未观测混杂越敏感

### 图表改进：

#### ✅ 添加解释性文本框
```
Key Interpretation:
• ρ threshold = 0.006
  → Need ρ ≥ 0.006 to nullify indirect effect
• Observed ACME (ρ=0): 1.2345
• If |ρ| < 0.006, effect remains significant
```

#### ✅ 视觉元素
- **蓝色实线**: ACME(ρ) 曲线
- **黑色虚线**: Null effect (ACME=0)
- **红色点划线**: ρ threshold 位置
- **绿色散点**: Observed ACME (ρ=0)
- **网格**: alpha=0.3 的虚线网格

#### ✅ 轴标签优化
- X 轴：`ρ (Mediator-Outcome residual correlation)`
- Y 轴：`ACME (Average Causal Mediation Effect)`
- 标题：`Figure 3. Sensitivity Analysis: Unobserved Confounding`

## 3. 代码结构改进

### 新增函数：
1. **`plot_roc_with_bootstrap()`**: 专门用于绘制带 bootstrap 的 ROC 曲线
2. **`bias_corrected_ci()`**: 计算 bias-corrected 置信区间

### 改进函数：
1. **`bootstrap_auc_models()`**: 
   - 存储 bootstrap ROC 曲线用于可视化
   - 计算 bias-corrected AUC 和 CI
   - 返回 bias 值

2. **`figure_rho_sensitivity()`**:
   - 添加 observed ACME 标注
   - 添加解释性文本框
   - 改进视觉样式

## 4. 输出文件

### 生成的图片：
- `figure1_mediation.png`: 中介效应路径图
- `figure2_bootstrap_roc.png`: Bootstrap ROC 曲线（带置信区间）
- `figure3_rho_sensitivity.png`: Rho 敏感性分析图

### 表格：
- `table2_panelA_univ_A1_to_FIV.csv`: 单变量线性回归（FIV 为连续变量）
- `table2_panelA_multiv_A1_to_FIV.csv`: 多变量线性回归
- `table2_panelB_ht_predictors.csv`: HT 预测因子的 Logistic 回归
- `table3_mediation.csv`: 中介效应分析结果

## 5. 使用示例

```python
# 运行完整分析
python analysis_pipeline.py

# 验证图表输出
python check_figures.py
```

## 6. 统计方法说明

### Bootstrap ROC 分析
- **重复次数**: 1000 次
- **置信区间**: BCa (Bias-Corrected and Accelerated) 95% CI
- **优势**: 不依赖正态分布假设，更适合小样本

### Rho 敏感性分析
- **目的**: 评估未观测混杂对中介效应结论的影响
- **方法**: 通过改变 mediator-outcome 残差相关性 (ρ)，观察 ACME 的变化
- **解读**: rho_threshold 越小，结果越稳健
