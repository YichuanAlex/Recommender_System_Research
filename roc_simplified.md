# ROC 曲线简化版本说明

## 修改内容

根据您的要求，已将 ROC 曲线图简化为**单模型展示**：

### ✅ 主要变化

1. **只保留一个模型** - AUC1 (Model 1)
   - 移除了 Model 2 和 Model 3
   - 只显示第一个模型的 ROC 曲线

2. **颜色方案**
   - **主曲线**: 天蓝色 (#87CEEB) - 浅蓝色
   - **Bootstrap 曲线**: 灰色
   - **置信区间**: 灰色半透明 (alpha=0.3)
   - **对角线**: 红色虚线

3. **AUC 标注简化**
   - **格式**: `AUC: 0.714 [0.66,0.78]`
   - **位置**: 右下角 (x=0.95, y=0.25)
   - **字体**: Times New Roman 14 号
   - **无背景框**: 直接显示文字

4. **图形元素**
   - **主曲线**: 线宽 2.5（粗线，蓝色）
   - **Bootstrap 曲线**: 显示所有 1000 个样本（灰色细线）
   - **置信区间**: 灰色阴影区域
   - **对角线**: 红色虚线，线宽 1.5

5. **坐标轴**
   - **刻度间隔**: 0.25
   - **标签字体**: Times New Roman 16 号
   - **刻度字体**: Times New Roman 14 号
   - **边框线宽**: 1.0

## 代码修改

### 1. `plot_roc_with_bootstrap` 函数

```python
def plot_roc_with_bootstrap(y_true, prob_list, model_names, auc_values, bc_ci_list, boot_curves, out_path, n_bootstrap=1000):
    """
    Plot single ROC curve with bootstrap samples showing confidence bands.
    Simplified version with only one model (AUC1).
    """
    # 只处理第一个模型
    fpr, tpr, _ = roc_curve(y_true, prob_list[0])
    
    # 蓝色主曲线
    ax.plot(fpr, tpr, color='#87CEEB', linewidth=2.5)
    
    # 灰色 bootstrap 曲线
    for fpr_b, tpr_b in boot_curves[1]:
        ax.plot(fpr_b, tpr_b, color='lightgray', linewidth=0.5, alpha=0.3)
    
    # 灰色置信区间
    ax.fill_between(fpr_points, tpr_lower_list, tpr_upper_list, 
                   alpha=0.3, color='gray', linewidth=0)
    
    # 简单 AUC 标注
    auc_text = f'AUC: {auc_val:.3f} [{bc_ci[0]:.2f},{bc_ci[1]:.2f}]'
    ax.text(text_pos_x, text_pos_y, auc_text, ...)
```

### 2. 调用处修改

```python
# 之前：三个模型
plot_roc_with_bootstrap(
    y, [p1, p2, p3], 
    ["Model 1", "Model 2", "Model 3"],
    [auc1, auc2, auc3],
    [bc_ci1, bc_ci2, bc_ci3],
    boot_curves,
    ...
)

# 现在：只保留 Model 1
plot_roc_with_bootstrap(
    y, [p1], 
    ["Model 1"],
    [auc1],
    [bc_ci1],
    boot_curves,
    ...
)
```

## 最终效果

生成的图片包含：
- 一条天蓝色的主 ROC 曲线
- 1000 条灰色 bootstrap 样本曲线（形成阴影效果）
- 灰色半透明置信区间
- 红色虚线对角线
- 右下角简洁的 AUC 标注
- 坐标轴标签：sensitivity 和 1-specificity
- 刻度间隔：0.00, 0.25, 0.50, 0.75, 1.00

## 文件位置

`outputs/figures/figure2_bootstrap_roc.png`

## 与之前版本的对比

| 特征 | 之前版本 | 简化版本 |
|------|----------|----------|
| 模型数量 | 3 个 | 1 个 |
| 曲线颜色 | 三种蓝色 | 单一浅蓝色 |
| Bootstrap 曲线 | 200 条 | 1000 条（全部） |
| 置信区间颜色 | 对应曲线颜色 | 灰色 |
| AUC 标注 | 三个带背景框 | 一个无背景框 |
| 标注格式 | `AUC 1: 0.714 [...]` | `AUC: 0.714 [...]` |
| 标签字体大小 | 12 号 | 16 号 |
| 刻度字体大小 | 12 号 | 14 号 |
| 边框线宽 | 0.5 | 1.0 |

简化版本更加清晰、专注，突出了 Model 1 的性能！
