# Figure 2 最终版本修改说明

## 本次修改内容

### ✅ 1. 刻度间隔调整
- **X 轴**: 0.0, 0.1, 0.2, ..., 1.0（间隔 0.1）
- **Y 轴**: 0.0, 0.1, 0.2, ..., 1.0（间隔 0.1）
- **之前**: 间隔 0.25

### ✅ 2. 字体加粗
所有文字均使用**粗体**：
- **轴标签**: Times New Roman 16 号 **加粗**
- **刻度标签**: Times New Roman 14 号 **加粗**
- **AUC 标注**: Times New Roman 14 号 **加粗**

### ✅ 3. 主曲线颜色
- **颜色**: 深蓝色 (#00008B)
- **之前**: 浅蓝色 (#87CEEB)
- **线宽**: 2.5（保持不变）

## 代码修改详情

### 1. 颜色修改
```python
# 之前
main_color = '#87CEEB'  # Sky blue

# 现在
main_color = '#00008B'  # Dark blue
```

### 2. 轴标签加粗
```python
# 之前
ax.set_xlabel('1 - specificity', fontsize=16, fontname='Times New Roman')
ax.set_ylabel('sensitivity', fontsize=16, fontname='Times New Roman')

# 现在
ax.set_xlabel('1 - specificity', fontsize=16, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('sensitivity', fontsize=16, fontname='Times New Roman', fontweight='bold')
```

### 3. 刻度间隔修改
```python
# 之前
ax.set_xticks(np.arange(0, 1.01, 0.25))
ax.set_yticks(np.arange(0, 1.01, 0.25))

# 现在
ax.set_xticks(np.arange(0, 1.01, 0.1))
ax.set_yticks(np.arange(0, 1.01, 0.1))
```

### 4. 刻度标签加粗
```python
# 之前
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 现在
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')
```

### 5. AUC 标注加粗
```python
# 之前
ax.text(text_pos_x, text_pos_y, 
        auc_text,
        transform=ax.transAxes, 
        fontsize=14, 
        fontname='Times New Roman',
        verticalalignment='center',
        horizontalalignment='right')

# 现在
ax.text(text_pos_x, text_pos_y, 
        auc_text,
        transform=ax.transAxes, 
        fontsize=14, 
        fontname='Times New Roman',
        fontweight='bold',
        verticalalignment='center',
        horizontalalignment='right')
```

## 最终图形特征

### 视觉元素
| 元素 | 属性 |
|------|------|
| 主 ROC 曲线 | 深蓝色 (#00008B)，线宽 2.5 |
| Bootstrap 曲线 | 浅灰色，1000 条，线宽 0.5，alpha 0.3 |
| 置信区间 | 灰色半透明阴影，alpha 0.3 |
| 对角线 | 红色虚线，线宽 1.5 |
| X 轴刻度 | 0.0, 0.1, 0.2, ..., 1.0 |
| Y 轴刻度 | 0.0, 0.1, 0.2, ..., 1.0 |
| 轴标签 | Times New Roman 16 号 加粗 |
| 刻度标签 | Times New Roman 14 号 加粗 |
| AUC 标注 | Times New Roman 14 号 加粗 |

### 标注内容
```
AUC: 0.714 [0.66,0.78]
```
- 位置：右下角 (x=0.95, y=0.25)
- 格式：简洁无背景框
- 字体：加粗 Times New Roman

## 文件位置
`outputs/figures/figure2_bootstrap_roc.png`

## 与上一版本的对比

| 特征 | 上一版本 | 当前版本 |
|------|----------|----------|
| 主曲线颜色 | 浅蓝色 (#87CEEB) | 深蓝色 (#00008B) |
| 刻度间隔 | 0.25 | 0.1 |
| 轴标签字体 | 常规 | **加粗** |
| 刻度标签字体 | 常规 | **加粗** |
| AUC 标注字体 | 常规 | **加粗** |

所有修改已完成，图形更加清晰、专业！
