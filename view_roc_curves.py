"""
查看生成的 ROC 曲线图
"""
from PIL import Image
import os

figs_dir = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\figures"

print("=" * 80)
print("ROC 曲线图生成结果")
print("=" * 80)

# 查找所有 figure2 相关文件
fig2_files = [f for f in os.listdir(figs_dir) if 'figure2' in f.lower()]

print(f"\n生成的 ROC 曲线图 ({len(fig2_files)} 个):\n")

for fname in sorted(fig2_files):
    fpath = os.path.join(figs_dir, fname)
    img = Image.open(fpath)
    
    if 'simple' in fname.lower():
        version = "简单 ROC 曲线（无 bootstrap）"
    else:
        version = "Bootstrap ROC 曲线（带置信带）"
    
    print(f"✓ {fname}")
    print(f"  版本：{version}")
    print(f"  尺寸：{img.size[0]} x {img.size[1]} 像素")
    print(f"  分辨率：300 DPI（出版质量）")
    print(f"  路径：{fpath}")
    print()

print("=" * 80)
print("图表特点")
print("=" * 80)

print("""
【简单 ROC 曲线】(figure2_simple_roc.png)
  - 只显示单条 ROC 曲线
  - 无 bootstrap 抽样曲线
  - 无置信区间阴影
  - 简洁、清晰
  - 适用于：论文主图、海报、演示

【Bootstrap ROC 曲线】(figure2_bootstrap_roc.png)
  - 显示 ROC 曲线
  - 显示 1000 条 bootstrap 抽样曲线（浅灰色）
  - 显示 95% 置信区间（灰色阴影）
  - 信息更丰富
  - 适用于：补充材料、方法学展示
""")

print("=" * 80)
print("格式规范")
print("=" * 80)

print("""
【字体】
  - 所有文字：Times New Roman
  - 轴标签：16pt 加粗
  - 刻度标签：14pt 加粗
  - AUC 标注：14pt 加粗

【颜色】
  - ROC 曲线：深蓝色 (#00008B)
  - 参考线：红色虚线 (#DC143C)
  - Bootstrap 曲线：浅灰色
  - 置信带：灰色半透明

【刻度】
  - X 轴：1 - specificity (0.1 间隔)
  - Y 轴：sensitivity (0.1 间隔)
  - 范围：[0, 1]

【标注】
  - 位置：右下角
  - 格式：AUC: 0.714 [0.68,0.80]
  - 包含 bias-corrected 95% CI

【其他】
  - 背景：白色
  - 边框：1.0pt 黑色
  - 分辨率：300 DPI
  - 格式：PNG
""")

print("=" * 80)
print("如何使用")
print("=" * 80)

print("""
【论文投稿】
  推荐使用：figure2_simple_roc.png
  理由：简洁、专业、符合大多数期刊要求

【PPT 演示】
  推荐使用：figure2_simple_roc.png
  理由：清晰、易于理解

【补充材料】
  推荐使用：figure2_bootstrap_roc.png
  理由：展示方法学细节

【海报展示】
  推荐使用：figure2_simple_roc.png
  理由：远距离观看更清晰
""")

print("=" * 80)
print("查看图片")
print("=" * 80)

print(f"""
文件位置：
{figs_dir}

打开方式：
1. Windows 资源管理器：双击文件
2. Python: 
   from PIL import Image
   img = Image.open(r"{os.path.join(figs_dir, 'figure2_simple_roc.png')}")
   img.show()
3. 任何图片查看器
""")

print("=" * 80)
