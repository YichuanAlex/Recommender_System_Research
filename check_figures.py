import os
import matplotlib.pyplot as plt
from PIL import Image

figs_dir = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\figures"

print("=" * 80)
print("Figure 输出验证")
print("=" * 80)

# 检查生成的图片
fig_files = [
    "figure1_mediation.png",
    "figure2_bootstrap_roc.png", 
    "figure3_rho_sensitivity.png"
]

for fname in fig_files:
    fpath = os.path.join(figs_dir, fname)
    if os.path.exists(fpath):
        print(f"\n✓ {fname} 已生成")
        img = Image.open(fpath)
        print(f"  尺寸：{img.size[0]} x {img.size[1]} 像素")
    else:
        print(f"\n✗ {fname} 未生成")

print("\n" + "=" * 80)
print("Bootstrap ROC 分析详情")
print("=" * 80)

# 从 report.md 读取 ROC 信息
report_path = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\report.md"
if os.path.exists(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 查找 AUC 相关信息
        import re
        auc_matches = re.findall(r'AUC.*?\[.*?\]', content)
        if auc_matches:
            print("\nAUC 值 (bias-corrected):")
            for match in auc_matches:
                print(f"  {match}")

print("\n" + "=" * 80)
print("Rho 敏感性分析解读")
print("=" * 80)
print("""
关键数值说明：

1. **rho_threshold (ρ 阈值)**: 
   - 这是使 ACME 变为 0 所需的 ρ 值
   - 例如：rho_threshold = 0.0058 意味着需要 ρ ≥ 0.0058 的未观测混杂才能使间接效应消失
   - 值越小表示结果越稳健（只需要很小的混杂就能推翻结论）
   - 值越大表示结果越不稳健（需要很大的混杂才能推翻结论）

2. **Observed ACME (ρ=0)**: 
   - 这是在没有未观测混杂假设下的间接效应估计
   - 正常情况下应该显著不为 0

3. **曲线斜率**:
   - 表示 ACME 对 ρ 变化的敏感程度
   - 斜率越陡，结果越敏感

4. **解读示例**:
   - 如果 rho_threshold = 0.3，意味着需要较强的未观测混杂（ρ≥0.3）才能推翻结论
   - 如果 rho_threshold = 0.01，意味着只需要很弱的未观测混杂（ρ≥0.01）就能推翻结论
   - 通常认为 |ρ| > 0.3 是较强的混杂，|ρ| < 0.1 是较弱的混杂
""")

print("\n" + "=" * 80)
print("Bootstrap ROC 曲线说明")
print("=" * 80)
print("""
1. **抽样次数**: 1000 次 bootstrap 重复

2. **图中元素**:
   - 粗实线：原始样本的 ROC 曲线
   - 灰色细线：100 个 bootstrap 样本的 ROC 曲线（展示变异性）
   - 阴影区域：95% 置信区间（bias-corrected）
   - 红色虚线：随机猜测线（AUC=0.5）

3. **标注内容**:
   - AUC 值：bias-corrected 的 AUC 估计
   - 95% CI：bias-corrected 置信区间
   - 使用 sensitivity 和 1-specificity 作为轴标签

4. **Bias-corrected AUC**:
   - 使用 BCa (Bias-Corrected and Accelerated) 方法计算
   - 比传统百分位数法更准确，特别是当 bootstrap 分布有偏时
   - 同时报告 bias 值（bootstrap 均值 - 原始估计）
""")

print("\n图表已更新完成！")
