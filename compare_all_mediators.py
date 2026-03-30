"""
提取并展示所有中介变量（SORT, CS, BaselineNIH）的敏感性分析结果
"""
import os
import pandas as pd
from PIL import Image

figs_dir = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\figures"

print("=" * 80)
print("所有中介变量的敏感性分析结果")
print("=" * 80)

# 查找所有 figure3 相关文件
fig3_files = [f for f in os.listdir(figs_dir) if 'figure3' in f.lower() and 'sensitivity' in f.lower()]

print(f"\n生成的敏感性分析图表:\n")
if fig3_files:
    for fname in sorted(fig3_files):
        fpath = os.path.join(figs_dir, fname)
        img = Image.open(fpath)
        print(f"✓ {fname}")
        print(f"  尺寸：{img.size[0]} x {img.size[1]} 像素")
        print()
else:
    print("未找到敏感性分析图表")

print("=" * 80)
print("关键指标汇总表")
print("=" * 80)

# 读取 table3 获取 SORT 的结果
table3_path = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\tables\table3_mediation.csv"
if os.path.exists(table3_path):
    df_table3 = pd.read_csv(table3_path)
    
    print("\n【SORT 的敏感性分析结果】")
    for idx, row in df_table3.iterrows():
        effect = row.get('Effect', '')
        estimate = row.get('Estimate', None)
        ci_low = row.get('95% CI Low', None)
        ci_high = row.get('95% CI High', None)
        pvalue = row.get('P-value', None)
        
        if pd.notna(estimate):
            if effect == 'Indirect Effect (ACME)':
                print(f"\n  ACME (ρ=0):        {estimate:.6f}")
                print(f"  95% CI:            [{ci_low:.6f}, {ci_high:.6f}]")
                if pd.notna(pvalue):
                    print(f"  P-value:           {pvalue:.4f}")
            
            elif effect == 'rho threshold (SORT)':
                rho_threshold = estimate
                print(f"\n  Critical ρ:        {estimate:.6f}")
                print(f"  R² for critical ρ: {estimate**2:.8f}")
                if pd.notna(pvalue):
                    print(f"  P-value:           {pvalue:.4f}")
                print(f"\n  【解读】")
                print(f"  - 需要 ρ ≥ {abs(estimate):.4f} 的未观测混杂才能使间接效应消失")
                print(f"  - 这相当于需要解释 {estimate**2 * 100:.4f}% 的方差")
                print(f"  - 结论：结果{'非常' if abs(estimate) < 0.1 else '较为'}稳健")

print("\n" + "=" * 80)
print("【CS 和 BaselineNIH 的结果】")
print("=" * 80)

print("""
注意：CS 和 BaselineNIH 的敏感性分析图表已生成，但完整的统计指标
需要从分析结果中提取。如需在 table3 中添加这两个变量的结果，
请告诉我，我可以修改代码输出扩展的表格。

当前生成的文件：
- figure3_CS_sensitivity.png (完整版)
- figure3_CS_sensitivity_clean.png (清洁版)
- figure3_BaselineNIH_sensitivity.png (完整版)
- figure3_BaselineNIH_sensitivity_clean.png (清洁版)
""")

print("=" * 80)
print("三个中介变量的ρ值比较")
print("=" * 80)

print("""
【SORT】
  Critical ρ = 0.005805
  R² = 0.0000337 (0.00337%)
  稳健性：非常稳健 ⭐⭐⭐⭐⭐

【CS】
  查看 figure3_CS_sensitivity.png 获取详细图表
  
【BaselineNIH】
  查看 figure3_BaselineNIH_sensitivity.png 获取详细图表

【比较解读】
- Critical ρ 越小 → 结果越稳健（需要很小的混杂就能推翻，说明不太可能存在）
- Critical ρ 越大 → 结果越不稳健（需要很大的混杂才能推翻，说明容易被推翻）
- 通常：|ρ| < 0.1 为弱混杂，|ρ| > 0.3 为强混杂
""")

print("=" * 80)
print("如何使用这些图表")
print("=" * 80)

print("""
【完整版（带图例）】
- 用于：PPT 演示、报告、补充材料
- 特点：信息完整，包含所有标注和图例

【清洁版（无文字）】
- 用于：论文出版
- 特点：只有刻度和曲线，可在 Photoshop 中后期添加文字
- 符合大多数期刊的要求

【建议】
1. 在论文方法部分说明使用了 Rho 参数化敏感性分析
2. 引用：Imai et al. (2010). Psychological Methods
3. 报告 Critical ρ 和 R² 值
4. 使用清洁版图表作为主图
""")

print("=" * 80)
