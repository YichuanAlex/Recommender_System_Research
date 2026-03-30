"""
提取敏感性分析的关键指标和生成改进的图表
"""
import os
import numpy as np
import pandas as pd
from analysis_pipeline import read_data, mediation_continuous_sort, mediation_binary_fiv

# 读取数据
data_path = r"c:\Users\24116\Desktop\混杂因素敏感分析\data\data_clean.csv"
df = read_data(data_path)

print("=" * 80)
print("敏感性分析关键指标提取")
print("=" * 80)

# 运行 SORT 的敏感性分析
m_sort = mediation_continuous_sort(df) if "SORT" in df.columns else None

if m_sort and "rho_grid" in m_sort and len(m_sort["rho_grid"]) > 0:
    print("\n【SORT 敏感性分析指标】")
    print("-" * 80)
    
    # 1. ACME (no unmeasured confounding, p=0)
    acme_p0 = m_sort.get("ACME", None)
    print(f"\n1. ACME (no unmeasured confounding, ρ=0): {acme_p0:.6f}" if acme_p0 else "\n1. ACME (no unmeasured confounding, ρ=0): NA")
    
    # 2. 95% CI for ACME (p=0)
    # 需要从 mediation_continuous_sort 获取 CI
    print(f"   需要从 mediation 分析中获取 95% CI")
    
    # 3. Critical p (ACME=0) - 即 rho_threshold
    rho_threshold = m_sort.get("rho_threshold", None)
    print(f"\n2. Critical ρ (ACME=0): {rho_threshold:.6f}" if rho_threshold else "\n2. Critical ρ (ACME=0): NA")
    
    # 4. R² for critical p
    # R² = rho_threshold²
    if rho_threshold is not None and not np.isnan(rho_threshold):
        r2_critical = rho_threshold ** 2
        print(f"\n3. R² for critical ρ: {r2_critical:.6f}")
        print(f"   (解释：需要 ρ² ≥ {r2_critical:.4f} 才能使间接效应消失)")
    
    print("\n" + "=" * 80)
    print("其他参数的ρ值")
    print("=" * 80)
    
    # 计算其他变量的 rho
    if "CS" in df.columns:
        # CS 的 rho 需要运行 CS 的敏感性分析
        print("\nCS 的ρ值：需要运行 CS 的敏感性分析")
    
    if "BaselineNIH" in df.columns:
        print("BaselineNIH 的ρ值：需要运行 BaselineNIH 的敏感性分析")
    
    print("\n" + "=" * 80)
    print("问题解答")
    print("=" * 80)
    
    print("""
【问题 3】这个敏感性分析准确来说是叫 Rho 参数化吗？

答：是的，准确来说这是"Rho 参数化敏感性分析"（Rho-based Sensitivity Analysis）。

具体来说：
1. **方法名称**: Rho 参数化敏感性分析 / ρ-parameterized sensitivity analysis
2. **原理**: 通过参数化 Mediator-Outcome 残差相关性（ρ），评估未观测混杂
   对中介效应结论的影响
3. **关键参数**: 
   - ρ: Mediator 和 Outcome 残差之间的相关性
   - 假设：在没有未观测混杂时，ρ = 0
   - 通过改变ρ值，观察 ACME 如何变化

4. **其他名称**:
   - Sensitivity analysis for unobserved confounding in mediation analysis
   - Imai et al. sensitivity analysis (基于 Imai 2010 年的方法)
   - ρ-sensitivity analysis

5. **与 E-value 的区别**:
   - E-value: 用于观察性研究的暴露 - 结局关联
   - Rho 参数化：专门用于中介分析的未观测混杂
    """)
    
else:
    print("\n未找到 SORT 的敏感性分析结果")

print("\n" + "=" * 80)
