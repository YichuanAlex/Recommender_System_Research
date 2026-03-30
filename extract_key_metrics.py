"""
提取敏感性分析的所有关键指标
"""
import os
import pandas as pd

# 读取 table3_mediation.csv
table3_path = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\tables\table3_mediation.csv"

print("=" * 80)
print("Table 3 中介分析结果")
print("=" * 80)

if os.path.exists(table3_path):
    df_table3 = pd.read_csv(table3_path)
    print("\n" + df_table3.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("关键指标 (Key Indicators)")
    print("=" * 80)
    
    # 提取关键值
    for idx, row in df_table3.iterrows():
        effect = row.get('Effect', '')
        estimate = row.get('Estimate', None)
        ci_low = row.get('95% CI Low', None)
        ci_high = row.get('95% CI High', None)
        pvalue = row.get('P-value', None)
        
        if pd.notna(estimate):
            if effect == 'Indirect Effect (ACME)':
                print(f"\n【ACME 指标】")
                print(f"  ACME (ρ=0):        {estimate:.6f}")
                print(f"  95% CI:            [{ci_low:.6f}, {ci_high:.6f}]")
                print(f"  P-value:           {pvalue:.4f}" if pd.notna(pvalue) else "")
            
            elif effect == 'rho threshold (SORT)':
                rho_threshold = estimate
                print(f"\n【Critical ρ 指标】")
                print(f"  Critical ρ:        {estimate:.6f}")
                print(f"  R² for critical ρ: {estimate**2:.8f}")
                print(f"  P-value:           {pvalue:.4f}" if pd.notna(pvalue) else "")
                print(f"\n  解读：需要 ρ ≥ {abs(estimate):.4f} 的未观测混杂才能使间接效应消失")
                print(f"        这相当于需要解释 {estimate**2 * 100:.4f}% 的方差")
    
    print("\n" + "=" * 80)
    print("问题解答总结")
    print("=" * 80)
    
    print("""
【问题 2 答案】关键指标值：

1. ACME (no unmeasured confounding, ρ=0): 见上表 ACME 估计值
2. 95% CI for ACME (ρ=0): 见上表 ACME 的 95% CI
3. Critical ρ (ACME=0): 见上表 rho threshold
4. R² for critical ρ: (Critical ρ)²

【问题 3 答案】
是的，这是"Rho 参数化敏感性分析"（Rho-parameterized Sensitivity Analysis）
- 基于 Imai et al. (2010) 的方法
- 专门用于中介分析的未观测混杂评估

【问题 5 答案】
当前只分析了 SORT 作为中介变量的ρ值。
如需 CS 和 BaselineNIH 的ρ值，需要分别运行它们的敏感性分析。
    """)
    
else:
    print(f"\n错误：未找到文件 {table3_path}")
    print("请先运行 analysis_pipeline.py 生成结果")

print("\n" + "=" * 80)
