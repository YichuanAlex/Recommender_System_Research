import pandas as pd
import numpy as np
import os

tables_dir = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\tables"

print("=" * 80)
print("Table 3 Mediation Analysis Results")
print("=" * 80)

df = pd.read_csv(os.path.join(tables_dir, "table3_mediation.csv"))
print("\nTable 3 内容:")
print(df.to_string(index=False))

print("\n" + "=" * 80)
print("数值合理性检查")
print("=" * 80)

acme = df.loc[df['Effect'] == 'Indirect Effect (ACME)', 'Estimate'].iloc[0]
acme_ci_low = df.loc[df['Effect'] == 'Indirect Effect (ACME)', '95% CI Low'].iloc[0]
acme_ci_high = df.loc[df['Effect'] == 'Indirect Effect (ACME)', '95% CI High'].iloc[0]
acme_p = df.loc[df['Effect'] == 'Indirect Effect (ACME)', 'P-value'].iloc[0]

print(f"\nACME (间接效应):")
print(f"  Estimate = {acme:.6f}")
print(f"  95% CI = [{acme_ci_low:.6f}, {acme_ci_high:.6f}]")
print(f"  P-value = {acme_p:.6f}")
print(f"  ✓ ACME 不为 0: {acme != 0}")
print(f"  ✓ CI 不包含 NaN: {not pd.isna(acme_ci_low) and not pd.isna(acme_ci_high)}")
print(f"  ✓ P-value 不为 0 或 1: {acme_p != 0 and acme_p != 1}")

pm = df.loc[df['Effect'] == 'Proportion Mediated', 'Estimate'].iloc[0]
pm_ci_low = df.loc[df['Effect'] == 'Proportion Mediated', '95% CI Low'].iloc[0]
pm_ci_high = df.loc[df['Effect'] == 'Proportion Mediated', '95% CI High'].iloc[0]

print(f"\nProportion Mediated (中介比例):")
print(f"  Estimate = {pm:.6f} ({pm*100:.2f}%)")
print(f"  95% CI = [{pm_ci_low:.6f}, {pm_ci_high:.6f}]")
if pm > 1:
    print(f"  ⚠ 注意：中介比例 > 100%，可能表示 suppressor effect")
else:
    print(f"  ✓ 中介比例在合理范围内")

rho = df.loc[df['Effect'] == 'rho threshold (SORT)', 'Estimate'].iloc[0]
rho_p = df.loc[df['Effect'] == 'rho threshold (SORT)', 'P-value'].iloc[0]

print(f"\nRho Threshold (敏感性分析):")
print(f"  Estimate = {rho:.6f}")
print(f"  P-value = {rho_p:.6f}")
print(f"  ✓ Rho 值合理: {0 <= rho <= 1}")

print("\n" + "=" * 80)
print("Table 2 Panel A (Univariate) Results")
print("=" * 80)

df_univ = pd.read_csv(os.path.join(tables_dir, "table2_panelA_univ_A1_to_FIV.csv"))
print("\nTable 2 Panel A (Univariate):")
print(df_univ.to_string(index=False))

print("\n" + "=" * 80)
print("Table 2 Panel A (Multivariate) Results")
print("=" * 80)

df_multiv = pd.read_csv(os.path.join(tables_dir, "table2_panelA_multiv_A1_to_FIV.csv"))
print("\nTable 2 Panel A (Multivariate):")
print(df_multiv.to_string(index=False))

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("✓ 所有表格已生成")
print(f"✓ Table 3 中 ACME = {acme:.6f} (p = {acme_p:.6f})")
print(f"✓ Table 3 中 Proportion Mediated = {pm:.2%}")
print(f"✓ Table 3 中 Rho threshold = {rho:.6f}")
print("\n修复完成！")
