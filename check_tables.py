import pandas as pd
import numpy as np
import os

tables_dir = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\tables"

print("=" * 80)
print("自动化表格检查报告")
print("=" * 80)

for fname in ["table2_panelA_univ_A1_to_FIV.csv", 
              "table2_panelA_multiv_A1_to_FIV.csv", 
              "table3_mediation.csv"]:
    fpath = os.path.join(tables_dir, fname)
    if os.path.exists(fpath):
        print(f"\n检查文件：{fname}")
        print("-" * 80)
        df = pd.read_csv(fpath)
        print(f"行数：{len(df)}, 列数：{len(df.columns)}")
        print(f"\n列名：{list(df.columns)}")
        print(f"\n数据内容:")
        print(df.to_string())
        
        print(f"\n数值检查:")
        for col in df.columns:
            if df[col].dtype.kind in "biufc":
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}:")
                    print(f"    最小值：{vals.min()}, 最大值：{vals.max()}")
                    print(f"    是否有 0: {(vals == 0).any()}")
                    print(f"    是否有 inf: {np.isinf(vals).any()}")
                    print(f"    是否有 NaN: {vals.isna().any()}")
                    
                    zero_mask = (vals == 0)
                    if zero_mask.any():
                        print(f"    ⚠️  警告：{col} 列有 {zero_mask.sum()} 个值为 0")
                    
                    inf_mask = np.isinf(vals)
                    if inf_mask.any():
                        print(f"    ⚠️  警告：{col} 列有 {inf_mask.sum()} 个值为 inf")
    else:
        print(f"\n文件不存在：{fname}")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
