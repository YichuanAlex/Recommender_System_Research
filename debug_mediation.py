import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 读取数据
df = pd.read_excel("总数据.xlsx", engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]

# 数据预处理
for col in ["A1", "FIV", "HT", "sex", "Hyper", "Dia", "AF", "CAD", "CS", "IVT", "postAS"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ["age", "BaselineNIH", "SORT"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 填补缺失值
num_cols = ["age", "BaselineNIH", "SORT"]
for col in num_cols:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

bin_cols = ["A1", "FIV", "HT", "sex", "Hyper", "Dia", "AF", "CAD", "CS", "IVT"]
for col in bin_cols:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

if "postAS" in df.columns and df["postAS"].isna().any():
    df["postAS"] = df["postAS"].fillna(df["postAS"].mode().iloc[0])

df = df.dropna(subset=["A1", "FIV", "HT"])

print(f"样本量：{len(df)}")
print(f"\nA1 分布：{df['A1'].value_counts().to_dict()}")
print(f"FIV 分布：{df['FIV'].value_counts().to_dict()}")
print(f"HT 分布：{df['HT'].value_counts().to_dict()}")

# 检查 A1 和 FIV 的关系
print(f"\nA1 与 FIV 的交叉表:")
print(pd.crosstab(df['A1'], df['FIV']))

# 检查 FIV 和 HT 的关系
print(f"\nFIV 与 HT 的交叉表:")
print(pd.crosstab(df['FIV'], df['HT']))

# 检查 A1 和 HT 的关系
print(f"\nA1 与 HT 的交叉表:")
print(pd.crosstab(df['A1'], df['HT']))
