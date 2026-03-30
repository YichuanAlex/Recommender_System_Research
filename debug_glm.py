import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from analysis_pipeline import read_data

df = read_data(r"c:\Users\24116\Desktop\混杂因素敏感分析\总数据.xlsx")
print("数据形状:", df.shape)
print("\n变量类型:")
print(df.dtypes)
print("\nFIV 的描述统计:")
print(df["FIV"].describe())
print("\nA1 的分布:")
print(df["A1"].value_counts())
print("\nHT 的分布:")
print(df["HT"].value_counts())

import statsmodels.api as sm
print("\n" + "="*80)
print("检查 FIV 与 A1 的关系")
print("="*80)
model_univ = smf.glm(formula="FIV ~ A1", data=df, family=sm.families.Gaussian()).fit()
print("\n单变量模型 (FIV ~ A1, Gaussian):")
print(model_univ.summary())
print("\n系数:")
print(model_univ.params)

print("\n" + "="*80)
print("检查多变量模型")
print("="*80)
covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
covs = [c for c in covs if c in df.columns]
formula_multiv = "FIV ~ A1 + " + " + ".join(covs)
print(f"\n公式：{formula_multiv}")

try:
    model_multiv = smf.glm(formula=formula_multiv, data=df, family=sm.families.Gaussian()).fit()
    print("\n多变量模型系数:")
    print(model_multiv.params)
    print("\n标准误:")
    print(model_multiv.bse)
except Exception as e:
    print(f"\n错误：{e}")

print("\n" + "="*80)
print("检查 FIV 的分布")
print("="*80)
print(f"FIV 唯一值：{sorted(df['FIV'].unique())}")
print(f"FIV 是否只有 0 和 1: {set(df['FIV'].unique()) == {0, 1}}")
