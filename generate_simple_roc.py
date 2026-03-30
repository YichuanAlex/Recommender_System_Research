"""
生成简单的 ROC 曲线图（不含 bootstrap 曲线）
论文标准格式
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
import statsmodels.api as sm
from scipy import stats

def fit_glm_binomial_robust(formula, df):
    """Fit GLM with robust standard errors"""
    model = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial())
    result = model.fit(cov_type='HC1')
    return result

def predict_prob_glm(model, df):
    """Predict probabilities"""
    return model.predict(df)

# 数据路径
data_path = r"c:\Users\24116\Desktop\混杂因素敏感分析\总数据.xlsx"
output_dir = r"c:\Users\24116\Desktop\混杂因素敏感分析\outputs\figures"

# 读取数据
print("读取数据...")
df = pd.read_excel(data_path)
print(f"数据维度: {df.shape}")

# 构建模型
print("\n构建 ROC 模型...")
covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
covs = [c for c in covs if c in df.columns]

formula1 = "HT ~ A1" + (" + " + " + ".join(covs) if covs else "")

print(f"模型公式: {formula1}")

# 拟合模型
res1 = fit_glm_binomial_robust(formula1, df)
p1 = predict_prob_glm(res1, df)
y = df["HT"].values

# 计算 ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y, p1)
auc_val = roc_auc_score(y, p1)

print(f"AUC: {auc_val:.3f}")

# Bootstrap 计算置信区间
print("\n计算 95% 置信区间 (Bootstrap, n=1000)...")
B = 1000
rng = np.random.default_rng(2026)
aucs_boot = []

for b in range(B):
    idx = rng.choice(np.arange(len(df)), len(df), replace=True)
    d = df.iloc[idx].reset_index(drop=True)
    r1_b = fit_glm_binomial_robust(formula1, d)
    p1b = predict_prob_glm(r1_b, d)
    yb = d["HT"].values
    try:
        auc_b = roc_auc_score(yb, p1b)
        aucs_boot.append(auc_b)
    except:
        pass

aucs_boot = np.array(aucs_boot)
ci_low = np.percentile(aucs_boot, 2.5)
ci_high = np.percentile(aucs_boot, 97.5)

print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

# 调用新的简单 ROC 曲线函数
from analysis_pipeline import plot_simple_roc

# 保存简单 ROC 曲线
out_path_simple = os.path.join(output_dir, "figure2_simple_roc.png")
print(f"\n生成简单 ROC 曲线...")
plot_simple_roc(
    y, 
    [p1], 
    ["Model 1"],
    [auc_val],
    [(ci_low, ci_high)],
    out_path_simple
)

print(f"✓ 简单 ROC 曲线已保存: {out_path_simple}")

# 同时保留原来的 bootstrap ROC 曲线
print("\n生成带 bootstrap 的 ROC 曲线...")
from analysis_pipeline import bootstrap_auc_models
roc_info = bootstrap_auc_models(df, B=1000, seed=2026, out_dir=output_dir)
print("✓ Bootstrap ROC 曲线已保存")

print("\n" + "="*60)
print("生成完成！")
print("="*60)
print(f"\n生成的文件:")
print(f"  1. figure2_simple_roc.png - 简单 ROC 曲线（无 bootstrap）")
print(f"  2. figure2_bootstrap_roc.png - 带 bootstrap 的 ROC 曲线")
print(f"\n所有文件位于: {output_dir}")
print("="*60)
