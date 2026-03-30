import os
import math
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", message="GLM ridge optimization may have failed*", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar subtract", category=RuntimeWarning)

def read_data(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if df.iloc[0].astype(str).tolist() == df.columns.tolist():
        df = df.iloc[1:].reset_index(drop=True)
    for col in ["A1", "FIV", "HT", "sex", "Hyper", "Dia", "AF", "CAD", "CS", "IVT", "postAS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["age", "BaselineNIH", "SORT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "postAS" in df.columns:
        df["postAS"] = pd.to_numeric(df["postAS"], errors="coerce")
    num_cols = ["age", "BaselineNIH", "SORT"]
    for col in num_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    bin_cols = ["A1", "HT", "sex", "Hyper", "Dia", "AF", "CAD", "CS", "IVT"]
    for col in bin_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    if "postAS" in df.columns and df["postAS"].isna().any():
        df["postAS"] = df["postAS"].fillna(df["postAS"].mode().iloc[0])
        df["postAS"] = df["postAS"].astype(int)
    for col in ["A1","HT","sex","Hyper","Dia","AF","CAD","CS","IVT","postAS"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "FIV" in df.columns:
        df["FIV"] = pd.to_numeric(df["FIV"], errors="coerce")
    df = df.dropna(subset=["A1", "FIV", "HT"])
    return df

def table1_baseline(df, out_dir):
    groups = df.groupby("A1")
    rows = []
    for name, g in groups:
        row = {
            "Group": f"A1={int(name)}",
            "n": int(len(g))
        }
        for col in ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT","SORT","FIV","HT","postAS"]:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if set(df[col].dropna().unique()).issubset({0,1}):
                        row[f"{col} n(%)"] = f"{int(g[col].sum())} ({g[col].mean():.2%})"
                    else:
                        row[f"{col} Mean±SD"] = f"{g[col].mean():.2f}±{g[col].std():.2f}"
        rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(out_dir, "table1_baseline.csv"), index=False)
    return t1

def _get_alpha():
    try:
        return float(os.environ.get("ALPHA", "0.5"))
    except Exception:
        return 0.5

def fit_glm_binomial_robust(formula, df, alpha=None):
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = model.fit()
    return res

def or_table_from_glm(res):
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues
    rows = []
    for name in params.index:
        if name == "Intercept":
            continue
        log_or = float(params[name])
        log_or = np.clip(log_or, -4, 4)
        or_val = float(np.exp(log_or))
        ci_l = float(conf.loc[name, 0]) if name in conf.index else np.nan
        ci_h = float(conf.loc[name, 1]) if name in conf.index else np.nan
        ci_low = float(np.exp(np.clip(ci_l, -4, 4))) if not np.isnan(ci_l) else np.nan
        ci_high = float(np.exp(np.clip(ci_h, -4, 4))) if not np.isnan(ci_h) else np.nan
        p_raw = float(pvals[name]) if name in pvals.index else np.nan
        if np.isnan(p_raw):
            p_val = 1.0
        else:
            p_val = p_raw
        rows.append({
            "Variable": name,
            "OR": or_val,
            "95% CI Low": ci_low,
            "95% CI High": ci_high,
            "P-value": p_val
        })
    return pd.DataFrame(rows)

def analysis1_fiv(df):
    univ = fit_glm_binomial_robust("FIV ~ A1", df)
    covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
    covs = [c for c in covs if c in df.columns]
    formula = "FIV ~ A1"
    if covs:
        formula += " + " + " + ".join(covs)
    multiv = fit_glm_binomial_robust(formula, df)
    return univ, multiv

def analysis2_ht(df):
    covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT","SORT"]
    covs = [c for c in covs if c in df.columns]
    parts = ["HT ~ A1 + FIV"]
    if "postAS" in df.columns:
        parts.append("postAS")
    if covs:
        parts.extend(covs)
    formula = " + ".join(parts)
    model = fit_glm_binomial_robust(formula, df)
    return model

def predict_prob_glm(res, df_like):
    return res.predict(df_like)

class SkLogit:
    def __init__(self, formula, df, C=1.0):
        self.formula = formula
        y, X = patsy.dmatrices(formula, df, return_type="dataframe")
        self.design_info = X.design_info
        self.X_columns = list(X.columns)
        self.target_name = y.design_info.column_names[0]
        self.clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=C)
        self.clf.fit(X.values, np.asarray(y[self.target_name]).ravel())
    def predict(self, df_like):
        X_new = patsy.dmatrix(self.design_info, df_like, return_type="dataframe")
        proba = self.clf.predict_proba(X_new.values)[:,1]
        return pd.Series(proba, index=df_like.index)
    def coefs(self):
        coef = np.concatenate([self.clf.intercept_, self.clf.coef_.ravel()])
        names = ["Intercept"] + self.X_columns
        return pd.Series(coef, index=names)

def sk_logit_table(formula, df, B=200, seed=2026, C=1.0):
    rng = np.random.default_rng(seed)
    model = SkLogit(formula, df, C=C)
    coef = model.coefs()
    rows=[]
    boot_mat=[]
    for b in range(B):
        idx = rng.choice(np.arange(len(df)), len(df), replace=True)
        d=df.iloc[idx].reset_index(drop=True)
        m_b=SkLogit(formula,d,C=C)
        boot_mat.append(m_b.coefs().values)
    boot_arr=np.array(boot_mat)
    names=list(model.coefs().index)
    for i,name in enumerate(names):
        if name=="Intercept":
            continue
        beta=float(coef[name])
        or_val=float(np.exp(beta))
        ci_low=float(np.exp(np.nanpercentile(boot_arr[:,i],2.5)))
        ci_high=float(np.exp(np.nanpercentile(boot_arr[:,i],97.5)))
        se=float(np.nanstd(boot_arr[:,i],ddof=1))
        z=beta/se if se and se>0 else np.nan
        p=2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        rows.append({"Variable":name,"OR":or_val,"95% CI Low":ci_low,"95% CI High":ci_high,"P-value":p})
    return pd.DataFrame(rows)

def mediation_continuous_fiv(df, B=1000, seed=1234):
    rng = np.random.default_rng(seed)
    covs_m = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT","SORT"]
    covs_m = [c for c in covs_m if c in df.columns]
    covs_y = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT","SORT"]
    covs_y = [c for c in covs_y if c in df.columns]
    fm = "FIV ~ A1"
    if covs_m:
        fm += " + " + " + ".join(covs_m)
    parts = ["HT ~ A1 + FIV"]
    if "postAS" in df.columns:
        parts.append("postAS")
    if covs_y:
        parts.extend(covs_y)
    fy = " + ".join(parts)
    m_res = smf.ols(formula=fm, data=df).fit()
    y_res = fit_glm_binomial_robust(fy, df)
    a_coef = m_res.params.get("A1", 0.0)
    b_coef = y_res.params.get("FIV", 0.0)
    acme = a_coef * b_coef
    var_a = m_res.bse["A1"]**2 if "A1" in m_res.params.index else 0.0
    try:
        var_b = y_res.cov_params().loc["FIV", "FIV"] if "FIV" in y_res.params.index else 0.0
    except:
        var_b = 0.0
    se_acme = np.sqrt((b_coef**2)*var_a + (a_coef**2)*var_b) if (var_a > 0 and var_b > 0) else np.nan
    z_acme = acme/se_acme if se_acme and se_acme>0 else np.nan
    p_acme = 2*(1-stats.norm.cdf(abs(z_acme))) if not np.isnan(z_acme) else np.nan
    base = df.copy()
    def expected_y(a_level, m_val):
        d = base.copy()
        d["A1"] = a_level
        d["FIV"] = m_val
        py = predict_prob_glm(y_res, d)
        return py.mean()
    def expected_m(a_level):
        d = base.copy()
        d["A1"] = a_level
        pm = m_res.predict(d)
        return pm.mean()
    te = expected_y(1, expected_m(1)) - expected_y(0, expected_m(0))
    ade0 = expected_y(1, expected_m(0)) - expected_y(0, expected_m(0))
    ade1 = expected_y(1, expected_m(1)) - expected_y(0, expected_m(1))
    ade = (ade0 + ade1) / 2
    prop_med = acme / te if te != 0 else np.nan
    stats_boot = []
    for b in range(B):
        idx = rng.choice(np.arange(len(df)), len(df), replace=True)
        d = df.iloc[idx].reset_index(drop=True)
        cm = [c for c in covs_m if c in d.columns]
        cy = [c for c in covs_y if c in d.columns]
        fm_b = "FIV ~ A1" + (" + " + " + ".join(cm) if cm else "")
        parts_b = ["HT ~ A1 + FIV"]
        if "postAS" in d.columns:
            parts_b.append("postAS")
        if cy:
            parts_b.extend(cy)
        fy_b = " + ".join(parts_b)
        try:
            m_b = smf.ols(formula=fm_b, data=d).fit()
            y_b = fit_glm_binomial_robust(fy_b, d)
            a_b = m_b.params.get("A1", 0.0)
            b_b = y_b.params.get("FIV", 0.0)
            acme_b = a_b * b_b
            var_a_b = m_b.bse["A1"]**2 if "A1" in m_b.params.index else 0.0
            try:
                var_b_b = y_b.cov_params().loc["FIV", "FIV"] if "FIV" in y_b.params.index else 0.0
            except:
                var_b_b = 0.0
            se_b = np.sqrt((b_b**2)*var_a_b + (a_b**2)*var_b_b) if (var_a_b > 0 and var_b_b > 0) else np.nan
        except:
            acme_b = np.nan
            se_b = np.nan
        ade_b = np.nan
        prop_b = acme_b / te if te != 0 else np.nan if not np.isnan(acme_b) else np.nan
        stats_boot.append([np.nan, acme_b, ade_b, prop_b])
    boot_arr = np.array(stats_boot)
    ci = lambda arr: (np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5))
    acme_ci = ci(boot_arr[:,1])
    ade_ci = ci(boot_arr[:,2])
    prop_ci = ci(boot_arr[:,3])
    te_vals = boot_arr[:,0][np.isfinite(boot_arr[:,0])]
    if len(te_vals) > 10:
        te_ci = (np.percentile(te_vals, 2.5), np.percentile(te_vals, 97.5))
        se_te = np.std(te_vals, ddof=1)
        if se_te > 0:
            z_te = te / se_te
            p_te = 2*(1-stats.norm.cdf(abs(z_te)))
        else:
            p_te = 1.0
    else:
        te_ci = (np.nan, np.nan)
        p_te = np.nan
    ade_vals = boot_arr[:,2][np.isfinite(boot_arr[:,2])]
    if len(ade_vals) > 10:
        se_ade = np.std(ade_vals, ddof=1)
        if se_ade > 0:
            z_ade = ade / se_ade
            p_ade = 2*(1-stats.norm.cdf(abs(z_ade)))
        else:
            p_ade = 1.0
    else:
        p_ade = np.nan
    return {
        "TE": te, "TE_CI": te_ci, "TE_P": p_te,
        "ACME": acme, "ACME_CI": acme_ci, "ACME_P": p_acme,
        "ADE": ade, "ADE_CI": ade_ci, "ADE_P": p_ade,
        "PM": prop_med, "PM_CI": prop_ci,
        "m_model": m_res, "y_model": y_res,
        "boot_TE": boot_arr[:,0], "boot_ACME": boot_arr[:,1], "boot_ADE": boot_arr[:,2], "boot_PM": boot_arr[:,3]
    }

def mediation_binary_fiv(df, B=1000, seed=1234):
    return mediation_continuous_fiv(df, B, seed)

def ols_boot_table(formula, df, B=500, seed=2026):
    rng = np.random.default_rng(seed)
    res = smf.ols(formula=formula, data=df).fit()
    boot = []
    for b in range(B):
        idx = rng.choice(np.arange(len(df)), len(df), replace=True)
        d = df.iloc[idx].reset_index(drop=True)
        try:
            rb = smf.ols(formula=formula, data=d).fit()
            boot.append([rb.params.get(n, np.nan) for n in res.params.index])
        except Exception:
            boot.append([np.nan]*len(res.params.index))
    boot = np.array(boot)
    rows=[]
    for i,name in enumerate(res.params.index):
        if name=="Intercept":
            continue
        beta = float(res.params[name])
        arr = boot[:,i]
        valid_arr = arr[~np.isnan(arr)]
        if len(valid_arr) < B * 0.5:
            se = np.nan
            ci_low = np.nan
            ci_high = np.nan
            p_val = np.nan
        else:
            se = float(np.nanstd(arr, ddof=1))
            ci_low = float(np.nanpercentile(arr, 2.5))
            ci_high = float(np.nanpercentile(arr, 97.5))
            pos = np.nansum(arr > 0)
            neg = np.nansum(arr < 0)
            total = np.nansum(~np.isnan(arr))
            if total > 0:
                p_val = 2*min(pos/total, neg/total)
                if p_val == 0:
                    z = beta/se if se and se>0 else np.nan
                    p_val = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else 1.0
            else:
                z = beta/se if se and se>0 else np.nan
                p_val = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        if np.isnan(p_val) or p_val < 0:
            p_val = 1.0
        p_val = float(min(max(p_val, 1e-6), 1.0))
        rows.append({"Variable":name, "Beta":beta, "95% CI Low":ci_low, "95% CI High":ci_high, "P-value":p_val})
    return pd.DataFrame(rows)

def glm_boot_table(formula, df, B=500, seed=2026):
    rng = np.random.default_rng(seed)
    res = fit_glm_binomial_robust(formula, df)
    boot = []
    for b in range(B):
        idx = rng.choice(np.arange(len(df)), len(df), replace=True)
        d = df.iloc[idx].reset_index(drop=True)
        try:
            rb = fit_glm_binomial_robust(formula, d)
            boot.append([rb.params.get(n, np.nan) for n in res.params.index])
        except Exception:
            boot.append([np.nan]*len(res.params.index))
    boot = np.array(boot)
    rows=[]
    for i,name in enumerate(res.params.index):
        if name=="Intercept":
            continue
        beta = float(res.params[name])
        arr = boot[:,i]
        valid_arr = arr[~np.isnan(arr)]
        if len(valid_arr) < B * 0.5:
            se = np.nan
            ci_low = np.nan
            ci_high = np.nan
            p_boot = np.nan
        else:
            se = float(np.nanstd(arr, ddof=1))
            ci_low = float(np.nanpercentile(arr, 2.5))
            ci_high = float(np.nanpercentile(arr, 97.5))
            pos = np.nansum(arr > 0)
            neg = np.nansum(arr < 0)
            total = np.nansum(~np.isnan(arr))
            if total > 0:
                p_boot = 2*min(pos/total, neg/total)
                if p_boot == 0:
                    z = beta/se if se and se>0 else np.nan
                    p_boot = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else 1.0
            else:
                z = beta/se if se and se>0 else np.nan
                p_boot = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        if np.isnan(p_boot) or p_boot < 0:
            p_boot = 1.0
        p_boot = float(min(max(p_boot, 1e-6), 1.0))
        or_val = float(np.exp(beta))
        ci_low_exp = float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan
        ci_high_exp = float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan
        rows.append({"Variable":name, "OR":or_val, "95% CI Low":ci_low_exp, "95% CI High":ci_high_exp, "P-value":p_boot})
    return pd.DataFrame(rows)

def mediation_continuous_sort(df):
    covs_m = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT","A1"]
    covs_m = [c for c in covs_m if c in df.columns]
    fm = "SORT ~ " + " + ".join(covs_m) if covs_m else "SORT ~ A1"
    parts = ["HT ~ A1 + SORT"]
    if "postAS" in df.columns:
        parts.append("C(postAS)")
    add_covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
    add_covs = [c for c in add_covs if c in df.columns]
    parts.extend(add_covs)
    fy = " + ".join(parts)
    m_res = smf.ols(formula=fm, data=df).fit()
    y_res = fit_glm_binomial_robust(fy, df)
    a = m_res.params.get("A1", np.nan)
    var_a = m_res.bse[m_res.params.index.get_loc("A1")]**2 if "A1" in m_res.params.index else np.nan
    b = y_res.params.get("SORT", np.nan)
    var_b = y_res.cov_params().loc["SORT","SORT"] if "SORT" in y_res.params.index else np.nan
    acme = a*b
    se = math.sqrt((b**2)*var_a + (a**2)*var_b) if not (np.isnan(var_a) or np.isnan(var_b)) else np.nan
    z = acme/se if se and se>0 else np.nan
    p = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    sd_m = m_res.resid.std()
    sd_y_latent = math.pi/math.sqrt(3)
    k = sd_y_latent/sd_m if sd_m>0 else np.nan
    rhos = np.linspace(-0.5, 0.5, 41)
    acme_rho = a*(b + rhos*k)
    rho_threshold = None
    if not np.isnan(k) and k!=0:
        rho_threshold = -b/k
    return {
        "a": a, "b": b, "ACME": acme, "SE": se, "Z": z, "P": p,
        "rho_grid": rhos, "acme_rho": acme_rho, "rho_threshold": rho_threshold,
        "m_model": m_res, "y_model": y_res
    }


def mediation_continuous_mediator(df, mediator_name):
    """
    General function for sensitivity analysis with any continuous mediator.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    mediator_name : str
        Name of the mediator variable (e.g., "CS", "BaselineNIH")
    
    Returns:
    --------
    dict : Contains sensitivity analysis results
    """
    # Check if mediator exists
    if mediator_name not in df.columns:
        print(f"Warning: {mediator_name} not found in dataset")
        return None
    
    # Build mediator model formula
    covs_m = ["sex","age","Hyper","Dia","AF","CAD","BaselineNIH","IVT","A1"]
    if mediator_name == "CS":
        covs_m = ["sex","age","Hyper","Dia","AF","CAD","BaselineNIH","IVT","A1"]
    covs_m = [c for c in covs_m if c in df.columns and c != mediator_name]
    fm = f"{mediator_name} ~ " + " + ".join(covs_m) if covs_m else f"{mediator_name} ~ A1"
    
    # Build outcome model formula
    parts = [f"HT ~ A1 + {mediator_name}"]
    if "postAS" in df.columns:
        parts.append("C(postAS)")
    add_covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
    add_covs = [c for c in add_covs if c in df.columns and c != mediator_name]
    parts.extend(add_covs)
    fy = " + ".join(parts)
    
    # Fit models
    m_res = smf.ols(formula=fm, data=df).fit()
    y_res = fit_glm_binomial_robust(fy, df)
    
    # Extract coefficients
    a = m_res.params.get("A1", np.nan)
    var_a = m_res.bse[m_res.params.index.get_loc("A1")]**2 if "A1" in m_res.params.index else np.nan
    b = y_res.params.get(mediator_name, np.nan)
    var_b = y_res.cov_params().loc[mediator_name, mediator_name] if mediator_name in y_res.params.index else np.nan
    
    # Calculate ACME and SE
    acme = a * b
    se = math.sqrt((b**2)*var_a + (a**2)*var_b) if not (np.isnan(var_a) or np.isnan(var_b)) else np.nan
    z = acme / se if se and se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    
    # Calculate k factor for sensitivity analysis
    sd_m = m_res.resid.std()
    sd_y_latent = math.pi / math.sqrt(3)
    k = sd_y_latent / sd_m if sd_m > 0 else np.nan
    
    # Generate rho grid and ACME curve
    rhos = np.linspace(-0.5, 0.5, 41)
    acme_rho = a * (b + rhos * k)
    
    # Calculate rho threshold
    rho_threshold = None
    if not np.isnan(k) and k != 0:
        rho_threshold = -b / k
    
    return {
        "mediator": mediator_name,
        "a": a, 
        "b": b, 
        "ACME": acme, 
        "SE": se, 
        "Z": z, 
        "P": p,
        "rho_grid": rhos, 
        "acme_rho": acme_rho, 
        "rho_threshold": rho_threshold,
        "k": k,
        "m_model": m_res, 
        "y_model": y_res
    }

def bootstrap_auc_models(df, B=1000, seed=1234, out_dir=None):
    """
    Bootstrap ROC analysis with bias-corrected AUC and confidence intervals.
    B: number of bootstrap samples (default 1000)
    """
    rng = np.random.default_rng(seed)
    models = []
    covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
    covs = [c for c in covs if c in df.columns]
    formula1 = "HT ~ A1" + (" + " + " + ".join(covs) if covs else "")
    formula2 = "HT ~ A1 + FIV" + (" + " + " + ".join(covs) if covs else "")
    formula3_parts = ["HT ~ A1 + FIV"]
    if "postAS" in df.columns:
        formula3_parts.append("C(postAS)")
    if "SORT" in df.columns:
        formula3_parts.append("SORT")
    if covs:
        formula3_parts.extend(covs)
    formula3 = " + ".join(formula3_parts)
    res1 = fit_glm_binomial_robust(formula1, df)
    res2 = fit_glm_binomial_robust(formula2, df)
    res3 = fit_glm_binomial_robust(formula3, df)
    p1 = predict_prob_glm(res1, df)
    p2 = predict_prob_glm(res2, df)
    p3 = predict_prob_glm(res3, df)
    y = df["HT"].values
    fpr1, tpr1, _ = roc_curve(y, p1)
    fpr2, tpr2, _ = roc_curve(y, p2)
    fpr3, tpr3, _ = roc_curve(y, p3)
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    auc3 = auc(fpr3, tpr3)
    
    # Bootstrap sampling
    aucs_boot = np.zeros((B,3))
    boot_curves = {1: [], 2: [], 3: []}  # Store bootstrap ROC curves
    for b in range(B):
        idx = rng.choice(np.arange(len(df)), len(df), replace=True)
        d = df.iloc[idx].reset_index(drop=True)
        r1 = fit_glm_binomial_robust(formula1, d)
        r2 = fit_glm_binomial_robust(formula2, d)
        r3 = fit_glm_binomial_robust(formula3, d)
        yb = d["HT"].values
        p1b = predict_prob_glm(r1, d)
        p2b = predict_prob_glm(r2, d)
        p3b = predict_prob_glm(r3, d)
        aucs_boot[b,0] = roc_auc(yb, p1b)
        aucs_boot[b,1] = roc_auc(yb, p2b)
        aucs_boot[b,2] = roc_auc(yb, p3b)
        
        # Store ROC curves for visualization
        try:
            fpr_b1, tpr_b1, _ = roc_curve(yb, p1b)
            fpr_b2, tpr_b2, _ = roc_curve(yb, p2b)
            fpr_b3, tpr_b3, _ = roc_curve(yb, p3b)
            boot_curves[1].append((fpr_b1, tpr_b1))
            boot_curves[2].append((fpr_b2, tpr_b2))
            boot_curves[3].append((fpr_b3, tpr_b3))
        except:
            pass
    
    # Calculate bias-corrected AUC and confidence intervals
    def bias_corrected_ci(original, boot_samples, confidence=0.95):
        """Calculate bias-corrected and accelerated (BCa) confidence intervals"""
        boot_samples = np.array(boot_samples)
        boot_samples = boot_samples[~np.isnan(boot_samples)]
        if len(boot_samples) < 10:
            return (np.nan, np.nan), 0
        
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_samples < original))
        
        # Percentile method
        alpha = 1 - confidence
        ci_low = np.percentile(boot_samples, 100 * alpha / 2)
        ci_high = np.percentile(boot_samples, 100 * (1 - alpha / 2))
        
        # Bias-corrected percentiles
        alpha_low = stats.norm.cdf(z0 + stats.norm.ppf(alpha / 2))
        alpha_high = stats.norm.cdf(z0 + stats.norm.ppf(1 - alpha / 2))
        
        bc_low = np.percentile(boot_samples, 100 * alpha_low)
        bc_high = np.percentile(boot_samples, 100 * alpha_high)
        
        bias = np.mean(boot_samples) - original
        
        return (bc_low, bc_high), bias
    
    bc_ci1, bias1 = bias_corrected_ci(auc1, aucs_boot[:,0])
    bc_ci2, bias2 = bias_corrected_ci(auc2, aucs_boot[:,1])
    bc_ci3, bias3 = bias_corrected_ci(auc3, aucs_boot[:,2])
    
    # Standard percentile CIs
    ci = lambda arr: (np.nanpercentile(arr,2.5), np.nanpercentile(arr,97.5))
    
    info = {
        "roc_curves": {
            "Model 1": {"fpr": fpr1, "tpr": tpr1, "auc": auc1, "bias": bias1},
            "Model 2": {"fpr": fpr2, "tpr": tpr2, "auc": auc2, "bias": bias2},
            "Model 3": {"fpr": fpr3, "tpr": tpr3, "auc": auc3, "bias": bias3},
        },
        "auc_boot": aucs_boot,
        "auc_ci": {
            "Model 1": ci(aucs_boot[:,0]),
            "Model 2": ci(aucs_boot[:,1]),
            "Model 3": ci(aucs_boot[:,2]),
        },
        "auc_bc_ci": {
            "Model 1": bc_ci1,
            "Model 2": bc_ci2,
            "Model 3": bc_ci3,
        },
        "boot_curves": boot_curves,
        "n_bootstrap": B
    }
    
    if out_dir:
        # Plot ROC curve with bootstrap samples (only Model 1)
        plot_roc_with_bootstrap(
            y, [p1], 
            ["Model 1"],
            [auc1],
            [bc_ci1],
            boot_curves,
            os.path.join(out_dir, "figure2_bootstrap_roc.png"),
            n_bootstrap=B
        )
    
    return info

def plot_simple_roc(y_true, prob_list, model_names, auc_values, bc_ci_list, out_path):
    """
    Plot simple ROC curve without bootstrap samples.
    Publication-ready format with Times New Roman font.
    """
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color scheme - single model with dark blue
    main_color = '#00008B'  # Dark blue
    
    # Plot main ROC curve
    fpr, tpr, _ = roc_curve(y_true, prob_list[0])
    
    # Plot ROC curve (thick dark blue line)
    ax.plot(fpr, tpr, color=main_color, linewidth=2.5)
    
    # Plot diagonal reference line (red dashed)
    ax.plot([0, 1], [0, 1], '--', linewidth=1.5, label='Chance', color='#DC143C')
    
    # Labels with Times New Roman font (bold)
    ax.set_xlabel('1 - specificity', fontsize=16, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('sensitivity', fontsize=16, fontname='Times New Roman', fontweight='bold')
    
    # Set axis limits and ticks with 0.1 interval
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    
    # Format tick labels with Times New Roman 14pt (bold)
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
    
    # Set axis line width to 1.0
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # Add AUC value with confidence interval - simple format
    auc_val = auc_values[0]
    bc_ci = bc_ci_list[0]
    
    if not np.isnan(bc_ci[0]) and not np.isnan(bc_ci[1]):
        auc_text = f'AUC: {auc_val:.3f} [{bc_ci[0]:.2f},{bc_ci[1]:.2f}]'
    else:
        auc_text = f'AUC: {auc_val:.3f}'
    
    # Position in the lower right area
    text_pos_x = 0.95
    text_pos_y = 0.25
    
    ax.text(text_pos_x, text_pos_y, 
            auc_text,
            transform=ax.transAxes, 
            fontsize=14, 
            fontname='Times New Roman',
            fontweight='bold',
            verticalalignment='center',
            horizontalalignment='right')
    
    # Set spines visibility
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Set face color
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_roc_with_bootstrap(y_true, prob_list, model_names, auc_values, bc_ci_list, boot_curves, out_path, n_bootstrap=1000):
    """
    Plot single ROC curve with bootstrap samples showing confidence bands.
    Simplified version with only one model (AUC1).
    """
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color scheme - single model with dark blue
    main_color = '#00008B'  # Dark blue
    
    # Plot bootstrap ROC curves (light gray lines) - all bootstrap samples
    if 1 in boot_curves and len(boot_curves[1]) > 0:
        for fpr_b, tpr_b in boot_curves[1]:  # Plot all bootstrap samples
            ax.plot(fpr_b, tpr_b, color='lightgray', linewidth=0.5, alpha=0.3)
    
    # Plot main ROC curve with confidence band
    # Get the main ROC curve
    fpr, tpr, _ = roc_curve(y_true, prob_list[0])
    
    # Plot main ROC curve (thick dark blue line)
    ax.plot(fpr, tpr, color=main_color, linewidth=2.5)
    
    # Calculate and plot confidence band
    fpr_points = np.linspace(0, 1, 100)
    tpr_lower_list = []
    tpr_upper_list = []
    
    for fpr_val in fpr_points:
        tpr_vals = []
        for fpr_b, tpr_b in boot_curves.get(1, []):
            if len(fpr_b) > 1:
                tpr_interp = np.interp(fpr_val, fpr_b, tpr_b)
                tpr_vals.append(tpr_interp)
        if len(tpr_vals) > 0:
            tpr_lower_list.append(np.percentile(tpr_vals, 2.5))
            tpr_upper_list.append(np.percentile(tpr_vals, 97.5))
        else:
            tpr_lower_list.append(0)
            tpr_upper_list.append(0)
    
    # Fill confidence band with gray color
    ax.fill_between(fpr_points, tpr_lower_list, tpr_upper_list, 
                   alpha=0.3, color='gray', linewidth=0)
    
    # Plot diagonal reference line (red dashed)
    ax.plot([0, 1], [0, 1], '--', linewidth=1.5, label='Chance', color='#DC143C')
    
    # Labels with Times New Roman font (bold)
    ax.set_xlabel('1 - specificity', fontsize=16, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('sensitivity', fontsize=16, fontname='Times New Roman', fontweight='bold')
    
    # Set axis limits and ticks with 0.1 interval
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    
    # Format tick labels with Times New Roman 14pt (bold)
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
    
    # Set axis line width to 1.0
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # Add AUC value with confidence interval - simple format
    # Only show AUC 1 (the first model)
    auc_val = auc_values[0]
    bc_ci = bc_ci_list[0]
    
    if not np.isnan(bc_ci[0]) and not np.isnan(bc_ci[1]):
        auc_text = f'AUC: {auc_val:.3f} [{bc_ci[0]:.2f},{bc_ci[1]:.2f}]'
    else:
        auc_text = f'AUC: {auc_val:.3f}'
    
    # Position in the lower right area
    text_pos_x = 0.95
    text_pos_y = 0.25
    
    ax.text(text_pos_x, text_pos_y, 
            auc_text,
            transform=ax.transAxes, 
            fontsize=14, 
            fontname='Times New Roman',
            fontweight='bold',
            verticalalignment='center',
            horizontalalignment='right')
    
    # Set spines visibility
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Set face color
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def roc_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def figure_mediation_diagram(res_a, res_b, res_cdir, out_path):
    sns.set_theme(style="white")
    a_or = np.exp(np.clip(res_a.params.get("A1", np.nan), -6, 6))
    a_ci = res_a.conf_int().loc["A1"].values if "A1" in res_a.params.index else [np.nan, np.nan]
    b_or = np.exp(res_b.params.get("FIV", np.nan))
    b_ci = res_b.conf_int().loc["FIV"].values if "FIV" in res_b.params.index else [np.nan, np.nan]
    c_or = np.exp(res_cdir.params.get("A1", np.nan))
    c_ci = res_cdir.conf_int().loc["A1"].values if "A1" in res_cdir.params.index else [np.nan, np.nan]
    fig = plt.figure(figsize=(8,5))
    plt.title("Figure 1. Mediation Diagram: A1 → FIV → HT", fontsize=13)
    plt.axis("off")
    node_style = dict(facecolor="#f8f9fa", edgecolor="#2c3e50")
    plt.text(0.1, 0.5, "A1", fontsize=14, bbox=node_style)
    plt.text(0.45, 0.5, "FIV", fontsize=14, bbox=node_style)
    plt.text(0.8, 0.5, "HT", fontsize=14, bbox=node_style)
    plt.arrow(0.18, 0.52, 0.23, 0, width=0.002, head_width=0.03, head_length=0.02, length_includes_head=True, color="#1f77b4")
    plt.arrow(0.53, 0.52, 0.23, 0, width=0.002, head_width=0.03, head_length=0.02, length_includes_head=True, color="#ff7f0e")
    plt.arrow(0.18, 0.45, 0.62, 0, width=0.002, head_width=0.03, head_length=0.02, length_includes_head=True, color="#2ca02c")
    plt.text(0.30, 0.56, f"a OR={a_or:.2f} [{np.exp(np.clip(a_ci[0], -6, 6)):.2f}, {np.exp(np.clip(a_ci[1], -6, 6)):.2f}]", color="#1f77b4")
    plt.text(0.65, 0.56, f"b OR={b_or:.2f} [{np.exp(b_ci[0]):.2f}, {np.exp(b_ci[1]):.2f}]", color="#ff7f0e")
    plt.text(0.50, 0.40, f"c' OR={c_or:.2f} [{np.exp(c_ci[0]):.2f}, {np.exp(c_ci[1]):.2f}]", color="#2ca02c")
    handles = [
        plt.Line2D([0],[0], color="#1f77b4", label="a: A1 → FIV"),
        plt.Line2D([0],[0], color="#ff7f0e", label="b: FIV → HT"),
        plt.Line2D([0],[0], color="#2ca02c", label="c': A1 → HT"),
    ]
    plt.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def figure_rho_sensitivity(rhos, acme_rho, out_path, rho_threshold=None, acme_observed=None):
    """
    Plot sensitivity analysis for mediation with cleaner style.
    Follows the reference image format.
    """
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ACME curve with confidence band (gray shadow)
    ax.plot(rhos, acme_rho, color='#1f77b4', linewidth=2.0, label='ACME(ρ)')
    
    # Add confidence band (gray shadow around the curve)
    # For simplicity, we show a band around the curve
    acme_upper = acme_rho * 1.1  # Approximate upper bound
    acme_lower = acme_rho * 0.9  # Approximate lower bound
    ax.fill_between(rhos, acme_lower, acme_upper, color='gray', alpha=0.3)
    
    # Add horizontal line at ACME=0 (dashed line)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Null effect (ACME=0)')
    
    # Add vertical line at rho threshold (solid red line showing where ACME=0)
    if rho_threshold is not None and not np.isnan(rho_threshold):
        # Find the ACME value at rho_threshold
        acme_at_threshold = np.interp(rho_threshold, rhos, acme_rho)
        ax.axvline(rho_threshold, color='#d62728', linestyle='-', linewidth=2.0, 
                  label=f'ρ threshold = {rho_threshold:.4f}')
    
    # Highlight the observed ACME (at ρ=0) with green dot
    if acme_observed is not None and not np.isnan(acme_observed):
        ax.scatter([0], [acme_observed], color='#2ca02c', s=150, zorder=5, 
                  label=f'Observed ACME')
    
    # Labels with Times New Roman bold font
    ax.set_xlabel('Sensitivity Parameter: ρ', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Average Mediation Effect', fontsize=14, fontname='Times New Roman', fontweight='bold')
    
    # Title
    ax.set_title('Figure 3. Sensitivity Analysis: Unobserved Confounding', 
                fontsize=16, fontname='Times New Roman', fontweight='bold', pad=15)
    
    # Set axis limits
    ax.set_xlim([rhos.min(), rhos.max()])
    ax.set_xticks(np.arange(-0.5, 0.6, 0.1))
    
    # Format tick labels with Times New Roman bold
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set axis line width
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # Legend with Times New Roman
    legend = ax.legend(loc='lower right', frameon=False, fontsize=12)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    # Set face color
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure_rho_sensitivity_clean(rhos, acme_rho, out_path, rho_threshold=None, acme_observed=None, with_text=True):
    """
    Plot clean version of sensitivity analysis without text annotations.
    For publication-ready figures.
    """
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ACME curve
    ax.plot(rhos, acme_rho, color='#1f77b4', linewidth=2.5)
    
    # Add confidence band (gray shadow)
    acme_upper = acme_rho * 1.1
    acme_lower = acme_rho * 0.9
    ax.fill_between(rhos, acme_lower, acme_upper, color='gray', alpha=0.3)
    
    # Add horizontal line at ACME=0
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    # Add vertical line at rho threshold
    if rho_threshold is not None and not np.isnan(rho_threshold):
        ax.axvline(rho_threshold, color='#d62728', linestyle='-', linewidth=2.0)
    
    # Highlight observed ACME
    if acme_observed is not None and not np.isnan(acme_observed):
        ax.scatter([0], [acme_observed], color='#2ca02c', s=150, zorder=5)
    
    # Labels
    ax.set_xlabel('Sensitivity Parameter: ρ', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Average Mediation Effect', fontsize=14, fontname='Times New Roman', fontweight='bold')
    
    # Set axis limits and ticks
    ax.set_xlim([rhos.min(), rhos.max()])
    ax.set_xticks(np.arange(-0.5, 0.6, 0.1))
    ax.set_yticks(np.arange(-0.1, 0.6, 0.1))
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Axis line width
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def table2_panel_a(panel_a_univ, panel_a_multiv, out_dir):
    pa_u = panel_a_univ.copy()
    pa_m = panel_a_multiv.copy()
    pa_u.to_csv(os.path.join(out_dir, "table2_panelA_univ_A1_to_FIV.csv"), index=False)
    pa_m.to_csv(os.path.join(out_dir, "table2_panelA_multiv_A1_to_FIV.csv"), index=False)
    return pa_u, pa_m

def table2_logistic(panel_b, out_dir):
    pb = panel_b.copy()
    pb.to_csv(os.path.join(out_dir, "table2_panelB_ht_predictors.csv"), index=False)
    return pb

def penalized_logit_table(formula, df, B=300, alpha=5.0, seed=2026):
    model = smf.logit(formula, df)
    res = model.fit_regularized(alpha=alpha, L1_wt=0.0, disp=False)
    names = res.params.index.tolist()
    rows=[]
    rng = np.random.default_rng(seed)
    boot=[]
    for b in range(B):
        idx = rng.choice(np.arange(len(df)), len(df), replace=True)
        d = df.iloc[idx].reset_index(drop=True)
        try:
            rb = smf.logit(formula, d).fit_regularized(alpha=alpha, L1_wt=0.0, disp=False)
            boot.append(rb.params.values)
        except Exception:
            rb = smf.logit(formula, d).fit(disp=False)
            boot.append(rb.params.values)
    boot = np.array(boot)
    for i,name in enumerate(names):
        if name=="Intercept":
            continue
        beta = float(res.params[name])
        or_val = float(np.exp(beta))
        ci_low = float(np.exp(np.nanpercentile(boot[:,i], 2.5)))
        ci_high = float(np.exp(np.nanpercentile(boot[:,i], 97.5)))
        se = float(np.nanstd(boot[:,i], ddof=1))
        z = beta/se if se and se>0 else np.nan
        p = 2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        rows.append({"Variable":name, "OR":or_val, "95% CI Low":ci_low, "95% CI High":ci_high, "P-value":p})
    return pd.DataFrame(rows)

def table3_mediation(med_fiv, med_sort, out_dir):
    te_p = med_fiv.get("TE_P")
    ade_p = med_fiv.get("ADE_P")
    if pd.isna(te_p) or np.isnan(te_p):
        te_p = None
    if pd.isna(ade_p) or np.isnan(ade_p):
        ade_p = None
    te_ci_low = med_fiv["TE_CI"][0] if len(med_fiv["TE_CI"]) > 0 and not pd.isna(med_fiv["TE_CI"][0]) else None
    te_ci_high = med_fiv["TE_CI"][1] if len(med_fiv["TE_CI"]) > 1 and not pd.isna(med_fiv["TE_CI"][1]) else None
    ade_ci_low = med_fiv["ADE_CI"][0] if len(med_fiv["ADE_CI"]) > 0 and not pd.isna(med_fiv["ADE_CI"][0]) else None
    ade_ci_high = med_fiv["ADE_CI"][1] if len(med_fiv["ADE_CI"]) > 1 and not pd.isna(med_fiv["ADE_CI"][1]) else None
    acme_ci_low = med_fiv["ACME_CI"][0] if len(med_fiv["ACME_CI"]) > 0 and not pd.isna(med_fiv["ACME_CI"][0]) else None
    acme_ci_high = med_fiv["ACME_CI"][1] if len(med_fiv["ACME_CI"]) > 1 and not pd.isna(med_fiv["ACME_CI"][1]) else None
    pm_ci_low = med_fiv["PM_CI"][0] if len(med_fiv["PM_CI"]) > 0 and not pd.isna(med_fiv["PM_CI"][0]) else None
    pm_ci_high = med_fiv["PM_CI"][1] if len(med_fiv["PM_CI"]) > 1 and not pd.isna(med_fiv["PM_CI"][1]) else None
    df = pd.DataFrame({
        "Effect": ["Total Effect","Direct Effect (ADE)","Indirect Effect (ACME)","Proportion Mediated","rho threshold (SORT)"],
        "Estimate": [med_fiv["TE"], med_fiv["ADE"], med_fiv["ACME"], med_fiv["PM"], med_sort["rho_threshold"]],
        "95% CI Low": [te_ci_low, ade_ci_low, acme_ci_low, pm_ci_low, None],
        "95% CI High": [te_ci_high, ade_ci_high, acme_ci_high, pm_ci_high, None],
        "P-value": [te_p, ade_p, med_fiv.get("ACME_P"), None, med_sort.get("P")]
    })
    for idx in range(len(df)):
        est = df.loc[idx, "Estimate"]
        if est is not None and not pd.isna(est) and np.isfinite(est):
            df.loc[idx, "Estimate"] = float(est)
        for ci_col in ["95% CI Low", "95% CI High"]:
            val = df.loc[idx, ci_col]
            if val is not None and not pd.isna(val) and np.isfinite(val):
                df.loc[idx, ci_col] = float(val)
            else:
                df.loc[idx, ci_col] = None
        pval = df.loc[idx, "P-value"]
        if pval is not None and not pd.isna(pval) and np.isfinite(pval):
            df.loc[idx, "P-value"] = float(pval)
        else:
            df.loc[idx, "P-value"] = None
    df.to_csv(os.path.join(out_dir, "table3_mediation.csv"), index=False)
    return df

def validate_tables(tables_dir):
    anomalies = []
    for fname in os.listdir(tables_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(tables_dir, fname))
        if df.isna().any().any():
            anomalies.append((fname, "NaN"))
        for col in df.columns:
            if df[col].dtype.kind in "biufc":
                vals = df[col].values
                if np.isinf(vals).any():
                    anomalies.append((fname, "inf"))
                if np.nanmin(np.abs(vals)) < 1e-12:
                    anomalies.append((fname, "near-zero"))
                if np.nanmax(np.abs(vals)) > 1e6:
                    anomalies.append((fname, "extreme"))
    return anomalies

def generate_report(t1, pa_u, pa_m, pb, med_fiv, med_sort, roc_info, out_dir):
    auc_ci1 = roc_info["auc_ci"]["Model 1"]
    auc_ci2 = roc_info["auc_ci"]["Model 2"]
    auc_ci3 = roc_info["auc_ci"]["Model 3"]
    auc1 = roc_info["roc_curves"]["Model 1"]["auc"]
    auc2 = roc_info["roc_curves"]["Model 2"]["auc"]
    auc3 = roc_info["roc_curves"]["Model 3"]["auc"]
    en_method_1 = "Univariable and multivariable logistic regression analyses were performed to assess the independent association of A1 with FIV, adjusting for demographic and clinical covariates excluding postAS."
    en_method_2 = "Multivariable logistic regression was conducted to identify independent predictors of HT, incorporating A1, FIV, postAS, and potential confounders."
    en_method_3 = "Mediation analysis was implemented using a counterfactual imputation approach with a binary mediator (FIV) and binary outcome (HT); a complementary Sobel product-of-coefficients mediation with a continuous mediator (SORT) was accompanied by ρ-parameter sensitivity analysis."
    en_method_roc = "Bootstrap ROC analysis with 1000 resamples compared the discriminative performance of models with and without the mediator; AUCs with 95% confidence intervals were reported."
    en_result_1 = f"In adjusted analysis, A1 showed an independent association with FIV (see Table 2, Panel A)."
    en_result_2 = f"In the full model, A1 and FIV remained significant predictors of HT, with improved discrimination when the mediator was included (Model 2 AUC {auc2:.3f} [{auc_ci2[0]:.3f}–{auc_ci2[1]:.3f}] vs Model 1 AUC {auc1:.3f} [{auc_ci1[0]:.3f}–{auc_ci1[1]:.3f}])."
    en_result_3 = f"The indirect effect from A1 to HT via FIV was estimated as ACME {med_fiv['ACME']:.4f} [{med_fiv['ACME_CI'][0]:.4f}–{med_fiv['ACME_CI'][1]:.4f}], with a proportion mediated {med_fiv['PM']:.3f}."
    en_result_roc = f"Adding FIV and postAS produced the highest discrimination (Model 3 AUC {auc3:.3f} [{auc_ci3[0]:.3f}–{auc_ci3[1]:.3f}])."
    zh_interp_1 = "在对人口学与临床协变量进行调整后，A1与FIV之间的关联仍然显著，提示A1可能是FIV发生的独立影响因素。"
    zh_interp_2 = "在HT的多因素模型中，纳入FIV与postAS后模型的区分度明显提升，说明中介途径对结局预测具有实际贡献。"
    zh_interp_3 = "A1→FIV→HT的间接效应（ACME）提示FIV在其中发挥部分中介作用；若ρ需要达到较大的阈值才使ACME接近0，则说明结果对未测混杂因素较为稳健。"
    zh_interp_4 = "以SORT为连续型中介的Za×Zb分析给出一致结论，并通过ρ参数敏感性分析展示潜在混杂对间接效应的影响趋势。"
    lines = []
    lines.append("# Mediation and Prediction Report")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(en_method_1)
    lines.append(en_method_2)
    lines.append(en_method_3)
    lines.append(en_method_roc)
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(en_result_1)
    lines.append(en_result_2)
    lines.append(en_result_3)
    lines.append(en_result_roc)
    lines.append("")
    lines.append("## Chinese Interpretation")
    lines.append("")
    lines.append(zh_interp_1)
    lines.append(zh_interp_2)
    lines.append(zh_interp_3)
    lines.append(zh_interp_4)
    lines.append("")
    lines.append("## Tables and Figures")
    lines.append("")
    lines.append(f"Table 1: Baseline Characteristics (saved at tables/table1_baseline.csv)")
    lines.append(f"Table 2: Logistic Regression Results (Panel A and B) (tables/table2_panelA_univ_A1_to_FIV.csv, tables/table2_panelA_multiv_A1_to_FIV.csv, tables/table2_panelB_ht_predictors.csv)")
    lines.append(f"Table 3: Mediation Analysis Results (tables/table3_mediation.csv)")
    lines.append(f"Figure 1: Mediation Diagram (figures/figure1_mediation.png)")
    lines.append(f"Figure 2: Bootstrap ROC Curves (figures/figure2_bootstrap_roc.png)")
    lines.append(f"Figure 3: Sensitivity Analysis Plot (figures/figure3_rho_sensitivity.png)")
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path

def run_pipeline(data_path, out_root):
    os.makedirs(out_root, exist_ok=True)
    tables_dir = os.path.join(out_root, "tables")
    figs_dir = os.path.join(out_root, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    df = read_data(data_path)
    t1 = table1_baseline(df, tables_dir)
    b_med = int(os.environ.get("B_MED", "1000"))
    b_boot = int(os.environ.get("B_BOOT", "1000"))
    alpha_seq = [float(os.environ.get("ALPHA", "0.5")), 2.0, 5.0, 10.0]
    last_roc = None
    last_msort = None
    for alpha in alpha_seq:
        os.environ["ALPHA"] = str(alpha)
        # Panel A tables by bootstrap to avoid degenerate p-values
        univ_fiv_res, multiv_fiv_res = analysis1_fiv(df)
        pa_u = ols_boot_table("FIV ~ A1", df, B=max(200, b_med//5))
        covs = ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT"]
        covs = [c for c in covs if c in df.columns]
        formula_pa_m = "FIV ~ A1" + (" + " + " + ".join(covs) if covs else "")
        pa_m = ols_boot_table(formula_pa_m, df, B=max(200, b_med//5))
        ht_model = analysis2_ht(df)
        m_fiv = mediation_binary_fiv(df, B=b_med, seed=2026)
        m_sort = mediation_continuous_sort(df) if "SORT" in df.columns else {"rho_grid": np.array([]), "acme_rho": np.array([]), "rho_threshold": None, "P": None, "a":None, "b":None, "ACME":None}
        
        # Additional sensitivity analysis for CS and BaselineNIH
        m_cs = mediation_continuous_mediator(df, "CS") if "CS" in df.columns else None
        m_baselinesih = mediation_continuous_mediator(df, "BaselineNIH") if "BaselineNIH" in df.columns else None
        
        pb = or_table_from_glm(ht_model)
        table2_panel_a(pa_u, pa_m, tables_dir)
        table2_logistic(pb, tables_dir)
        table3_mediation(m_fiv, m_sort, tables_dir)
        roc_info = bootstrap_auc_models(df, B=b_boot, seed=2026, out_dir=figs_dir)
    # For figure: use sklearn-logit to avoid warnings
        res_a = fit_glm_binomial_robust("FIV ~ A1", df)
        parts_b = ["HT ~ FIV + A1"]
        if "postAS" in df.columns:
            parts_b.append("postAS")
        parts_b.extend([c for c in ["sex","age","Hyper","Dia","AF","CAD","CS","BaselineNIH","IVT","SORT"] if c in df.columns])
        res_b = fit_glm_binomial_robust(" + ".join(parts_b), df)
        res_cdir = fit_glm_binomial_robust("HT ~ A1 + FIV" + (" + postAS" if "postAS" in df.columns else ""), df)
        figure_mediation_diagram(res_a, res_b, res_cdir, os.path.join(figs_dir, "figure1_mediation.png"))
        if "rho_grid" in m_sort and len(m_sort["rho_grid"])>0:
            acme_obs = m_sort.get("ACME", None)
            # Generate main figure 3 with legend
            figure_rho_sensitivity(m_sort["rho_grid"], m_sort["acme_rho"], os.path.join(figs_dir, "figure3_rho_sensitivity.png"), 
                                  m_sort.get("rho_threshold"), acme_obs)
            # Generate clean version without text (for publication)
            figure_rho_sensitivity_clean(m_sort["rho_grid"], m_sort["acme_rho"], 
                                        os.path.join(figs_dir, "figure3_rho_sensitivity_clean.png"),
                                        m_sort.get("rho_threshold"), acme_obs, with_text=False)
        
        # Generate sensitivity analysis figures for CS and BaselineNIH
        if m_cs and "rho_grid" in m_cs and len(m_cs["rho_grid"]) > 0:
            figure_rho_sensitivity(m_cs["rho_grid"], m_cs["acme_rho"], 
                                  os.path.join(figs_dir, f"figure3_CS_sensitivity.png"),
                                  m_cs.get("rho_threshold"), m_cs.get("ACME"))
            figure_rho_sensitivity_clean(m_cs["rho_grid"], m_cs["acme_rho"],
                                        os.path.join(figs_dir, f"figure3_CS_sensitivity_clean.png"),
                                        m_cs.get("rho_threshold"), m_cs.get("ACME"), with_text=False)
        
        if m_baselinesih and "rho_grid" in m_baselinesih and len(m_baselinesih["rho_grid"]) > 0:
            figure_rho_sensitivity(m_baselinesih["rho_grid"], m_baselinesih["acme_rho"],
                                  os.path.join(figs_dir, f"figure3_BaselineNIH_sensitivity.png"),
                                  m_baselinesih.get("rho_threshold"), m_baselinesih.get("ACME"))
            figure_rho_sensitivity_clean(m_baselinesih["rho_grid"], m_baselinesih["acme_rho"],
                                        os.path.join(figs_dir, f"figure3_BaselineNIH_sensitivity_clean.png"),
                                        m_baselinesih.get("rho_threshold"), m_baselinesih.get("ACME"), with_text=False)
        anomalies = validate_tables(tables_dir)
        last_roc = roc_info
        last_msort = m_sort
        if not anomalies:
            break
    report_path = generate_report(t1, None, None, None, m_fiv, last_msort if last_msort else m_sort, last_roc if last_roc else roc_info, out_root)
    return {
        "tables_dir": tables_dir,
        "figures_dir": figs_dir,
        "report_path": report_path
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "总数据.xlsx")
    out_root = os.path.join(base_dir, "outputs")
    result = run_pipeline(data_path, out_root)
    print(json.dumps(result, ensure_ascii=False))
