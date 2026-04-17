# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 08:24:19 2025

@author: qpf
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uncertainty_eval.py

目标
- 加载 train_compact_plots.py 保存的最优模型（.npz + .json），支持：
  - Data-only（纯数据）
  - Residual（残差式，需要 physics_params）
- 使用 MC Dropout 进行不确定性评估，给出：
  - 90% 与 95% 置信区间覆盖率（Empirical vs Nominal）
  - 平均区间宽度（Sharpness）
  - NLL（假设高斯，mean/variance 来自 MC）
- 合并出图（英文，中文日志）：
  U1) Calibration（Nominal vs Empirical coverage）
  U2) Z-score 直方图（标准化残差）
  U3) Abs Error vs Predicted Std 散点

使用示例
python uncertainty_eval.py --data concrete_data.csv --model figs_compact/best_residual.npz \
  --meta figs_compact/best_residual.json --outdir figs_uncertainty --n_mc 200 \
  --font_family "Times New Roman" --font_size 12
"""

import argparse, json, os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import erf, sqrt

# -----------
# 样式
# -----------
def setup_matplotlib(font_family: str = "Times New Roman", font_size: int = 12):
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = font_size + 1
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size - 1
    plt.rcParams["figure.dpi"] = 160

# -----------
# 数据/物理
# -----------
def load_and_split(filepath: str, random_state:int=42):
    df = pd.read_csv(filepath); df.columns=df.columns.str.strip()
    mapping = {
        'cement':'cement','blast_furnace_slag':'slag','slag':'slag',
        'fly_ash':'fly_ash','flyash':'fly_ash','water':'water',
        'superplasticizer':'superplasticizer',
        'coarse_aggregate':'coarse_agg','fine_aggregate':'fine_agg',
        'age':'age','concrete_compressive_strength':'strength','strength':'strength'
    }
    rename={}; used=set()
    for src,dst in mapping.items():
        for col in df.columns:
            if col in used: continue
            if col.lower()==src.lower() or src.replace('_','') in col.lower().replace('_',''):
                rename[col]=dst; used.add(col); break
    df=df.rename(columns=rename)
    for z in ['slag','fly_ash','superplasticizer']:
        if z in df.columns: df[z]=df[z].fillna(0.0)
    df=df.dropna()
    for req in ['cement','water','age','strength']:
        if req not in df.columns: raise ValueError(f"缺少必要列: {req}")
    feats=[c for c in ['cement','slag','fly_ash','water','superplasticizer','coarse_agg','fine_agg','age'] if c in df.columns]
    tr, te = train_test_split(df, test_size=0.2, random_state=random_state)
    Xtr_raw = tr[feats].values; Xte_raw = te[feats].values
    ytr_raw = tr['strength'].values.astype(float); yte_raw = te['strength'].values.astype(float)
    return feats, Xtr_raw, Xte_raw, ytr_raw, yte_raw

def complete_physics_model(feat: Dict[str,float], params: List[float]) -> float:
    A,B,k_s,k_f,eta_max,n = params
    C=feat.get('cement',0.0); S=feat.get('slag',0.0); F=feat.get('fly_ash',0.0)
    W=feat.get('water',0.0);  SP=feat.get('superplasticizer',0.0); age=feat.get('age',0.0)
    k_s_t = k_s*age/(7.0+age); k_f_t = k_f*age/(14.0+age)
    C_eff = C + k_s_t*S + k_f_t*F
    SP_max=30.0; reduction = 1.0 - eta_max*((SP/(SP_max+1e-8))**0.6)
    W_eff=W*reduction; wcr=W_eff/(C_eff+1e-8); age_factor=(age/28.0)**n
    strength = A*(1.0/(wcr+0.5))**B * age_factor
    return float(np.clip(strength,5.0,150.0))

# -----------
# 载入模型
# -----------
class LoadedMLP:
    def __init__(self, weights:dict, hidden:List[int], dropout:float, lr:float=0.01):
        self.layers=[]
        # weights: {"W0","b0","W1","b1",...}
        keys = sorted([k for k in weights.keys() if k.startswith("W")], key=lambda x:int(x[1:]))
        for i,k in enumerate(keys):
            W = weights[k]; b = weights[f"b{i}"]
            self.layers.append({'W':W.copy(), 'b':b.copy()})
        self.dropout=dropout
        self.lr=lr
        self._masks=[]; self._pre=[]; self._acts=[]
    @staticmethod
    def relu(x): return np.maximum(0.0,x)
    def forward(self, X, training:bool=True):
        h=X; self._masks=[]
        for layer in self.layers[:-1]:
            z = h@layer['W'] + layer['b']
            h = self.relu(z)
            if training and self.dropout>0:
                m = np.random.binomial(1, 1-self.dropout, size=h.shape).astype(np.float32)/(1-self.dropout)
                h = h*m
        out = h@self.layers[-1]['W'] + self.layers[-1]['b']
        return out

def load_saved_model(model_path:str, meta_path:str):
    npz = np.load(model_path)
    with open(meta_path,"r",encoding="utf-8") as f:
        meta=json.load(f)
    weights = {k:npz[k] for k in npz.files}
    mdl = LoadedMLP(weights, hidden=meta.get("hidden",[64,32]), dropout=meta.get("dropout",0.1), lr=meta.get("lr",0.01))
    return mdl, meta

# -----------
# 不确定性评估
# -----------
def mc_predict_data_only(mdl:LoadedMLP, X_scaled:np.ndarray, n_mc:int):
    preds=[]
    for _ in range(n_mc):
        y = mdl.forward(X_scaled, training=True)
        preds.append(y)
    P = np.concatenate(preds, axis=1)  # (N, n_mc)
    mean = np.mean(P, axis=1)
    std  = np.std(P, axis=1) + 1e-8
    return mean, std

def mc_predict_residual(mdl:LoadedMLP, X_scaled:np.ndarray, X_raw:np.ndarray, meta:dict, n_mc:int):
    # 物理部分（缩放）
    feat_names = meta["feat_names"]; physics_params = np.array(meta["physics_params"], float)
    y_mean = float(meta["y_mean"]); y_std=float(meta["y_std"])
    def phys_scaled_for_row(row):
        feats={feat_names[j]:float(row[j]) for j in range(len(feat_names))}
        val = complete_physics_model(feats, physics_params)
        return (val - y_mean)/(y_std+1e-8)
    y_phys_scaled = np.array([phys_scaled_for_row(row) for row in X_raw])
    preds=[]
    for _ in range(n_mc):
        r = mdl.forward(X_scaled, training=True).reshape(-1)
        y_hat = r + y_phys_scaled
        preds.append(y_hat.reshape(-1,1))
    P = np.concatenate(preds, axis=1)
    mean = np.mean(P, axis=1)
    std  = np.std(P, axis=1) + 1e-8
    return mean, std

def coverage_empirical(y_true, mu, sigma, alpha=0.95):
    # 正态区间：mu ± z * sigma
    from scipy.stats import norm
    z = norm.ppf( (1+alpha)/2.0 )
    lower = mu - z*sigma
    upper = mu + z*sigma
    return float(np.mean((y_true >= lower) & (y_true <= upper))), float(np.mean(upper-lower))

def nll_gaussian(y_true, mu, sigma):
    # -log p(y|mu,sigma) = 0.5*log(2πσ^2) + 0.5*((y-mu)/σ)^2
    sigma = np.maximum(sigma, 1e-8)
    return float(np.mean(0.5*np.log(2*np.pi*sigma**2) + 0.5*((y_true-mu)/sigma)**2))

# -----------
# 合并出图（英文）
# -----------
def fig_calibration(save_path, nominals, empiricals):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.plot([0,1],[0,1],'k--',label="Ideal")
    ax.plot(nominals, empiricals, marker='o', label="Empirical")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Prediction Interval Calibration")
    ax.grid(True, ls='--', alpha=.5); ax.legend()
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)

def fig_z_hist(save_path, z):
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.hist(z, bins=40, alpha=.85)
    ax.set_title("Standardized Residuals"); ax.set_xlabel("z = (y - μ)/σ"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)

def fig_uncertainty_vs_error(save_path, abs_err, sigma):
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.scatter(sigma, abs_err, s=12, alpha=.6, edgecolors='none')
    ax.set_title("Abs Error vs Predicted Std"); ax.set_xlabel("Predicted std (σ)"); ax.set_ylabel("|Error|")
    ax.grid(True, ls='--', alpha=.5)
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)

# -----------
# 主流程
# -----------
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    setup_matplotlib(args.font_family, args.font_size)

    mdl, meta = load_saved_model(args.model, args.meta)
    print(f"→ 已载入模型类型：{meta['type']}；隐藏层：{meta.get('hidden')}；dropout={meta.get('dropout')}")
    feats, Xtr_raw, Xte_raw, ytr_raw, yte_raw = load_and_split(args.data, random_state=42)

    # 统一用“全量训练”统计还原（与保存的meta对齐）
    feat_mean = np.array(meta["feat_mean"], float); feat_std=np.array(meta["feat_std"], float)+1e-8
    Xte_scaled = (Xte_raw - feat_mean)/feat_std
    y_mean = float(meta["y_mean"]); y_std=float(meta["y_std"])
    yte_scaled = (yte_raw - y_mean)/(y_std+1e-8)

    # MC 预测（区分 data-only / residual）
    print("→ 开始MC Dropout不确定性推断 ...")
    if meta["type"]=="data_only":
        mu_s, std_s = mc_predict_data_only(mdl, Xte_scaled, n_mc=args.n_mc)
        y_mu = mu_s*y_std + y_mean
        y_std_pred = std_s*y_std
    else:
        mu_s, std_s = mc_predict_residual(mdl, Xte_scaled, Xte_raw, meta, n_mc=args.n_mc)
        y_mu = mu_s*y_std + y_mean
        y_std_pred = std_s*y_std

    # 评价指标
    r2 = r2_score(yte_raw, y_mu)
    rmse = mean_squared_error(yte_raw, y_mu, squared=False)
    mae = float(np.mean(np.abs(yte_raw - y_mu)))
    cov90, w90 = coverage_empirical(yte_raw, y_mu, y_std_pred, alpha=0.90)
    cov95, w95 = coverage_empirical(yte_raw, y_mu, y_std_pred, alpha=0.95)
    z = (yte_raw - y_mu)/(y_std_pred + 1e-8)
    nll = nll_gaussian(yte_raw, y_mu, y_std_pred)

    # 中文表格打印
    df = pd.DataFrame([{
        "模型类型": "纯数据" if meta["type"]=="data_only" else "残差式(数据+物理)",
        "R2": r2, "RMSE": rmse, "MAE": mae,
        "90%覆盖率": cov90, "90%平均区间宽度": w90,
        "95%覆盖率": cov95, "95%平均区间宽度": w95,
        "NLL(高斯)": nll
    }])
    print("\n=== 不确定性评估指标（中文表格） ===")
    print(df.to_string(index=False))

    # 合并出图
    nominals = [0.5, 0.8, 0.9, 0.95]
    empiricals=[]
    for a in nominals:
        cov,_ = coverage_empirical(yte_raw, y_mu, y_std_pred, alpha=a)
        empiricals.append(cov)
    fig_calibration(os.path.join(args.outdir, "U1_Calibration.png"), nominals, empiricals)
    fig_z_hist(os.path.join(args.outdir, "U2_ZscoreHist.png"), z)
    fig_uncertainty_vs_error(os.path.join(args.outdir, "U3_Uncertainty_vs_Error.png"),
                             np.abs(yte_raw - y_mu), y_std_pred)

    print(f"\n→ 已输出不确定性相关图片到目录：{args.outdir}")
    print("说明：图中文字为英文；终端为中文。MC样本数可用 --n_mc 调整（建议 100~500）。")

# -----------
# CLI
# -----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="载入最优模型并进行不确定性评估（中文日志 / 英文图片）")
    ap.add_argument("--data", type=str, default="concrete_data.csv")
    ap.add_argument("--model", type=str, default="figs_compact/best_residual.npz")
    ap.add_argument("--meta",  type=str, default="figs_compact/best_residual.json")
    ap.add_argument("--n_mc", type=int, default=200)
    ap.add_argument("--font_family", type=str, default="Times New Roman")
    ap.add_argument("--font_size", type=int, default=12)
    ap.add_argument("--outdir", type=str, default="figs_uncertainty")
    args = ap.parse_args()
    run(args)
