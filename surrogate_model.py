# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 07:55:11 2025

@author: qpf
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File 2: train_pinn_residual.py
基于“残差式物理融合”的 Bayesian PINN 训练与实验（无单调性）。
模式：
  - physics_only：仅使用物理模型，作为强基线（不训练神经网络）；
  - physics_plus_data：数据 + 物理残差正则，神经网络仅学习残差 r。
同时支持不同训练数据比例的实验并绘制性能曲线图。
"""

import argparse, json, time, os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------
# 数据与物理模型（与文件一一致）
# ------------------------------
def load_and_preprocess(filepath: str, test_size: float=0.2, random_state: int=42):
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

    feat_order=[c for c in ['cement','slag','fly_ash','water','superplasticizer','coarse_agg','fine_agg','age'] if c in df.columns]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_raw = train_df[feat_order].reset_index(drop=True)
    test_raw  = test_df[feat_order].reset_index(drop=True)
    y_train_raw = train_df['strength'].values.astype(float)
    y_test_raw  = test_df['strength'].values.astype(float)

    scaler = StandardScaler().fit(train_raw.values)
    X_train = scaler.transform(train_raw.values)
    X_test  = scaler.transform(test_raw.values)

    y_mean = y_train_raw.mean()
    y_std  = y_train_raw.std() if y_train_raw.std()>0 else 1.0
    y_train = (y_train_raw-y_mean)/y_std
    y_test  = (y_test_raw -y_mean)/y_std
    return train_raw, test_raw, X_train, X_test, y_train, y_test, y_mean, y_std, feat_order, y_train_raw, y_test_raw

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

# ------------------------------
# 残差式 PINN（无单调性）
# ------------------------------
class ResidualPINN:
    """
    网络输出残差 r(x;θ)，最终预测 y_hat = y_phys_scaled + r。
    损失: L = λ_data * MSE(y_hat, y_true) + λ_r * mean(r^2)
    """
    def __init__(self, input_dim:int, feature_names:List[str], physics_params:np.ndarray,
                 y_mean:float, y_std:float, hidden_dims:List[int]=None, dropout:float=0.1,
                 lambda_r: float=0.05, lambda_data: float=1.0, lr: float=0.01, seed:int=42):
        self.in_dim = input_dim
        self.feature_names = feature_names
        self.physics_params = np.array(physics_params, float)
        self.y_mean=float(y_mean); self.y_std=float(y_std)
        self.hidden = hidden_dims if hidden_dims is not None else [64,32]
        self.dropout = float(dropout)
        self.lambda_r = float(lambda_r)
        self.lambda_data = float(lambda_data)
        self.lr = float(lr)
        np.random.seed(seed)
        dims=[self.in_dim]+self.hidden+[1]
        self.layers=[]
        for i in range(len(dims)-1):
            n_in,n_out=dims[i],dims[i+1]
            W = np.random.randn(n_in,n_out)*np.sqrt(2.0/n_in)
            b = np.zeros((1,n_out))
            self.layers.append({'W':W,'b':b})
        self._adam_m=[{'W':np.zeros_like(l['W']), 'b':np.zeros_like(l['b'])} for l in self.layers]
        self._adam_v=[{'W':np.zeros_like(l['W']), 'b':np.zeros_like(l['b'])} for l in self.layers]
        self._t=0

    @staticmethod
    def relu(x): return np.maximum(0.0,x)
    @staticmethod
    def relu_grad(x): return (x>0).astype(np.float32)

    def forward(self, X, training:bool=True):
        h=X; self._masks=[]
        for layer in self.layers[:-1]:
            z = h@layer['W'] + layer['b']
            h = self.relu(z)
            if training and self.dropout>0:
                m = np.random.binomial(1, 1-self.dropout, size=h.shape).astype(np.float32)/(1-self.dropout)
                h = h*m; self._masks.append(m)
            else:
                self._masks.append(np.ones_like(h))
        out = h@self.layers[-1]['W'] + self.layers[-1]['b']  # 输出残差 r
        return out

    def physics_scaled(self, X_raw: np.ndarray)->np.ndarray:
        N=X_raw.shape[0]
        preds=np.empty(N,dtype=float)
        for i in range(N):
            feats={self.feature_names[j]:float(X_raw[i,j]) for j in range(len(self.feature_names))}
            preds[i]=complete_physics_model(feats, self.physics_params)
        preds_scaled=(preds-self.y_mean)/(self.y_std+1e-8)
        return preds_scaled.reshape(-1,1)

    def compute_loss(self, X_scaled, X_raw, y_true):
        r = self.forward(X_scaled, training=True)
        y_phys = self.physics_scaled(X_raw)
        y_hat = r + y_phys
        data_loss = np.mean((y_hat - y_true.reshape(-1,1))**2)
        reg_r = np.mean(r**2)
        total = self.lambda_data*data_loss + self.lambda_r*reg_r
        return {'total':float(total), 'data':float(data_loss), 'reg_r':float(reg_r)}, (r, y_hat, y_phys)

    def backward(self, X_scaled, cache, y_true):
        r, y_hat, _ = cache
        N=X_scaled.shape[0]
        # dL/dy_hat = 2*λ_data/N*(y_hat - y_true)
        dy = 2*self.lambda_data*(y_hat - y_true.reshape(-1,1))/N
        # 因 y_hat = r + y_phys，且 y_phys 不依赖参数，故 dL/dr = dy + 2*λ_r/N*r
        dr = dy + 2*self.lambda_r*r/N

        # 反传
        acts=[X_scaled]; pre=[]; masks=[]
        h=X_scaled
        for layer,mask in zip(self.layers[:-1], self._masks):
            z=h@layer['W']+layer['b']; pre.append(z)
            h=self.relu(z); masks.append(mask)
            h=h*mask; acts.append(h)
        # 输出层梯度
        grads=[]
        dW = acts[-1].T @ dr
        db = np.sum(dr, axis=0, keepdims=True)
        grads.append({'W':dW,'b':db})
        dh = dr @ self.layers[-1]['W'].T
        # 隐层逆序
        for i in range(len(self.layers)-2, -1, -1):
            dh = dh * masks[i] if i<len(masks) else dh
            dz = dh * self.relu_grad(pre[i]) if i<len(pre) else dh
            dW = acts[i].T @ dz
            db = np.sum(dz, axis=0, keepdims=True)
            grads.append({'W':dW,'b':db})
            if i>0: dh = dz @ self.layers[i]['W'].T
        grads.reverse()
        return grads

    def step(self, grads):
        beta1,beta2,eps=0.9,0.999,1e-8
        self._t += 1; t=self._t
        for i,(layer,g) in enumerate(zip(self.layers,grads)):
            for p in ['W','b']:
                self._adam_m[i][p] = beta1*self._adam_m[i][p] + (1-beta1)*g[p]
                self._adam_v[i][p] = beta2*self._adam_v[i][p] + (1-beta2)*(g[p]**2)
                mhat = self._adam_m[i][p]/(1-beta1**t)
                vhat = self._adam_v[i][p]/(1-beta2**t)
                layer[p] -= self.lr*mhat/(np.sqrt(vhat)+eps)

    def train(self, Xtr, Xtr_raw, ytr, Xval, Xval_raw, yval,
              epochs:int=200, batch:int=32, verbose:bool=False,
              lambda_r0:float=None, lambda_r_min:float=None, anneal_T:int=None,
              early_stop:bool=True, patience:int=30, min_delta:float=1e-4):
        # 可选：指数退火 λ_r
        lam0 = self.lambda_r if lambda_r0 is None else float(lambda_r0)
        lam_min = self.lambda_r if lambda_r_min is None else float(lambda_r_min)
        T = max(1, epochs//3) if anneal_T is None else max(1,int(anneal_T))
        best=(np.inf, None, 0)  # val_loss, snapshot, epoch
        wait=0
        hist={'train':[],'val':[]}
        N=Xtr.shape[0]
        for ep in range(epochs):
            # 退火
            self.lambda_r = lam_min + (lam0-lam_min)*np.exp(-ep/T)
            # 小批
            perm=np.random.permutation(N)
            Xs, Xr, ys = Xtr[perm], Xtr_raw[perm], ytr[perm]
            losses=[]
            for i in range(0,N,batch):
                xb, xrb, yb = Xs[i:i+batch], Xr[i:i+batch], ys[i:i+batch]
                l, cache = self.compute_loss(xb, xrb, yb)
                grads = self.backward(xb, cache, yb)
                self.step(grads)
                losses.append(l['total'])
            # 验证
            val_pred = self.predict_mean(Xval, Xval_raw)
            val_loss = float(np.mean((val_pred - yval.reshape(-1,1))**2))
            hist['train'].append(float(np.mean(losses)))
            hist['val'].append(val_loss)
            if verbose and (ep==0 or (ep+1)%20==0):
                print(f"Epoch {ep+1:3d} | train {np.mean(losses):.4f} | val {val_loss:.4f} | lambda_r={self.lambda_r:.4f}")
            # 早停
            if early_stop:
                if val_loss < best[0]-min_delta:
                    best=(val_loss, [ {'W':l['W'].copy(),'b':l['b'].copy()} for l in self.layers ], ep); wait=0
                else:
                    wait+=1
                    if wait>=patience:
                        if verbose: print(f"Early stop at epoch {ep+1}")
                        break
        # 恢复最佳
        if best[1] is not None:
            for L, snap in zip(self.layers, best[1]): L['W']=snap['W']; L['b']=snap['b']
        return hist

    # MC Dropout 预测（用于均值与不确定性）
    def predict_mc(self, X, X_raw, n:int=30):
        preds=[]
        for _ in range(n):
            r = self.forward(X, training=True)
            y_phys = self.physics_scaled(X_raw)
            y_hat = r + y_phys
            preds.append(y_hat)
        P = np.concatenate(preds, axis=1)  # (N,n)
        mean = np.mean(P, axis=1, keepdims=True)
        std  = np.std(P, axis=1, keepdims=True)
        return mean, std
    def predict_mean(self, X, X_raw):
        r = self.forward(X, training=False)
        return r + self.physics_scaled(X_raw)

# ------------------------------
# 指标与绘图
# ------------------------------
def metrics(y, yhat):
    r2 = r2_score(y, yhat)
    rmse = mean_squared_error(y, yhat, squared=False)
    mae = np.mean(np.abs(y - yhat))
    return r2, rmse, mae

def plot_r2_curve(save_path:str, fractions:List[float], curves:Dict[str,List[float]]):
    plt.figure()
    xs = [100*f for f in fractions]
    for label, ys in curves.items():
        plt.plot(xs, ys, marker='o', label=label)
    plt.xlabel("训练数据比例 (%)")
    plt.ylabel("测试集 R²")
    plt.title("不同数据比例下的测试 R² 对比")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    print(f"已保存曲线图: {save_path}")

# ------------------------------
# 实验管线
# ------------------------------
def run_experiments(data_path:str, params_json:str, mode:str,
                    fractions:List[float], epochs:int, batch:int,
                    ensemble:int, dropout:float, lambda_r:float,
                    lambda_r_min:float, anneal_T:int, verbose:bool):
    # 载入物理参数
    with open(params_json,"r",encoding="utf-8") as f:
        payload=json.load(f)
    physics_params = np.array(payload["params"], dtype=float)
    print("载入物理参数:", np.round(physics_params,4))

    # 固定一次全量数据以得到稳定的 test 划分
    (train_raw_full, test_raw, X_train_full, X_test_full,
     y_train_full, y_test_full, y_mean_full, y_std_full,
     feat_names, y_train_full_raw, y_test_full_raw) = load_and_preprocess(data_path, 0.2, 42)

    r2_curves = {}
    r2_phys_curve = []  # 物理基线与比例无关，但为了画图我们重复记录

    for frac in fractions:
        n_total = train_raw_full.shape[0]
        n_sub = max(1, int(frac*n_total))
        idx = np.random.default_rng(42).permutation(n_total)[:n_sub]
        train_raw = train_raw_full.iloc[idx].reset_index(drop=True)
        X_train = StandardScaler().fit_transform(train_raw.values)
        X_test  = StandardScaler().fit(train_raw.values).transform(test_raw.values)  # 用子集统计量
        # 目标缩放按子集
        y_train_raw = y_train_full_raw[idx]
        y_mean = y_train_raw.mean(); y_std = y_train_raw.std() if y_train_raw.std()>0 else 1.0
        y_train = (y_train_raw - y_mean)/y_std
        y_test  = (y_test_full_raw - y_mean)/y_std

        # 物理基线（MPa 上评估）
        def apply_phys(df): 
            return np.array([complete_physics_model(row.to_dict(), physics_params) for _,row in df.iterrows()])
        yhat_phys_test = apply_phys(test_raw)
        r2_phys, rmse_phys, mae_phys = metrics(y_test_full_raw, yhat_phys_test)
        r2_phys_curve.append(r2_phys)

        if mode == "physics_only":
            print(f"[frac={frac:.2f}] 纯物理: Test R²={r2_phys:.3f}, RMSE={rmse_phys:.3f}, MAE={mae_phys:.3f}")
            # 只画物理一条线
            r2_curves["Physics-only"] = [r2_phys]*len(fractions)
            continue

        # physics_plus_data：训练残差式 PINN 集成
        test_r2_list=[]
        for m in range(ensemble):
            model = ResidualPINN(
                input_dim=X_train.shape[1], feature_names=feat_names,
                physics_params=physics_params, y_mean=y_mean, y_std=y_std,
                hidden_dims=[64,32], dropout=dropout,
                lambda_r=lambda_r, lambda_data=1.0, lr=0.01, seed=42+m
            )
            _ = model.train(
                X_train, train_raw.values, y_train,
                X_test,  test_raw.values,  y_test,
                epochs=epochs, batch=batch, verbose=False,
                lambda_r0=lambda_r, lambda_r_min=lambda_r_min, anneal_T=anneal_T,
                early_stop=True, patience=30, min_delta=1e-4
            )
            # 测试集预测并还原到 MPa
            y_pred_scaled, _ = model.predict_mc(X_test, test_raw.values, n=30)
            y_pred_raw = (y_pred_scaled.flatten())*y_std + y_mean
            r2, rmse, mae = metrics(y_test_full_raw, y_pred_raw)
            test_r2_list.append(r2)
        mean_r2 = float(np.mean(test_r2_list))
        print(f"[frac={frac:.2f}] 残差式: Ensemble Test R²={mean_r2:.3f} | 物理基线={r2_phys:.3f}")
        r2_curves.setdefault("Physics+Data (Residual)", []).append(mean_r2)

    # 如果只是 physics_only，构造曲线字典
    if mode == "physics_only":
        r2_curves = {"Physics-only": r2_phys_curve}
    else:
        # 添加物理基线到图中
        r2_curves["Physics-only"] = r2_phys_curve

    # 绘图
    os.makedirs("figs", exist_ok=True)
    plot_r2_curve(os.path.join("figs","r2_vs_fraction.png"), fractions, r2_curves)

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="残差式 PINN 训练/实验（无单调性）")
    ap.add_argument("--data", type=str, default="concrete_data.csv")
    ap.add_argument("--params", type=str, default="physics_params.json", help="文件一导出的 JSON 参数")
    ap.add_argument("--mode", type=str, default="physics_plus_data", choices=["physics_only","physics_plus_data"])
    ap.add_argument("--fractions", type=str, default="1.0,0.8,0.6,0.4", help="训练数据比例，逗号分隔")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_r", type=float, default=0.05, help="残差正则初始权重 λ_r(0)")
    ap.add_argument("--lambda_r_min", type=float, default=0.01, help="退火后的最小权重")
    ap.add_argument("--anneal_T", type=int, default= max(1,200//3), help="λ_r 指数退火温度 T（以 epoch 为单位）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    fractions = [float(x.strip()) for x in args.fractions.split(',') if x.strip()]
    run_experiments(
        data_path=args.data, params_json=args.params, mode=args.mode,
        fractions=fractions, epochs=args.epochs, batch=args.batch,
        ensemble=args.ensemble, dropout=args.dropout,
        lambda_r=args.lambda_r, lambda_r_min=args.lambda_r_min,
        anneal_T=args.anneal_T, verbose=args.verbose
    )

if __name__=="__main__":
    main()
