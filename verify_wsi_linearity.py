# import os
# import json
# import torch
# import numpy as np
# from sklearn.cross_decomposition import CCA
# from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from collections import defaultdict
# import random

# # ==========================================
# # [설정] 경로
# # ==========================================
# TS_ROOT = "/mnt/i/brain/TS_tokens_backup"
# BS_ROOT = "/mnt/i/brain/BS_tokens_backup"
# TS_JSON = "./ts_train.json"
# BS_JSON = "./bs_train.json"

# MAX_PATCHES_PER_WSI = 500

# def get_case_id(path):
#     folder_name = os.path.dirname(path)
#     parts = folder_name.split('-')
#     if len(parts) >= 3: return "-".join(parts[:3])
#     return folder_name

# def load_and_aggregate(root_dir, json_path):
#     print(f"📂 Loading {json_path}...")
#     with open(json_path, 'r') as f:
#         rel_paths = json.load(f)
    
#     case_map = defaultdict(list)
#     for rel in rel_paths:
#         cid = get_case_id(rel)
#         full_path = os.path.join(root_dir, rel)
#         case_map[cid].append(full_path)
        
#     print(f"   👉 Found {len(case_map)} cases.")
    
#     wsi_vectors = {}
#     print("   🚀 Aggregating WSI features...")
#     for cid, files in tqdm(case_map.items()):
#         if len(files) > MAX_PATCHES_PER_WSI:
#             files = random.sample(files, MAX_PATCHES_PER_WSI)
#         feats = []
#         for fpath in files:
#             try:
#                 t = torch.load(fpath, map_location='cpu').float()
#                 t_vec = t.mean(dim=(1, 2)) if t.dim() == 3 else t.mean(dim=0)
#                 feats.append(t_vec.numpy())
#             except: pass
        
#         if len(feats) > 0:
#             wsi_vectors[cid] = np.mean(np.stack(feats), axis=0)
            
#     return wsi_vectors

# def main():
#     print("🔬 Verifying WSI-level Linearity (with PCA)...")
    
#     ts_dict = load_and_aggregate(TS_ROOT, TS_JSON)
#     bs_dict = load_and_aggregate(BS_ROOT, BS_JSON)
    
#     common_ids = sorted(list(set(ts_dict.keys()) & set(bs_dict.keys())))
#     print(f"\n🔗 Paired Cases: {len(common_ids)}")
    
#     if len(common_ids) < 10:
#         print("❌ Not enough data.")
#         return

#     X = np.array([ts_dict[cid] for cid in common_ids])
#     Y = np.array([bs_dict[cid] for cid in common_ids])
    
#     # ---------------------------------------------------------
#     # [Sanity Check] 혹시 데이터가 복사붙여넣기인지 확인
#     # ---------------------------------------------------------
#     diff = np.linalg.norm(X - Y)
#     if diff < 1e-5:
#         print("\n🚨 [WARNING] TS and BS data are IDENTICAL!")
#         print("   -> 데이터를 잘못 로드했거나 경로가 꼬였습니다. 확인하세요!")
#         return
#     else:
#         print(f"✅ Data Difference Check Passed (Diff: {diff:.2f})")

#     # ---------------------------------------------------------
#     # [핵심] PCA로 차원 축소 (N보다 훨씬 작게!)
#     # ---------------------------------------------------------
#     n_components = min(20, len(common_ids) - 1) # 20차원으로 압축
#     print(f"\n📉 Applying PCA (1536 -> {n_components})...")
    
#     # 정규화 (PCA 전 필수)
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#     X_scaled = scaler_x.fit_transform(X)
#     Y_scaled = scaler_y.fit_transform(Y)
    
#     pca_x = PCA(n_components=n_components)
#     pca_y = PCA(n_components=n_components)
    
#     X_pca = pca_x.fit_transform(X_scaled)
#     Y_pca = pca_y.fit_transform(Y_scaled)
    
#     print(f"   Explained Variance (X): {np.sum(pca_x.explained_variance_ratio_):.2f}")
#     print(f"   Explained Variance (Y): {np.sum(pca_y.explained_variance_ratio_):.2f}")

#     # ---------------------------------------------------------
#     # CCA & Regression 수행
#     # ---------------------------------------------------------
#     print("\n🧮 Running CCA on PCA features...")
#     cca = CCA(n_components=1)
#     X_c, Y_c = cca.fit_transform(X_pca, Y_pca)
#     corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    
#     print("🧮 Running Regression on PCA features...")
#     reg = LinearRegression()
#     reg.fit(X_pca, Y_pca)
#     r2 = reg.score(X_pca, Y_pca)
    
#     print("\n" + "="*50)
#     print(f"🔥 REAL RESULTS (PCA-{n_components})")
#     print(f"✅ Correlation (CCA): {corr:.4f}")
#     print(f"✅ Linear Predictability (R²): {r2:.4f}")
#     print("="*50)
    
#     # 해석 가이드
#     if corr > 0.7:
#         print("🎉 대박! 압축 후에도 선형성이 강력합니다. (진짜 선형)")
#     elif corr > 0.4:
#         print("👌 괜찮음. 어느 정도 선형 경향이 있습니다. (Mapper 학습 가능)")
#     else:
#         print("🤔 흠... 선형성이 약합니다. Non-linear Mapper가 필요할 수도 있습니다.")

#     # 시각화
#     plt.figure(figsize=(6, 6))
#     plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.6)
#     m, b = np.polyfit(X_c[:, 0], Y_c[:, 0], 1)
#     plt.plot(X_c[:, 0], m*X_c[:, 0] + b, 'r--', label=f'Fit (Corr={corr:.2f})')
#     plt.xlabel("TS (PCA+CCA)"); plt.ylabel("BS (PCA+CCA)")
#     plt.title(f"Linearity Check (PCA-{n_components})\nCorr={corr:.4f}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig("wsi_linearity_pca_check.png")
#     print("📸 Graph saved.")

# if __name__ == "__main__":
#     main()

from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict
import random
import math

# ==========================================
# [설정] 경로
# ==========================================
TS_ROOT = "/mnt/i/brain/TS_tokens_backup"
BS_ROOT = "/mnt/i/brain/BS_tokens_backup"
TS_JSON = "./ts_train.json"
BS_JSON = "./bs_train.json"

MAX_PATCHES_PER_WSI = 500

def get_case_id(path):
    folder_name = os.path.dirname(path)
    parts = folder_name.split('-')
    if len(parts) >= 3: return "-".join(parts[:3])
    return folder_name

def load_and_aggregate(root_dir, json_path):
    print(f"📂 Loading {json_path}...")
    with open(json_path, 'r') as f:
        rel_paths = json.load(f)
    
    case_map = defaultdict(list)
    for rel in rel_paths:
        cid = get_case_id(rel)
        full_path = os.path.join(root_dir, rel)
        case_map[cid].append(full_path)
        
    print(f"   👉 Found {len(case_map)} cases.")
    
    wsi_vectors = {}
    print("   🚀 Aggregating WSI features...")
    for cid, files in tqdm(case_map.items()):
        if len(files) > MAX_PATCHES_PER_WSI:
            files = random.sample(files, MAX_PATCHES_PER_WSI)
        feats = []
        for fpath in files:
            try:
                t = torch.load(fpath, map_location='cpu').float()
                t_vec = t.mean(dim=(1, 2)) if t.dim() == 3 else t.mean(dim=0)
                feats.append(t_vec.numpy())
            except: pass
        
        if len(feats) > 0:
            wsi_vectors[cid] = np.mean(np.stack(feats), axis=0)
            
    return wsi_vectors

def adjusted_r2(r2, n, p):
    """
    Calculate Adjusted R-squared
    n: number of samples
    p: number of predictors
    """
    if n - p - 1 <= 0: return r2 # Avoid division by zero
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def main():
    print("🔬 Verifying WSI-level Linearity with Multiple Models & Metrics...")
    
    ts_dict = load_and_aggregate(TS_ROOT, TS_JSON)
    bs_dict = load_and_aggregate(BS_ROOT, BS_JSON)
    
    common_ids = sorted(list(set(ts_dict.keys()) & set(bs_dict.keys())))
    print(f"\n🔗 Paired Cases: {len(common_ids)}")
    
    if len(common_ids) < 10:
        print("❌ Not enough data.")
        return

    X = np.array([ts_dict[cid] for cid in common_ids])
    Y = np.array([bs_dict[cid] for cid in common_ids])
    
    # Sanity Check
    diff = np.linalg.norm(X - Y)
    if diff < 1e-5:
        print("\n🚨 [WARNING] TS and BS data are IDENTICAL!")
        return
    else:
        print(f"✅ Data Difference Check Passed (Diff: {diff:.2f})")

    # PCA
    n_components = min(20, len(common_ids) - 1) 
    print(f"\n📉 Applying PCA (1536 -> {n_components})...")
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)
    
    pca_x = PCA(n_components=n_components)
    pca_y = PCA(n_components=n_components)
    
    X_pca = pca_x.fit_transform(X_scaled)
    Y_pca = pca_y.fit_transform(Y_scaled)

    # ---------------------------------------------------------
    # CCA & Regression & PLS 수행
    # ---------------------------------------------------------
    print("\n🧮 Running Linear Model Comparison on PCA features...")
    
    # 1. CCA 
    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(X_pca, Y_pca)
    corr_cca = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    
    # 2. Multiple Linear Regression (MLR)
    reg_mlr = LinearRegression()
    reg_mlr.fit(X_pca, Y_pca)
    Y_pred_mlr = reg_mlr.predict(X_pca)
    
    r2_mlr = r2_score(Y_pca, Y_pred_mlr)
    mse_mlr = mean_squared_error(Y_pca, Y_pred_mlr)
    rmse_mlr = np.sqrt(mse_mlr)
    mae_mlr = mean_absolute_error(Y_pca, Y_pred_mlr)
    adj_r2_mlr = adjusted_r2(r2_mlr, len(Y_pca), n_components)

    # 3. Partial Least Squares (PLS)
    pls_reg = PLSRegression(n_components=n_components) 
    pls_reg.fit(X_pca, Y_pca)
    Y_pred_pls = pls_reg.predict(X_pca)
    
    r2_pls = r2_score(Y_pca, Y_pred_pls) # PLS score는 R^2와 다를 수 있어 r2_score로 통일
    mse_pls = mean_squared_error(Y_pca, Y_pred_pls)
    rmse_pls = np.sqrt(mse_pls)
    mae_pls = mean_absolute_error(Y_pca, Y_pred_pls)
    adj_r2_pls = adjusted_r2(r2_pls, len(Y_pca), n_components)

    # ---------------------------------------------------------
    # 결과 출력
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(f"🔥 FINAL LINEARITY METRICS (PCA-{n_components})")
    print("="*80)
    
    print(f"1. CCA (Canonical Correlation Analysis)")
    print(f"   - Max Canonical Correlation: {corr_cca:.4f}")
    print("-" * 40)
    
    print(f"2. Multiple Linear Regression (MLR)")
    print(f"   - R² Score:         {r2_mlr:.4f}")
    print(f"   - Adjusted R²:      {adj_r2_mlr:.4f}")
    print(f"   - MSE:              {mse_mlr:.4f}")
    print(f"   - RMSE:             {rmse_mlr:.4f}")
    print(f"   - MAE:              {mae_mlr:.4f}")
    print("-" * 40)
    
    print(f"3. Partial Least Squares Regression (PLS)")
    print(f"   - R² Score:         {r2_pls:.4f}")
    print(f"   - Adjusted R²:      {adj_r2_pls:.4f}")
    print(f"   - MSE:              {mse_pls:.4f}")
    print(f"   - RMSE:             {rmse_pls:.4f}")
    print(f"   - MAE:              {mae_pls:.4f}")
    print("="*80)
    
    # 시각화 (CCA)
    plt.figure(figsize=(6, 6))
    plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.6, label='Samples')
    m, b = np.polyfit(X_c[:, 0], Y_c[:, 0], 1)
    plt.plot(X_c[:, 0], m*X_c[:, 0] + b, 'r--', label=f'Fit (Corr={corr_cca:.2f})')
    plt.xlabel("TS (PCA+CCA)"); plt.ylabel("BS (PCA+CCA)")
    plt.title(f"Linearity Check (PCA-{n_components})\nCCA Correlation={corr_cca:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("wsi_linearity_metrics_check.png") 
    print("📸 Graph saved as wsi_linearity_metrics_check.png")


if __name__ == "__main__":
    main()