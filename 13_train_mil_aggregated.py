# import pandas as pd
# import numpy as np
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_curve
# from sklearn.model_selection import StratifiedGroupKFold

# # ==========================================
# # 1. 설정 (여기서 숫자를 바꾸세요!)
# # ==========================================
# BASE_DIR = r"D:\workspace\dataset\brain\BS_CONCH\aggregated_slides_v3"
# RESULT_FILE = r"D:\workspace\ts2bs_20251213\grand_final_10fold.xlsx"
# MODEL_DIR = r"D:\workspace\ts2bs_20251213\models\mil_10fold"

# EPOCHS = 30       # Fold가 많아지면(데이터 적어짐) Epoch 너무 길게 잡지 마세요
# LR = 2e-4
# N_SPLITS = 15     # 👈 여기를 10으로 바꾸면 모든 곳에 적용됩니다.
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# EXPERIMENTS = [
#     ("TS Only", "agg_TS.csv"),
#     ("BS Only", "agg_BS.csv"),
#     ("Combined (Real)", "agg_Combined.csv"),
#     ("Gen BS Only", "agg_GEN_BS.csv"),
#     ("Combined2 (TS+Gen)", "agg_TS_GEN_BS.csv")
# ]

# # ... (Dataset, Model 클래스는 기존과 동일하므로 생략) ...
# class BagDataset(Dataset):
#     def __init__(self, df):
#         self.df = df
#     def __len__(self):
#         return len(self.df)
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         try:
#             f = torch.load(row['file_path'], map_location='cpu')
#             if f.dim() == 1: f = f.unsqueeze(0)
#             return f, int(row['label'])
#         except:
#             return torch.zeros(1, 512), 0

# class GatedAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.L=128; self.D=64; self.K=1
#         self.fe = nn.Sequential(nn.Linear(512, self.L), nn.ReLU())
#         self.av = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
#         self.au = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
#         self.w = nn.Linear(self.D, self.K)
#         self.clf = nn.Sequential(nn.Linear(self.L*self.K, 1), nn.Sigmoid())
#     def forward(self, x):
#         x = x.squeeze(0)
#         f = self.fe(x)
#         A = self.w(self.av(f) * self.au(f))
#         A = torch.transpose(A, 1, 0)
#         A = nn.functional.softmax(A, dim=1)
#         M = torch.mm(A, f)
#         return self.clf(M), A

# # ==========================================
# # 2. 메트릭 계산 (안전장치 추가됨)
# # ==========================================
# def calculate_metrics(y_true, y_pred_prob):
#     # [안전장치] 데이터가 없거나 클래스가 하나뿐이면 계산 불가
#     if len(y_true) < 2 or len(np.unique(y_true)) < 2:
#         return {
#             "AUC": 0.5, "Accuracy": 0, "F1": 0, 
#             "Sensitivity": 0, "Specificity": 0, "Threshold": 0.5
#         }

#     # AUC
#     try: auc = roc_auc_score(y_true, y_pred_prob)
#     except: auc = 0.5

#     # Optimal Threshold (Youden Index)
#     try:
#         fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
#         J = tpr - fpr
#         ix = np.argmax(J)
#         best_thresh = thresholds[ix]
#     except:
#         best_thresh = 0.5
    
#     if best_thresh < 0.1 or best_thresh > 0.9: best_thresh = 0.5

#     y_pred = [1 if p >= best_thresh else 0 for p in y_pred_prob]
    
#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     sens = recall_score(y_true, y_pred, zero_division=0)
    
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0,0,0,0)
#     spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
#     return {
#         "AUC": auc, "Accuracy": acc, "F1": f1, 
#         "Sensitivity": sens, "Specificity": spec, "Threshold": best_thresh
#     }

# # ==========================================
# # 3. 공통 환자 찾기 (N_SPLITS 인자 적용)
# # ==========================================
# def prepare_common_folds(experiments, n_splits, random_state=42):
#     print(f"🔒 [데이터 검증] {n_splits}-Fold 분할을 위한 공통 환자 찾기...")
#     patient_sets = []
#     dfs = {}
    
#     for name, fname in experiments:
#         path = os.path.join(BASE_DIR, fname)
#         if os.path.exists(path):
#             df = pd.read_csv(path)
#             patient_sets.append(set(df['patient_id'].unique()))
#             dfs[name] = df
            
#     common_patients = sorted(list(set.intersection(*patient_sets)))
#     print(f"✅ 공통 환자 수: {len(common_patients)}명")

#     first_df = list(dfs.values())[0]
#     pid_to_label = first_df.groupby('patient_id')['label'].first().to_dict()
#     common_labels = [pid_to_label[pid] for pid in common_patients]

#     # 여기서 n_splits 변수를 사용함!
#     sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     patient_to_fold = {}
    
#     dummy_df = pd.DataFrame({'pid': common_patients, 'label': common_labels})
#     for fold, (train_idx, test_idx) in enumerate(sgkf.split(dummy_df, dummy_df['label'], dummy_df['pid'])):
#         test_pids = dummy_df.iloc[test_idx]['pid'].values
#         for pid in test_pids:
#             patient_to_fold[pid] = fold
            
#     return patient_to_fold

# # ==========================================
# # 4. 실행 함수 (안전장치 추가됨)
# # ==========================================
# def run_experiment_no_val(exp_name, filename, patient_to_fold, n_splits):
#     print(f"\n🧪 [{exp_name}] 실험 시작 ({n_splits} Fold - No Val)")
#     csv_path = os.path.join(BASE_DIR, filename)
#     df = pd.read_csv(csv_path)
    
#     df = df[df['patient_id'].isin(patient_to_fold.keys())].reset_index(drop=True)
    
#     fold_metrics = []
    
#     # 여기서 n_splits 변수만큼 돔!
#     for fold in range(n_splits):
#         current_fold_mask = df['patient_id'].map(patient_to_fold) == fold
        
#         test_df = df[current_fold_mask]
#         train_df = df[~current_fold_mask] 
        
#         # [중요] 빈 깡통 Fold 방지 (에러 원인 차단)
#         if len(test_df) == 0:
#             print(f"   ⚠️ Fold {fold+1}: 테스트 데이터가 없습니다 (Skip)")
#             continue

#         train_loader = DataLoader(BagDataset(train_df), batch_size=1, shuffle=True)
#         test_loader = DataLoader(BagDataset(test_df), batch_size=1, shuffle=False)
        
#         model = GatedAttention().to(DEVICE)
#         opt = optim.Adam(model.parameters(), LR)
#         crit = nn.BCELoss()
        
#         # Train
#         model.train()
#         for ep in range(EPOCHS):
#             for d, l in train_loader:
#                 d, l = d.to(DEVICE), l.to(DEVICE).float()
#                 opt.zero_grad()
#                 prob, _ = model(d)
#                 loss = crit(prob.view(-1), l.view(-1))
#                 loss.backward()
#                 opt.step()
                
#         # Test
#         model.eval()
#         probs, labels = [], []
#         with torch.no_grad():
#             for d, l in test_loader:
#                 d, l = d.to(DEVICE), l.to(DEVICE).float()
#                 probs.append(model(d)[0].item())
#                 labels.append(l.item())
        
#         # 결과 계산
#         unique, counts = np.unique(labels, return_counts=True)
#         dist_str = str(dict(zip(unique, counts)))
        
#         met = calculate_metrics(labels, probs)
#         print(f"   👉 Fold {fold+1}: AUC={met['AUC']:.4f} | Acc={met['Accuracy']:.4f} (Test: {len(labels)}명, {dist_str})")
#         fold_metrics.append(met)
        
#     if not fold_metrics: return None

#     avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
#     avg_metrics['Experiment'] = exp_name
#     print(f"   ✨ 평균 결과: AUC={avg_metrics['AUC']:.4f}, Acc={avg_metrics['Accuracy']:.4f}")
#     return avg_metrics

# # ==========================================
# # 5. 메인 실행
# # ==========================================
# if __name__ == "__main__":
#     # 1. 여기서 N_SPLITS(10)을 넘겨줍니다.
#     fold_map = prepare_common_folds(EXPERIMENTS, n_splits=N_SPLITS)
    
#     results = []
#     for name, fname in EXPERIMENTS:
#         # 2. 여기도 N_SPLITS(10)을 넘겨줍니다.
#         res = run_experiment_no_val(name, fname, fold_map, n_splits=N_SPLITS)
#         if res: results.append(res)
        
#     if results:
#         df_res = pd.DataFrame(results)
#         df_res.to_excel(RESULT_FILE, index=False)
#         print(f"\n💾 결과 저장 완료: {RESULT_FILE}")

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

# ==========================================
# 1. 설정 (15-Fold 적용됨)
# ==========================================
BASE_DIR = r"D:\workspace\dataset\brain\BS_CONCH\aggregated_slides_v3"
RESULT_FILE = r"D:\workspace\ts2bs_20251213\grand_final_15fold.xlsx"
MODEL_DIR = r"D:\workspace\ts2bs_20251213\models\mil_15fold"

EPOCHS = 30       # 데이터가 적고 Fold가 많으므로 30회면 충분
LR = 2e-4
N_SPLITS = 5     # 👈 15-Fold 설정 (여기만 바꾸면 전체 적용)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EXPERIMENTS = [
    ("TS Only", "agg_TS.csv"),
    ("BS Only", "agg_BS.csv"),
    ("Combined (Real)", "agg_Combined.csv"),
    ("Gen BS Only", "agg_GEN_BS.csv"),
    ("Combined2 (TS+Gen)", "agg_TS_GEN_BS.csv")
]

# ==========================================
# 2. 데이터셋 및 모델 클래스
# ==========================================
class BagDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            f = torch.load(row['file_path'], map_location='cpu')
            if f.dim() == 1: f = f.unsqueeze(0)
            return f, int(row['label'])
        except:
            return torch.zeros(1, 512), 0

class GatedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.L=128; self.D=64; self.K=1
        self.fe = nn.Sequential(nn.Linear(512, self.L), nn.ReLU())
        self.av = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.au = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.w = nn.Linear(self.D, self.K)
        self.clf = nn.Sequential(nn.Linear(self.L*self.K, 1), nn.Sigmoid())
    def forward(self, x):
        x = x.squeeze(0)
        f = self.fe(x)
        A = self.w(self.av(f) * self.au(f))
        A = torch.transpose(A, 1, 0)
        A = nn.functional.softmax(A, dim=1)
        M = torch.mm(A, f)
        return self.clf(M), A

# ==========================================
# 3. 메트릭 계산 (안전장치 + Optimal Threshold)
# ==========================================
def calculate_metrics(y_true, y_pred_prob):
    # [안전장치] 데이터가 없거나 클래스가 하나뿐이면 0.5 리턴
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return {
            "AUC": 0.5, "Accuracy": 0, "F1": 0, 
            "Sensitivity": 0, "Specificity": 0, "Threshold": 0.5
        }

    # AUC
    try: auc = roc_auc_score(y_true, y_pred_prob)
    except: auc = 0.5

    # Optimal Threshold (Youden Index)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    except:
        best_thresh = 0.5
    
    # 너무 극단적인 Threshold 보정
    if best_thresh < 0.1 or best_thresh > 0.9: best_thresh = 0.5

    y_pred = [1 if p >= best_thresh else 0 for p in y_pred_prob]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0,0,0,0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "AUC": auc, "Accuracy": acc, "F1": f1, 
        "Sensitivity": sens, "Specificity": spec, "Threshold": best_thresh
    }

# ==========================================
# 4. 공통 환자 찾기 및 Fold 지정
# ==========================================
def prepare_common_folds(experiments, n_splits, random_state=42):
    print(f"🔒 [데이터 검증] {n_splits}-Fold 분할을 위한 공통 환자 찾기...")
    patient_sets = []
    dfs = {}
    
    for name, fname in experiments:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            patient_sets.append(set(df['patient_id'].unique()))
            dfs[name] = df
            
    common_patients = sorted(list(set.intersection(*patient_sets)))
    print(f"✅ 공통 환자 수: {len(common_patients)}명")

    first_df = list(dfs.values())[0]
    pid_to_label = first_df.groupby('patient_id')['label'].first().to_dict()
    common_labels = [pid_to_label[pid] for pid in common_patients]

    # Stratified Group K-Fold (15개로 나눔)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    patient_to_fold = {}
    
    dummy_df = pd.DataFrame({'pid': common_patients, 'label': common_labels})
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(dummy_df, dummy_df['label'], dummy_df['pid'])):
        test_pids = dummy_df.iloc[test_idx]['pid'].values
        for pid in test_pids:
            patient_to_fold[pid] = fold
            
    return patient_to_fold

# ==========================================
# 5. 실험 실행 함수 (No Validation, Train 100%)
# ==========================================
def run_experiment_no_val(exp_name, filename, patient_to_fold, n_splits):
    print(f"\n🧪 [{exp_name}] 실험 시작 ({n_splits} Fold - No Val)")
    csv_path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(csv_path)
    
    # 공통 환자 필터링
    df = df[df['patient_id'].isin(patient_to_fold.keys())].reset_index(drop=True)
    
    fold_metrics = []
    
    # 0번부터 14번 Fold까지 순회
    for fold in range(n_splits):
        current_fold_mask = df['patient_id'].map(patient_to_fold) == fold
        
        test_df = df[current_fold_mask]
        train_df = df[~current_fold_mask] # 나머지 전부 Train
        
        # [예외처리] 혹시 Test 데이터가 0개면 스킵
        if len(test_df) == 0:
            print(f"   ⚠️ Fold {fold+1}: 테스트 데이터가 없습니다 (Skip)")
            continue

        train_loader = DataLoader(BagDataset(train_df), batch_size=1, shuffle=True)
        test_loader = DataLoader(BagDataset(test_df), batch_size=1, shuffle=False)
        
        model = GatedAttention().to(DEVICE)
        opt = optim.Adam(model.parameters(), LR)
        crit = nn.BCELoss()
        
        # Train Loop (Validation 없음)
        model.train()
        for ep in range(EPOCHS):
            for d, l in train_loader:
                d, l = d.to(DEVICE), l.to(DEVICE).float()
                opt.zero_grad()
                prob, _ = model(d)
                loss = crit(prob.view(-1), l.view(-1))
                loss.backward()
                opt.step()
                
        # Test Loop
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for d, l in test_loader:
                d, l = d.to(DEVICE), l.to(DEVICE).float()
                probs.append(model(d)[0].item())
                labels.append(l.item())
        
        # 결과 계산
        unique, counts = np.unique(labels, return_counts=True)
        dist_str = str(dict(zip(unique, counts)))
        
        met = calculate_metrics(labels, probs)
        print(f"   👉 Fold {fold+1}: AUC={met['AUC']:.4f} | Acc={met['Accuracy']:.4f} (Test: {len(labels)}명, {dist_str})")
        fold_metrics.append(met)
        
    if not fold_metrics: return None

    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
    avg_metrics['Experiment'] = exp_name
    print(f"   ✨ 평균 결과: AUC={avg_metrics['AUC']:.4f}, Acc={avg_metrics['Accuracy']:.4f}, F1={avg_metrics['F1']:.4f}")
    return avg_metrics

# ==========================================
# 6. 메인 실행
# ==========================================
if __name__ == "__main__":
    # 1. 15-Fold 지도를 만듭니다.
    fold_map = prepare_common_folds(EXPERIMENTS, n_splits=N_SPLITS)
    
    results = []
    for name, fname in EXPERIMENTS:
        # 2. 15-Fold 실험을 수행합니다.
        res = run_experiment_no_val(name, fname, fold_map, n_splits=N_SPLITS)
        if res: results.append(res)
        
    if results:
        df_res = pd.DataFrame(results)
        # 컬럼 순서 보기 좋게 정렬
        cols = ['Experiment', 'AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Threshold']
        final_cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[final_cols]
        
        df_res.to_excel(RESULT_FILE, index=False)
        print(f"\n💾 결과 저장 완료: {RESULT_FILE}")