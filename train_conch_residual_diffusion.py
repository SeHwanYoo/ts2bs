import json
import random
import pickle
import math
import glob
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import os

# ================= [설정] =================
CONFIG = {
    "TS_ROOT": "/mnt/d/workspace/dataset/brain/TS_CONCH", 
    "BS_ROOT": "/mnt/d/workspace/dataset/brain/BS_CONCH",
    "JSON_PATH": "./curriculum_case_matching.json",
    "PRECOMPUTED_DATA": "./ts2bs_pca_dataset_64_filtered.pt", 
    "PCA_SCALER_PATH": "./pca_scalers_64.pkl",
    "SAVE_DIR": "./checkpoints_full_stack",
    
    "PCA_DIM": 64,  # 128 -> 64 (Top 95% variance)
    "SAMPLES_PER_PATIENT": 1500,
    "CANDIDATE_MULTIPLIER": 3,
    "MAX_FILES_FOR_PCA": 100,
    
    # ★ Quality filtering (set ENABLE_QUALITY_FILTER=False to disable)
    "ENABLE_QUALITY_FILTER": False,  # True면 quality filtering 적용
    "MIN_NORM": 1.0,              
    "MIN_VARIANCE": 0.001,        
    "MIN_NONZERO_RATIO": 0.3,     
    "MAX_DYNAMIC_RANGE": 10000,
    
    "MLP_HIDDEN": 512, "MLP_LR": 1e-3, "MLP_EPOCHS": 300,
    "MLP_CKPT": "./checkpoints_full_stack/best_mlp_guide.pt",
    
    # ★ Unpaired data에서는 500 steps로 충분
    "DIFF_HIDDEN": 512, "DIFF_LAYERS": 6, 
    "TIMESTEPS": 500,  # 1000 -> 500
    "DIFF_LR": 1e-4, "DIFF_EPOCHS": 2000, "BATCH_SIZE": 2048,
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

# ================= [EMA 클래스 - 고친 버전] =================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Evaluation 전에 호출: EMA weights 적용"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Training 재개 전에 호출: 원래 weights 복원"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

# ================= [DDPMScheduler - 고친 버전] =================
class DDPMScheduler:
    def __init__(self, steps=1000, device='cuda'):
        self.steps = steps
        self.device = device
        
        # ★ FIX 2: Cosine Schedule 제대로 구현
        s = 0.008
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # t=0부터 t=T까지 모두 저장 (indexing 안전하게)
        self.alphas_cumprod = alphas_cumprod.to(device)
        
        # Betas 계산 (alpha_t = alpha_cumprod_t / alpha_cumprod_{t-1})
        self.betas = 1 - (self.alphas_cumprod[1:] / self.alphas_cumprod[:-1])
        self.alphas = 1 - self.betas

    def add_noise(self, x0, t):
        """Forward process: q(x_t | x_0)"""
        noise = torch.randn_like(x0)
        
        # t+1을 사용 (t=0일 때 alphas_cumprod[1] 사용)
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t + 1]).view(-1, 1)
        sqrt_one_minus = torch.sqrt(1 - self.alphas_cumprod[t + 1]).view(-1, 1)
        
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def sample(self, model, mlp_cond, ts_cond, shape, num_inference_steps=None):
        """
        DDIM-style sampling with fewer steps
        mlp_cond: MLP guide
        ts_cond: TS feature (for stronger conditioning)
        """
        if num_inference_steps is None:
            num_inference_steps = self.steps
            
        # ★ DDIM-style skip sampling
        if num_inference_steps < self.steps:
            step_ratio = self.steps // num_inference_steps
            timesteps = list(range(0, self.steps, step_ratio))
        else:
            timesteps = list(range(self.steps))
            
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            # 1. Model predicts x0 with both conditions
            pred_x0 = model(x, t, mlp_cond, ts_cond)
            pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
            
            # 2. Parameters (t+1 indexing)
            alpha_cumprod_t = self.alphas_cumprod[i + 1]
            
            if i > 0:
                alpha_cumprod_prev = self.alphas_cumprod[i]
            else:
                alpha_cumprod_prev = self.alphas_cumprod[0]
            
            # 3. Posterior mean & variance
            beta_t = 1 - alpha_cumprod_t / (alpha_cumprod_prev + 1e-8)
            alpha_t = alpha_cumprod_t / (alpha_cumprod_prev + 1e-8)
            
            coeff1 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t + 1e-8)
            coeff2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t + 1e-8)
            
            posterior_mean = coeff1 * pred_x0 + coeff2 * x
            posterior_variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t + 1e-8)
            
            # 4. Sampling
            if i > 0:
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance + 1e-8) * noise
            else:
                x = posterior_mean
                
        return x

# ================= [병렬 처리용 함수] =================
def process_patient_data(args):
    item, config, scaler_x, scaler_y, pca_x, pca_y = args
    ts_files, bs_files = item["ts_files"], item["bs_files"]
    n_total_pairs = min(len(ts_files), len(bs_files))
    if n_total_pairs == 0: return None
    
    target_count = config["SAMPLES_PER_PATIENT"]
    pool_size = target_count * config["CANDIDATE_MULTIPLIER"]
    
    if n_total_pairs > pool_size: candidate_indices = random.sample(range(n_total_pairs), pool_size)
    else: candidate_indices = range(n_total_pairs)
    
    valid_pairs = [] 
    for i in candidate_indices:
        ts_path = os.path.join(config["TS_ROOT"], ts_files[i])
        bs_path = os.path.join(config["BS_ROOT"], bs_files[i])
        try:
            v_ts = load_feature_vector(ts_path)
            v_bs = load_feature_vector(bs_path)
            if v_ts is not None and v_bs is not None:
                ts_np_raw = v_ts.numpy().reshape(1, -1)
                bs_np_raw = v_bs.numpy().reshape(1, -1)
                
                # ========== Quality Metrics ==========
                # 1. Feature norm (information content)
                norm_ts = np.linalg.norm(ts_np_raw)
                norm_bs = np.linalg.norm(bs_np_raw)
                
                # Cosine similarity (for pairing)
                similarity = np.dot(ts_np_raw, bs_np_raw.T).item() / (norm_ts * norm_bs + 1e-8)
                
                # ========== Optional Quality Filtering ==========
                if config.get("ENABLE_QUALITY_FILTER", False):
                    # 2. Feature variance (diversity)
                    var_ts = np.var(ts_np_raw)
                    var_bs = np.var(bs_np_raw)
                    
                    # 3. Non-zero ratio (sparsity check)
                    nonzero_ratio_ts = np.count_nonzero(np.abs(ts_np_raw) > 1e-5) / ts_np_raw.size
                    nonzero_ratio_bs = np.count_nonzero(np.abs(bs_np_raw) > 1e-5) / bs_np_raw.size
                    
                    # 4. Dynamic range (max/min ratio)
                    ts_abs = np.abs(ts_np_raw[ts_np_raw != 0])
                    bs_abs = np.abs(bs_np_raw[bs_np_raw != 0])
                    dynamic_range_ts = ts_abs.max() / (ts_abs.min() + 1e-8) if len(ts_abs) > 0 else 0
                    dynamic_range_bs = bs_abs.max() / (bs_abs.min() + 1e-8) if len(bs_abs) > 0 else 0
                    
                    # Quality checks
                    if norm_ts < config.get("MIN_NORM", 1.0) or norm_bs < config.get("MIN_NORM", 1.0):
                        continue
                    if var_ts < config.get("MIN_VARIANCE", 0.001) or var_bs < config.get("MIN_VARIANCE", 0.001):
                        continue
                    if nonzero_ratio_ts < config.get("MIN_NONZERO_RATIO", 0.3) or \
                       nonzero_ratio_bs < config.get("MIN_NONZERO_RATIO", 0.3):
                        continue
                    if dynamic_range_ts > config.get("MAX_DYNAMIC_RANGE", 10000) or \
                       dynamic_range_bs > config.get("MAX_DYNAMIC_RANGE", 10000):
                        continue
                    
                    # Quality score
                    quality_score = (
                        0.3 * (norm_ts + norm_bs) / 2 +
                        0.3 * (var_ts + var_bs) / 2 +
                        0.2 * (nonzero_ratio_ts + nonzero_ratio_bs) / 2 +
                        0.2 * (1.0 / (1.0 + abs(norm_ts - norm_bs)))
                    )
                else:
                    # No quality filtering - use norm as simple quality proxy
                    quality_score = (norm_ts + norm_bs) / 2
                
                t_pca = pca_x.transform(scaler_x.transform(ts_np_raw))
                b_pca = pca_y.transform(scaler_y.transform(bs_np_raw))
                
                # Store with both quality and similarity
                valid_pairs.append((quality_score, similarity, t_pca, b_pca))
        except: continue
    
    if not valid_pairs: return None
    
    # Sort by quality score first (descending), then by similarity
    valid_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Take top quality pairs
    final_pairs = valid_pairs[:target_count]
        
    return np.vstack([p[2] for p in final_pairs]), np.vstack([p[3] for p in final_pairs])

# ================= [유틸리티] =================
def extract_case_id(filename):
    """
    Extract case ID from filename or folder name
    Example: 
      TCGA-76-6656-01A-01-TS1.xxx_391_163/TCGA-76-6656-01A-01-TS1.xxx_0_130.pt
      → TCGA-76-6656-01A-01
    """
    # Try folder name first (more reliable)
    folder = os.path.basename(os.path.dirname(filename))
    if folder and folder.startswith("TCGA"):
        if "-TS" in folder: return folder.split("-TS")[0]
        elif "-BS" in folder: return folder.split("-BS")[0]
    
    # Fallback to filename
    name = os.path.basename(filename)
    if "-TS" in name: return name.split("-TS")[0]
    elif "-BS" in name: return name.split("-BS")[0]
    
    # Last resort: split by dot
    return name.split(".")[0]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_feature_vector(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            for k in ["feat", "feats", "features"]:
                if k in obj: obj = obj[k]; break
            else: obj = list(obj.values())[0]
        t = obj.float()
        if t.dim() == 3: return t.mean(dim=(1, 2))
        if t.dim() == 2: return t.mean(dim=0)
        return t
    except: return None

# ================= [전처리 파이프라인] =================
def generate_curriculum(ts_root, bs_root, save_path):
    print(f"\n📂 Scanning for Case ID Matching...")
    ts_files = glob.glob(os.path.join(ts_root, "**", "*.pt"), recursive=True)
    bs_files = glob.glob(os.path.join(bs_root, "**", "*.pt"), recursive=True)
    
    bs_map = {}
    for f in tqdm(bs_files, desc="BS Index"):
        cid = extract_case_id(f)
        if cid not in bs_map: bs_map[cid] = []
        bs_map[cid].append(os.path.relpath(f, bs_root))

    matched = {}
    for f in tqdm(ts_files, desc="TS Match"):
        cid = extract_case_id(f)
        if cid in bs_map:
            if cid not in matched: matched[cid] = {"case_id": cid, "ts_files": [], "bs_files": bs_map[cid]}
            matched[cid]["ts_files"].append(os.path.relpath(f, ts_root))
            
    curr = list(matched.values())
    print(f"✅ Matched {len(curr)} Cases.")
    with open(save_path, "w") as f: json.dump(curr, f, indent=4)
    return curr

def collect_pca_samples(curriculum, ts_root, bs_root, max_files):
    X, Y = [], []
    for item in curriculum:
        ts, bs = item["ts_files"], item["bs_files"]
        n = min(len(ts), len(bs), max_files)
        if n==0: continue
        idx_t = random.sample(range(len(ts)), n)
        idx_b = random.sample(range(len(bs)), n)
        for t, b in zip(idx_t, idx_b):
            vt = load_feature_vector(os.path.join(ts_root, ts[t]))
            vb = load_feature_vector(os.path.join(bs_root, bs[b]))
            if vt is not None and vb is not None:
                X.append(vt.numpy()); Y.append(vb.numpy())
    return np.stack(X), np.stack(Y)

def fit_pca(curriculum):
    X, Y = collect_pca_samples(curriculum, CONFIG["TS_ROOT"], CONFIG["BS_ROOT"], CONFIG["MAX_FILES_FOR_PCA"])
    print(f"🔧 Fitting PCA...")
    sx, sy = StandardScaler().fit(X), StandardScaler().fit(Y)
    px = PCA(n_components=CONFIG['PCA_DIM']).fit(sx.transform(X))
    py = PCA(n_components=CONFIG['PCA_DIM']).fit(sy.transform(Y))
    return sx, sy, px, py

def create_dataset(curriculum, sx, sy, px, py):
    print(f"🚜 Creating Dataset (Parallel)...")
    tasks = [(item, CONFIG, sx, sy, px, py) for item in curriculum]
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    all_ts, all_bs = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(process_patient_data, tasks), total=len(tasks)):
            if res:
                all_ts.append(torch.from_numpy(res[0]).float())
                all_bs.append(torch.from_numpy(res[1]).float())
                
    X, Y = torch.cat(all_ts, 0), torch.cat(all_bs, 0)
    print(f"✅ Dataset Size: {X.shape}")
    torch.save({"X": X, "Y": Y}, CONFIG["PRECOMPUTED_DATA"])
    return X, Y

# ================= [모델] =================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return x + self.linear2(self.dropout(self.act(self.linear1(self.act(self.norm(x))))))

class MLPGuide(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden),
            nn.LayerNorm(hidden), nn.Linear(hidden, dim)
        )
    def forward(self, x): return self.net(x)

class DiffusionDenoiser(nn.Module):
    def __init__(self, dim, hidden, depth):
        super().__init__()
        # ★ TS feature도 직접 conditioning
        self.input_proj = nn.Sequential(
            nn.Linear(dim*3, hidden),  # x + mlp_guide + ts_raw
            nn.SiLU(), 
            nn.Linear(hidden, hidden)
        )
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.final = nn.Linear(hidden, dim)
        nn.init.zeros_(self.final.weight)
        
    def forward(self, x, t, mlp_cond, ts_raw):
        """
        x: noisy BS
        mlp_cond: MLP guide (normalized)
        ts_raw: TS feature (normalized) - 직접 conditioning
        """
        t_emb = self.time_mlp(t)
        x = self.input_proj(torch.cat([x, mlp_cond, ts_raw], dim=1))
        for b in self.blocks: x = x + t_emb + b(x)
        return self.final(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        emb = x[:, None] * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# ================= [학습 루프] =================
def train_mlp(X, Y, device, data_mean, data_std):
    """MLP를 normalized space에서 학습 (TS도 normalize)"""
    print(f"\n🚀 Training MLP Guide (Normalized Space)...")
    
    # ★ X (TS)도 normalize
    X_norm = (X.to(device) - data_mean) / data_std
    Y_norm = (Y.to(device) - data_mean) / data_std
    
    ds = TensorDataset(X_norm, Y_norm)
    dl = DataLoader(ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    model = MLPGuide(CONFIG["PCA_DIM"], CONFIG["MLP_HIDDEN"]).to(device)
    optim = AdamW(model.parameters(), lr=CONFIG["MLP_LR"])
    
    for ep in range(CONFIG["MLP_EPOCHS"]):
        model.train()
        for x_norm, y_norm in dl:
            x_norm, y_norm = x_norm.to(device), y_norm.to(device)
            pred_norm = model(x_norm)
            loss = F.mse_loss(pred_norm, y_norm)
            optim.zero_grad(); loss.backward(); optim.step()
        if (ep+1)%10==0: print(f"   [MLP Ep {ep+1}] Loss: {loss.item():.5f}")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "data_mean": data_mean.cpu(),
        "data_std": data_std.cpu()
    }, CONFIG["MLP_CKPT"])
    return model

def main():
    multiprocessing.set_start_method('spawn', force=True)
    set_seed(42)
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    device = torch.device(CONFIG["DEVICE"])
    print(f"🔥 Full Stack Pipeline (FIXED: 1000 steps + proper EMA) on {device}")
    
    if not os.path.exists(CONFIG["JSON_PATH"]): 
        generate_curriculum(CONFIG["TS_ROOT"], CONFIG["BS_ROOT"], CONFIG["JSON_PATH"])
    with open(CONFIG["JSON_PATH"], 'r') as f: curr = json.load(f)
    
    if not os.path.exists(CONFIG["PCA_SCALER_PATH"]): 
        sx, sy, px, py = fit_pca(curr)
        with open(CONFIG["PCA_SCALER_PATH"], 'wb') as f: pickle.dump((sx, sy, px, py), f)
    else: 
        with open(CONFIG["PCA_SCALER_PATH"], 'rb') as f: sx, sy, px, py = pickle.load(f)
        
    if not os.path.exists(CONFIG["PRECOMPUTED_DATA"]): create_dataset(curr, sx, sy, px, py)
    
    print(f"📂 Loading Dataset...")
    d = torch.load(CONFIG["PRECOMPUTED_DATA"])
    X, Y = d["X"], d["Y"]
    
    print("📊 Computing Statistics...")
    all_bs = Y.to(device)
    DATA_MEAN = all_bs.mean(dim=0)
    DATA_STD_RAW = all_bs.std(dim=0) + 1e-6
    
    # ★ Option 1: Percentile-based clipping (더 강력)
    std_95_percentile = torch.quantile(DATA_STD_RAW, 0.95)
    DATA_STD = torch.clamp(DATA_STD_RAW, max=std_95_percentile)
    
    # ★ Option 2: Median-based clipping (원본)
    # std_median = DATA_STD_RAW.median()
    # std_threshold = std_median * 3
    # DATA_STD = torch.clamp(DATA_STD_RAW, max=std_threshold)
    
    clipped_dims = (DATA_STD_RAW > DATA_STD).sum()
    print(f"   Mean Norm: {DATA_MEAN.norm():.2f}")
    print(f"   Std Norm (original): {DATA_STD_RAW.norm():.2f}")
    print(f"   Std Norm (clipped):  {DATA_STD.norm():.2f}")
    print(f"   95th percentile std: {std_95_percentile:.4f}")
    print(f"   Clipped dimensions: {clipped_dims}/128")
    
    # ★ MLP도 normalized space에서 학습
    mlp = MLPGuide(CONFIG["PCA_DIM"], CONFIG["MLP_HIDDEN"]).to(device)
    if not os.path.exists(CONFIG["MLP_CKPT"]): 
        mlp = train_mlp(X, Y, device, DATA_MEAN, DATA_STD)
    else: 
        ckpt = torch.load(CONFIG["MLP_CKPT"], map_location=device)
        mlp.load_state_dict(ckpt["model_state_dict"])
        # Verify stats match
        if "data_mean" in ckpt:
            saved_mean = ckpt["data_mean"].to(device)
            saved_std = ckpt["data_std"].to(device)
            if not torch.allclose(DATA_MEAN, saved_mean, atol=1e-3):
                print("⚠️ Warning: DATA_MEAN mismatch! Retraining MLP...")
                mlp = train_mlp(X, Y, device, DATA_MEAN, DATA_STD)
    mlp.eval()
    
    print("\n🚀 Starting Diffusion Training...")
    
    # Train/Val split
    n = len(X)
    train_size = int(0.95 * n)
    indices = torch.randperm(n)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    model = DiffusionDenoiser(CONFIG["PCA_DIM"], CONFIG["DIFF_HIDDEN"], CONFIG["DIFF_LAYERS"]).to(device)
    ema = EMA(model, decay=0.9999)
    sched = DDPMScheduler(CONFIG["TIMESTEPS"], device)
    optim = AdamW(model.parameters(), lr=CONFIG["DIFF_LR"], weight_decay=0.01)  # Weight decay 추가
    
    best_val_mse = float('inf')
    
    for ep in range(CONFIG["DIFF_EPOCHS"]):
        # ===== Training Phase =====
        model.train()
        pbar = tqdm(train_dl, desc=f"Diff Ep {ep+1}")
        
        for ts, bs in pbar:
            ts, bs = ts.to(device), bs.to(device)
            
            # Normalize inputs
            ts_norm = (ts - DATA_MEAN) / DATA_STD
            bs_norm = (bs - DATA_MEAN) / DATA_STD
            
            # ★ FIX: MLP는 normalized input을 받아야 함!
            with torch.no_grad(): 
                mlp_norm = mlp(ts_norm)  # ts → ts_norm
            
            # Add noise
            t = torch.randint(0, CONFIG["TIMESTEPS"], (bs.size(0),), device=device).long()
            xt, noise = sched.add_noise(bs_norm, t)
            
            # ★ Pass both MLP guide and TS raw
            pred_x0 = model(xt, t, mlp_norm, ts_norm)
            loss = F.mse_loss(pred_x0, bs_norm)
            
            optim.zero_grad(); loss.backward(); optim.step()
            ema.update()
            
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})
        
        # ===== Validation Phase (every 10 epochs) =====
        if (ep+1) % 50 == 0:
            ema.apply_shadow()  # EMA 적용
            model.eval()
            
            val_mlp_mse = 0.0
            val_diff_mse = 0.0
            val_denoiser_mse = 0.0  # ★ 직접 denoiser 평가
            num_samples = min(100, len(val_ds))
            
            with torch.no_grad():
                for i in range(num_samples):
                    ts_val, bs_val = val_ds[i]
                    ts_val = ts_val.unsqueeze(0).to(device)
                    bs_val = bs_val.unsqueeze(0).to(device)
                    
                    # Normalize
                    ts_norm = (ts_val - DATA_MEAN) / DATA_STD
                    bs_norm = (bs_val - DATA_MEAN) / DATA_STD
                    
                    # MLP guide (input도 normalized)
                    mlp_norm = mlp(ts_norm)
                    mlp_out = mlp_norm * DATA_STD + DATA_MEAN
                    
                    # ★ Direct denoiser evaluation (no sampling)
                    t_random = torch.randint(0, CONFIG["TIMESTEPS"], (1,), device=device).long()
                    xt_random, _ = sched.add_noise(bs_norm, t_random)
                    pred_x0_direct = model(xt_random, t_random, mlp_norm, ts_norm)
                    val_denoiser_mse += F.mse_loss(pred_x0_direct, bs_norm).item()
                    
                    # ★ Full sampling (num_inference_steps=None for full steps)
                    gen_norm = sched.sample(model, mlp_norm, ts_norm, ts_val.shape, 
                                           num_inference_steps=None)  # Full steps
                    gen = gen_norm * DATA_STD + DATA_MEAN
                    
                    val_mlp_mse += F.mse_loss(mlp_out, bs_val).item()
                    val_diff_mse += F.mse_loss(gen, bs_val).item()
            
            val_mlp_mse /= num_samples
            val_diff_mse /= num_samples
            val_denoiser_mse /= num_samples
            
            print(f"\n   [Validation] MLP MSE: {val_mlp_mse:.5f} | "
                  f"Denoiser MSE: {val_denoiser_mse:.5f} | "
                  f"Sampling MSE: {val_diff_mse:.5f}")
            
            # Best model 저장
            if val_diff_mse < best_val_mse:
                best_val_mse = val_diff_mse
                torch.save({
                    'model_state': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'epoch': ep+1,
                    'val_mse': val_diff_mse
                }, f"{CONFIG['SAVE_DIR']}/best_diffusion.pt")
                print(f"   ✅ Best model saved! (Sampling MSE: {val_diff_mse:.5f})")
            
            torch.save(model.state_dict(), f"{CONFIG['SAVE_DIR']}/diff_ep{ep+1}.pt")
            
            ema.restore()  # 학습 재개

if __name__ == "__main__":
    main()