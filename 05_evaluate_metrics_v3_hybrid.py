#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_evaluate_metrics_v3_hybrid.py

Goal: Evaluate the 'Guide-Conditioned Direct Diffusion (V3)' model.
Crucial Difference from V1/V2:
  1. Model Input: Accepts 'guide_hint' (Compass Prediction) as input condition.
  2. Sampling: Does NOT add the guide at the end. (Direct Prediction)
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# 1. Metric Helper Functions
# -----------------------------------------------------------------------------
def calculate_metrics_pointwise(pred, target):
    mse = F.mse_loss(pred, target).item()
    cos = F.cosine_similarity(pred, target).mean().item()
    return mse, cos

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
            raise ValueError("Imaginary component in FID calculation")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(act1, act2):
    act1 = act1.cpu().numpy()
    act2 = act2.cpu().numpy()
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def polynomial_kernel(X, Y):
    gamma = 1.0 / X.shape[1]
    K = (gamma * torch.mm(X, Y.t()) + 1.0) ** 3
    return K

def calculate_kid(act1, act2, subsets=10, subset_size=1000):
    n_samples = len(act1)
    if n_samples < subset_size:
        subset_size = n_samples
        subsets = 1
    kid_values = []
    for _ in range(subsets):
        idx1 = torch.randperm(len(act1))[:subset_size]
        idx2 = torch.randperm(len(act2))[:subset_size]
        X = act1[idx1]
        Y = act2[idx2]
        m, n = X.shape[0], Y.shape[0]
        K_XX = polynomial_kernel(X, X)
        K_YY = polynomial_kernel(Y, Y)
        K_XY = polynomial_kernel(X, Y)
        sum_XX = (K_XX.sum() - torch.trace(K_XX)) / (m * (m - 1))
        sum_YY = (K_YY.sum() - torch.trace(K_YY)) / (n * (n - 1))
        sum_XY = K_XY.sum() / (m * n)
        kid = sum_XX + sum_YY - 2 * sum_XY
        kid_values.append(kid.item())
    return np.mean(kid_values)

def calculate_wsi_metrics(pred, target, case_ids):
    unique_cases = torch.unique(case_ids)
    mse_list, cos_list = [], []
    for case in unique_cases:
        mask = (case_ids == case)
        p = pred[mask]
        t = target[mask]
        if p.shape[0] < 2: continue
        mse = F.mse_loss(p, t).item()
        cos = F.cosine_similarity(p, t).mean().item()
        mse_list.append(mse)
        cos_list.append(cos)
    return (np.mean(mse_list), np.std(mse_list), np.mean(cos_list), np.std(cos_list))

# -----------------------------------------------------------------------------
# 2. Model Definitions (Must Match V3 Training)
# -----------------------------------------------------------------------------
class CompassMLP(nn.Module):
    def __init__(self, dim, hidden=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class DiffusionSchedule:
    def __init__(self, timestamps=1000, beta_start=1e-4, beta_end=0.02):
        self.timestamps = timestamps
        self.betas = torch.linspace(beta_start, beta_end, timestamps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.lin2(self.drop(self.act(self.lin1(self.act(self.norm(x))))))
        return x + h

# --- [V3 Model] Guided Direct Diffusion ---
class GuidedDirectDiffusionMLP(nn.Module):
    def __init__(self, dim, hidden=1024, depth=6, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden // 4),
            nn.Linear(hidden // 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Input: Noisy(dim) + TS(dim) + Hint(dim)
        self.input_proj = nn.Linear(dim * 3, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, t, condition, guide_hint):
        t_emb = self.time_mlp(t)
        h = self.input_proj(torch.cat([x, condition, guide_hint], dim=1))
        for block in self.blocks:
            h = h + t_emb
            h = block(h)
        return self.output_proj(h)

# --- [Baseline Model] Diffusion Only (Optional comparison) ---
class ConditionalDiffusionMLP(nn.Module):
    def __init__(self, dim, hidden=1024, depth=6, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden // 4),
            nn.Linear(hidden // 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.input_proj = nn.Linear(dim * 2, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))
    def forward(self, x, t, condition):
        t_emb = self.time_mlp(t)
        h = self.input_proj(torch.cat([x, condition], dim=1))
        for block in self.blocks:
            h = h + t_emb
            h = block(h)
        return self.output_proj(h)

# -----------------------------------------------------------------------------
# 3. Sampling Logic (V3)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_hybrid(model, compass, scheduler, ts_input, device):
    model.eval()
    compass.eval()
    B, D = ts_input.shape
    
    # 1. Get Hint (Compass)
    guide_hint = compass(ts_input)
    # raw_guide = compass(ts_input)
    # guide_hint = 0.8 * raw_guide + 0.2 * ts_input
    
    # alpha = 0.7
    # pred_guide = compass(ts_input)
    # guide_hint = alpha * pred_guide + (1 - alpha) * ts_input
    
    
    # 2. Start from Pure Noise
    img = torch.randn(B, D, device=device)
    
    for i in reversed(range(0, scheduler.timestamps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        
        # [KEY] Feed hint as input
        pred_noise = model(img, t, condition=ts_input, guide_hint=guide_hint)
        
        alpha = scheduler.alphas[i]
        alpha_hat = scheduler.alphas_cumprod[i]
        beta = scheduler.betas[i]
        noise_factor = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
        mean = (1 / torch.sqrt(alpha)) * (img - noise_factor * pred_noise)
        
        if i > 0:
            noise = torch.randn_like(img)
            img = mean + torch.sqrt(beta) * noise
        else:
            img = mean
            
    # [KEY] Return img directly (Do NOT add guide_hint)
    return img

@torch.no_grad()
def sample_diffusion(model, scheduler, condition, device):
    model.eval()
    B, D = condition.shape
    img = torch.randn(B, D, device=device)
    for i in reversed(range(0, scheduler.timestamps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        pred_noise = model(img, t, condition)
        alpha = scheduler.alphas[i]
        alpha_hat = scheduler.alphas_cumprod[i]
        beta = scheduler.betas[i]
        noise_factor = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
        mean = (1 / torch.sqrt(alpha)) * (img - noise_factor * pred_noise)
        if i > 0:
            noise = torch.randn_like(img)
            img = mean + torch.sqrt(beta) * noise
        else:
            img = mean
    return img

# -----------------------------------------------------------------------------
# 4. Main Evaluation
# -----------------------------------------------------------------------------
def load_pt_with_case(path):
    print(f"📂 Loading {path}...")
    obj = torch.load(path, map_location="cpu")
    x, y, case_ids = None, None, None
    if isinstance(obj, dict):
        x = obj.get("x") or obj.get("X")
        y = obj.get("y") or obj.get("Y")
        for k in ["case_ids", "case_id", "case"]:
            if k in obj: case_ids = obj[k]; break
    elif isinstance(obj, (tuple, list)):
        x, y = obj[0], obj[1]
        if len(obj) >= 3: case_ids = obj[2]
            
    if not torch.is_tensor(x): x = torch.tensor(x)
    if not torch.is_tensor(y): y = torch.tensor(y)
    
    if case_ids is not None:
        if isinstance(case_ids, list) and isinstance(case_ids[0], str):
            uniq = sorted(set(case_ids))
            mapping = {cid: i for i, cid in enumerate(uniq)}
            case_ids = torch.tensor([mapping[c] for c in case_ids], dtype=torch.long)
        elif not torch.is_tensor(case_ids):
            case_ids = torch.tensor(case_ids, dtype=torch.long)
    return x.float(), y.float(), case_ids

def main():
    
    '''
    python 05_evaluate_metrics_v3_hybrid.py --test_pt data/ts2bs_raw_dataset_512_test.pt --compass_ckpt ./runs_compass_v2/best_compass.pt --diffusion_only_ckpt ./runs_diffusion_without_guided/best_diffusion_standard.pt --diffusion_ckpt runs_diffusion_v3_hybrid/best_hybrid_diffusion.pt
    
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--compass_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True, help="Ours V3 Checkpoint")
    parser.add_argument("--diffusion_only_ckpt", type=str, default=None, help="Baseline Checkpoint")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    x_test, y_test, case_ids = load_pt_with_case(args.test_pt)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    if case_ids is not None: case_ids = case_ids.to(device)
    dim = x_test.shape[1]

    # 2. Load Models
    print("⚡ Loading Models...")
    
    # Compass (for Hint)
    compass = CompassMLP(dim).to(device)
    ckpt_c = torch.load(args.compass_ckpt, map_location="cpu")
    compass.load_state_dict(ckpt_c['model'] if 'model' in ckpt_c else ckpt_c)
    compass.eval()
    
    # Ours V3 (Hybrid)
    print("   -> Loading Ours (V3 Hybrid)...")
    model_v3 = GuidedDirectDiffusionMLP(dim).to(device)
    ckpt_v3 = torch.load(args.diffusion_ckpt, map_location="cpu")
    model_v3.load_state_dict(ckpt_v3)
    model_v3.eval()
    scheduler = DiffusionSchedule().to(device)

    # Baseline (Optional)
    model_base = None
    if args.diffusion_only_ckpt:
        print("   -> Loading Baseline (Diffusion Only)...")
        model_base = ConditionalDiffusionMLP(dim).to(device)
        ckpt_base = torch.load(args.diffusion_only_ckpt, map_location="cpu")
        model_base.load_state_dict(ckpt_base)
        model_base.eval()

    # 3. Generate
    print("🌊 Generating comparison groups...")
    
    # Group 1: Ours V3
    print("   -> Running Ours V3 (Hybrid)...")
    gen_list = []
    n_batches = math.ceil(len(x_test) / args.batch_size)
    for i in range(n_batches):
        start = i * args.batch_size
        end = min((i + 1) * args.batch_size, len(x_test))
        batch_x = x_test[start:end]
        batch_gen = sample_hybrid(model_v3, compass, scheduler, batch_x, device)
        gen_list.append(batch_gen)
    x_ours = torch.cat(gen_list, dim=0)

    # Group 2: Baseline
    x_base = None
    if model_base:
        print("   -> Running Baseline...")
        gen_list = []
        for i in range(n_batches):
            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, len(x_test))
            batch_x = x_test[start:end]
            batch_gen = sample_diffusion(model_base, scheduler, batch_x, device)
            gen_list.append(batch_gen)
        x_base = torch.cat(gen_list, dim=0)

    # 4. Metrics
    comparisons = [("Ours (V3 Hybrid)", x_ours)]
    if x_base is not None:
        comparisons.insert(0, ("Baseline (DiffOnly)", x_base))

    print("\n" + "="*115)
    print(f"{'Method':<20} | {'FID':<8} | {'KID(e-2)':<8} || {'Global COS':<10} || {'WSI COS (Mean ± Std)':<25} | {'WSI MSE (Mean ± Std)':<25}")
    print("-" * 115)

    for name, pred in comparisons:
        _, cos_g = calculate_metrics_pointwise(pred, y_test)
        fid = calculate_fid(pred, y_test)
        kid = calculate_kid(pred, y_test) * 100
        
        if case_ids is not None:
            wsi_mse_m, wsi_mse_s, wsi_cos_m, wsi_cos_s = calculate_wsi_metrics(pred, y_test, case_ids)
            wsi_cos_str = f"{wsi_cos_m:.4f} ± {wsi_cos_s:.4f}"
            wsi_mse_str = f"{wsi_mse_m:.4f} ± {wsi_mse_s:.4f}"
        else:
            wsi_cos_str = "N/A"
            wsi_mse_str = "N/A"

        print(f"{name:<20} | {fid:.4f}   | {kid:.4f}     || {cos_g:.4f}     || {wsi_cos_str:<25} | {wsi_mse_str:<25}")

    print("="*115 + "\n")

if __name__ == "__main__":
    main()
    
'''

python 05_evaluate_metrics_v3_hybrid.py --test_pt data/ts2bs_raw_dataset_512_test.pt --compass_ckpt ./runs_compass_v2/best_compass.pt --diffusion_only_ckpt ./runs_diffusion_without_guided/best_diffusion_standard.pt --diffusion_ckpt runs_diffusion_v3_robust/best_robust_model.pt


python 05_evaluate_metrics_v3_hybrid.py --test_pt data/ts2bs_raw_dataset_512_test.pt --compass_ckpt ./runs_compass_v2/best_compass.pt --diffusion_only_ckpt ./runs_diffusion_without_guided/best_diffusion_standard.pt --diffusion_ckpt runs_diffusion_v3_aggressive/best_aggressive.pt


'''