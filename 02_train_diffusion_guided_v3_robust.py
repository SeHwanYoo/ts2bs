#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_diffusion_guided_v3_robust.py [Final Robust Version]

[Strategy: Dynamic Hint Mixing from Scratch]
Goal: Train a model that is robust across different cancer types by learning to:
  1. Generate high-quality texture independently (via Hint Dropout).
  2. Trust structural hints from the guide channel (via TS Injection).
  3. Follow the Compass for style transformation (via Standard Training).

Input: Noisy_BS + TS + Hint
Target: Real BS (Direct Prediction)
"""

import os
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# 1. Architecture Components (Standard)
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
        
        self.alphas = self.alphas.to(device)
        # self.betas = self.betas.to(device)

        return self

    def add_noise(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

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

# -----------------------------------------------------------------------------
# 2. Main Model: Guided Direct Diffusion
# -----------------------------------------------------------------------------
class GuidedDirectDiffusionMLP(nn.Module):
    def __init__(self, dim, hidden=1024, depth=6, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden // 4),
            nn.Linear(hidden // 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Input: Noisy_BS(dim) + TS(dim) + Hint(dim)
        self.input_proj = nn.Linear(dim * 3, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, t, condition, guide_hint):
        t_emb = self.time_mlp(t)
        # [Explicit Guide Injection]
        h = self.input_proj(torch.cat([x, condition, guide_hint], dim=1))
        for block in self.blocks:
            h = h + t_emb
            h = block(h)
        return self.output_proj(h)

# -----------------------------------------------------------------------------
# 3. Sampling Logic (For Validation)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_hybrid(model, compass, scheduler, ts_input, device):
    model.eval()
    compass.eval()
    B, D = ts_input.shape
    
    # Validation에서는 'Standard Mode' (Pure Compass)로 성능 체크
    # (실전 Inference 때는 Mixing을 쓸 수 있지만, 학습 중에는 기본기 확인)
    guide_hint = compass(ts_input)
    
    img = torch.randn(B, D, device=device)
    for i in reversed(range(0, scheduler.timestamps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        pred_noise = model(img, t, condition=ts_input, guide_hint=guide_hint)
        # guide = compass(ts_input)
        # pred_c = model(img, t, condition=ts_input, guide_hint=guide)
        # pred_u = model(img, t, condition=ts_input, guide_hint=torch.zeros_like(guide))
        # cfg = 1.5  # 1.0~2.5 sweep
        # pred_noise = pred_u + cfg * (pred_c - pred_u)
        
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
# 4. Utils & Data
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pt_file(path):
    print(f"📂 Loading {path}...")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        x, y = obj.get("x") or obj.get("X"), obj.get("y") or obj.get("Y")
    else:
        x, y = obj[0], obj[1]
    return x.float(), y.float()

class SimpleDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# -----------------------------------------------------------------------------
# 5. Main Training Loop (Robust Version)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--compass_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs_diffusion_v3_robust")
    parser.add_argument("--epochs", type=int, default=100) # Full Training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    xtr, ytr = load_pt_file(args.train_pt)
    xte, yte = load_pt_file(args.test_pt)
    dl = DataLoader(SimpleDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    
    val_idx = torch.randperm(len(xte))[:2048]
    x_val = xte[val_idx].to(device)
    y_val = yte[val_idx].to(device)

    dim = xtr.shape[1]

    # Load Compass (Fixed)
    print(f"🧭 Loading Compass from {args.compass_ckpt}...")
    compass = CompassMLP(dim).to(device)
    ckpt = torch.load(args.compass_ckpt, map_location="cpu")
    compass.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    compass.eval()
    for p in compass.parameters(): p.requires_grad = False

    # Initialize Model
    model = GuidedDirectDiffusionMLP(dim=dim, depth=6).to(device)
    scheduler = DiffusionSchedule().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"🚀 Training Robust Diffusion (Dynamic Hint Mixing)...")
    print("   [Policy]: 15% Zero (FID) | 15% TS (COS) | 70% Compass (Main)")
    
    best_score = -1.0 # Combined Score로 체크해도 됨
    
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            
            # [Step 1] Generate Base Hint
            with torch.no_grad():
                hint = compass(x)
                
            # [Step 2] Dynamic Hint Mixing (Robustness Training)
            rand_val = random.random()
            
            # if rand_val < 0.15:
            if rand_val < 0.30:
                # Type A: No Hint (Learn to generate independently) -> Improves FID
                hint = torch.zeros_like(hint)
            
            # elif rand_val < 0.30:
            elif rand_val < 0.50:
                # Type B: Perfect Structural Hint (Inject Raw TS) -> Improves Cosine
                # Teaches model: "Trust the hint channel for structure!"
                hint = x.clone()
                
            else:
                # Type C: Standard Compass Hint (70%)
                pass 
            # p_zero = 0.08  # 0.05~0.10
            # rand_val = random.random()

            # if rand_val < p_zero:
            #     hint = torch.zeros_like(hint)
            # else:
            #     # 기본은 compass hint
            #     pass
            
            # [Step 3] Standard Diffusion Training
            t = torch.randint(0, 1000, (x.size(0),), device=device).long()
            y_noisy, noise = scheduler.add_noise(y, t) # Target is BS
            
            # Predict
            noise_pred = model(y_noisy, t, condition=x, guide_hint=hint)
            loss = F.mse_loss(noise_pred, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        # Validation
        if ep % 5 == 0 or ep == args.epochs:
            gen_bs = sample_hybrid(model, compass, scheduler, x_val, device)
            
            cos = F.cosine_similarity(gen_bs, y_val).mean().item()
            mse = F.mse_loss(gen_bs, y_val).item()
            
            print(f"Ep {ep:03d} | Loss {loss_sum/len(dl):.4f} | "
                  f"Val Cos {cos:.4f} | Val MSE {mse:.4f}")
            
            # Cosine 우선 저장 (의료 영상에서 구조가 중요하므로)
            if cos > best_score:
                best_score = cos
                torch.save(model.state_dict(), f"{args.out_dir}/best_robust_model.pt")
        else:
            print(f"Ep {ep:03d} | Loss {loss_sum/len(dl):.4f}")

if __name__ == "__main__":
    main()
    
'''

python 02_train_diffusion_guided_v3_robust.py   --train_pt ./data/ts2bs_raw_dataset_512_train.pt   --test_pt ./data/ts2bs_raw_dataset_512_test.pt   --compass_ckpt ./runs_compass_v2/best_compass.pt   --out_dir runs_diffusion_v3_robust   --batch_size 2048   --epochs 100;
python 05_evaluate_metrics_v3_hybrid.py --test_pt data/ts2bs_raw_dataset_512_test.pt --compass_ckpt ./runs_compass_v2/best_compass.pt --diffusion_only_ckpt ./runs_diffusion_without_guided/best_diffusion_standard.pt --diffusion_ckpt runs_diffusion_v3_robust/best_robust_model.pt

'''