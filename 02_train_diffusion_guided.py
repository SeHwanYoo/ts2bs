#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_diffusion_guided.py

[Step 2 of Linear-Guided Diffusion] -- CORRECTED VERSION
Role: Residual Generator
Goal: Learn P( (BS - Compass(TS)) | TS )
Strategy:
  1. Load Pre-trained Compass (Frozen).
  2. Calculate Residual = BS_ground_truth - Compass(TS).
  3. Train Diffusion to generate this Residual.
  4. Inference = Compass(TS) + Generated_Residual.
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
# 0. Define Compass Architecture (Must match Step 1)
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
        # Init mostly identity
        self.net[-1].weight.data *= 0.01
        self.net[-1].bias.data *= 0.01

    def forward(self, x):
        return x + self.net(x)

# -----------------------------------------------------------------------------
# 1. Diffusion Core
# -----------------------------------------------------------------------------
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

class ResidualDiffusionMLP(nn.Module):
    def __init__(self, dim, hidden=1024, depth=6, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden // 4),
            nn.Linear(hidden // 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Condition is TS input
        # self.input_proj = nn.Linear(dim * 2, hidden)
        # [mod]
        self.input_proj = nn.Linear(dim * 3, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))

    # def forward(self, x_noisy, t, condition):
    #     # x_noisy: Noisy Residual
    #     # condition: Clean TS (Original Input)
    #     t_emb = self.time_mlp(t)
    #     h = self.input_proj(torch.cat([x_noisy, condition], dim=1))
    #     for block in self.blocks:
    #         h = h + t_emb
    #         h = block(h)
    #     return self.output_proj(h)
    # [mod]
    def forward(self, x_noisy, t, condition, guide_pred): # guide_pred 추가
        t_emb = self.time_mlp(t)
        # NoisyResidual + TS + CompassPrediction 셋을 다 합침
        h = self.input_proj(torch.cat([x_noisy, condition, guide_pred], dim=1))
        
        for block in self.blocks:
            h = h + t_emb
            h = block(h)
        return self.output_proj(h)

# -----------------------------------------------------------------------------
# 2. Sampling (Inference)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_guided(diffusion, compass, scheduler, ts_input, device):
    diffusion.eval()
    compass.eval()
    B, D = ts_input.shape
    
    # 1. Get Linear Guide (The Mean)
    linear_guide = compass(ts_input)
    
    # 2. Sample Residual (The Texture)
    res = torch.randn(B, D, device=device) # Start from noise
    
    for i in reversed(range(0, scheduler.timestamps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        # pred_noise = diffusion(res, t, condition=ts_input)
        pred_noise = diffusion(res, t, condition=ts_input, guide_pred=linear_guide)
        
        alpha = scheduler.alphas[i]
        alpha_hat = scheduler.alphas_cumprod[i]
        beta = scheduler.betas[i]
        
        noise_factor = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
        mean = (1 / torch.sqrt(alpha)) * (res - noise_factor * pred_noise)
        
        if i > 0:
            noise = torch.randn_like(res)
            res = mean + torch.sqrt(beta) * noise
        else:
            res = mean
            
    # 3. Combine: Final = Guide + Residual
    return linear_guide + res

# -----------------------------------------------------------------------------
# 3. Utils & Data
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pt(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict): x, y = obj.get("x") or obj.get("X"), obj.get("y") or obj.get("Y")
    else: x, y = obj[0], obj[1]
    return x.float(), y.float()

class SimpleDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------
def main():
    
    '''
    
    python 02_train_diffusion_guided.py --train_pt ./data/ts2bs_raw_dataset_512_train.pt --test_pt ./data/ts2bs_raw_dataset_512_test.pt --out_dir test_fuck --compass_ckpt runs_compass_v2/best_compass.pt 
    
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--compass_ckpt", type=str, required=True, help="Path to best_compass.pt from Step 1")
    parser.add_argument("--out_dir", type=str, default="runs_diffusion_guided")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    xtr, ytr = load_pt(args.train_pt)
    xte, yte = load_pt(args.test_pt)
    dl = DataLoader(SimpleDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    
    # Validation subset
    idx = torch.randperm(len(xte))[:2048]
    x_val, y_val = xte[idx].to(device), yte[idx].to(device)

    # Load Compass (Step 1)
    print(f"🧭 Loading Compass from {args.compass_ckpt}...")
    dim = xtr.shape[1]
    compass = CompassMLP(dim).to(device)
    # Load state dict safely
    ckpt = torch.load(args.compass_ckpt, map_location="cpu")
    # If saved with 'model' key or direct state_dict
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    compass.load_state_dict(state_dict)
    compass.eval() # FREEZE
    for p in compass.parameters(): p.requires_grad = False

    # Init Diffusion (Step 2)
    diffusion = ResidualDiffusionMLP(dim).to(device)
    scheduler = DiffusionSchedule().to(device)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-4)

    print("🚀 Training Linear-Guided Diffusion (Residual Learning)...")
    
    best_cos = -1.0
    
    for ep in range(1, args.epochs + 1):
        diffusion.train()
        loss_sum = 0
        
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            
            # 1. Get Linear Guidance (No Grad)
            with torch.no_grad():
                y_guide = compass(x)
                
            # 2. Calculate Residual (Ground Truth for Diffusion)
            target_residual = y - y_guide
            
            # [DEBUG] 이 로그를 한번 찍어보세요
            # if ep == 1:
            #     print(f"Target BS Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            #     print(f"Residual Mean: {target_residual.mean():.4f}, Std: {target_residual.std():.4f}")
            #     exit() 
            
            
            # 3. Diffusion Training on Residual
            t = torch.randint(0, 1000, (x.size(0),), device=device).long()
            res_noisy, noise = scheduler.add_noise(target_residual, t)
            
            # Predict noise based on NoisyResidual + Condition(TS)
            # noise_pred = diffusion(res_noisy, t, condition=x)
            noise_pred = diffusion(res_noisy, t, condition=x, guide_pred=y_guide)
            
            loss = F.mse_loss(noise_pred, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        # Eval
        if ep % 5 == 0 or ep == args.epochs:
            # Generate: Compass(x) + Diffusion(x)
            gen_bs = sample_guided(diffusion, compass, scheduler, x_val, device)
            
            cos = F.cosine_similarity(gen_bs, y_val).mean().item()
            
            # Compare Variance: 
            # We want Gen Variance to be close to Real Variance, NOT close to Compass Variance
            var_gen = gen_bs.var(dim=0).mean().item()
            var_real = y_val.var(dim=0).mean().item()
            var_compass = compass(x_val).var(dim=0).mean().item()
            
            print(f"Ep {ep:03d} | Loss {loss_sum/len(dl):.4f} | "
                  f"Cos {cos:.4f} (Compass was ~0.72) | "
                  f"Var: Gen {var_gen:.2f} / Real {var_real:.2f} (Compass {var_compass:.2f})")
            
            if cos > best_cos:
                best_cos = cos
                torch.save(diffusion.state_dict(), f"{args.out_dir}/best_guided_diffusion.pt")
        else:
            print(f"Ep {ep:03d} | Loss {loss_sum/len(dl):.4f}")

if __name__ == "__main__":
    main()