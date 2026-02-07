#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_diffusion.py [Modified for Fair Comparison]

[Baseline Model: Standard Conditional Diffusion]
Role: Generates BS directly from TS (No Linear Guide)
Comparison Note:
  - Architecture is synced with '02_train_diffusion_guided.py' (Depth=6, ResBlock structure).
  - Goal: To prove if 'Residual Learning' (Guided) is better than 'Direct Learning'.
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
# 1. Diffusion Core (Identical to Guided Version)
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
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

# -----------------------------------------------------------------------------
# 2. Model: Conditional ResMLP (Synced with Guided Architecture)
# -----------------------------------------------------------------------------
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

# [CRITICAL] ResBlock Structure Matched to Guided Version
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        # Guided Version Flow: Lin2(Drop(Act(Lin1(Act(Norm(x))))))
        h = self.lin2(self.drop(self.act(self.lin1(self.act(self.norm(x))))))
        return x + h

class ConditionalDiffusionMLP(nn.Module):
    # [CRITICAL] Default depth changed to 6 to match Guided Version
    def __init__(self, dim, hidden=1024, depth=6, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden // 4),
            nn.Linear(hidden // 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        
        # Input: Noisy_BS (dim) + Condition_TS (dim)
        self.input_proj = nn.Linear(dim * 2, hidden)
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden, dropout) for _ in range(depth)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t, condition):
        # x: Noisy BS (Direct Target)
        # condition: Clean TS
        t_emb = self.time_mlp(t)
        
        x_input = torch.cat([x, condition], dim=1)
        h = self.input_proj(x_input)
        
        for block in self.blocks:
            h = h + t_emb # Add time embedding
            h = block(h)
            
        return self.output_proj(h)

# -----------------------------------------------------------------------------
# 3. Sampling Logic (Standard DDPM)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_diffusion(model, scheduler, condition, device):
    model.eval()
    B, D = condition.shape
    # Start from pure noise
    img = torch.randn(B, D, device=device)
    
    for i in reversed(range(0, scheduler.timestamps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        pred_noise = model(img, t, condition)
        
        beta = scheduler.betas[i]
        alpha = scheduler.alphas[i]
        alpha_hat = scheduler.alphas_cumprod[i]
        
        noise_factor = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
        mean = (1 / torch.sqrt(alpha)) * (img - noise_factor * pred_noise)
        
        if i > 0:
            noise = torch.randn_like(img)
            sigma = torch.sqrt(beta)
            img = mean + sigma * noise
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
# 5. Main Training
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs_diffusion_standard")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Timestamps fixed to 1000 to match Guided
    parser.add_argument("--timestamps", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    xtr, ytr = load_pt_file(args.train_pt)
    xte, yte = load_pt_file(args.test_pt)
    
    dl = DataLoader(SimpleDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    
    # Validation Subset
    val_subset_idx = torch.randperm(len(xte))[:2048]
    x_val_sample = xte[val_subset_idx].to(device)
    y_val_sample = yte[val_subset_idx].to(device)

    # Model Setup
    # Depth=6 ensures fair comparison with Guided Model
    dim = xtr.shape[1]
    model = ConditionalDiffusionMLP(dim=dim, depth=6).to(device)
    scheduler = DiffusionSchedule(timestamps=args.timestamps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"🚀 Training Standard Diffusion (Direct TS->BS) | Depth=6 | T={args.timestamps}...")

    best_cos = -1.0
    
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        
        for x, y in dl:
            x, y = x.to(device), y.to(device) # x=TS, y=BS (Real Target)
            
            # 1. Sample t
            t = torch.randint(0, args.timestamps, (x.size(0),), device=device).long()
            
            # 2. Add Noise to BS (Direct Target)
            y_noisy, noise = scheduler.add_noise(y, t)
            
            # 3. Predict Noise (Conditioned on TS)
            noise_pred = model(y_noisy, t, condition=x)
            
            loss = F.mse_loss(noise_pred, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            
        # Eval
        if ep % 5 == 0 or ep == args.epochs:
            gen_bs = sample_diffusion(model, scheduler, x_val_sample, device)
            
            cos = F.cosine_similarity(gen_bs, y_val_sample).mean().item()
            
            # Variance Check
            var_gen = gen_bs.var(dim=0).mean().item()
            var_real = y_val_sample.var(dim=0).mean().item()
            var_ratio = var_gen / (var_real + 1e-8)
            
            print(f"Ep {ep:03d} | Loss {loss_sum/len(dl):.4f} | "
                  f"Gen Cos {cos:.4f} | VarRatio {var_ratio:.2f}")
            
            if cos > best_cos:
                best_cos = cos
                torch.save(model.state_dict(), f"{args.out_dir}/best_diffusion_standard.pt")
        else:
            print(f"Ep {ep:03d} | Loss {loss_sum/len(dl):.4f}")

if __name__ == "__main__":
    main()
    
'''

python 02_train_diffusion.py \
  --train_pt ./data/ts2bs_raw_dataset_512_train.pt \
  --test_pt ./data/ts2bs_raw_dataset_512_test.pt \
  --out_dir runs_diffusion_without_guided \
  --batch_size 2048 \
  --epochs 100 

'''