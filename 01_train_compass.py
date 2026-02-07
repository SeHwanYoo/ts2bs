#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_train_compass.py

[Step 1 of Linear-Guided Diffusion]
Role: The Compass (Linear/Shallow Mapper)
Goal: Learn the deterministic 'Mean' direction from TS to BS.
Strategy:
  - Strict Case-Centering: Learn Delta (TS - TS_mean) -> Delta (BS - BS_mean)
  - Objective: Maximize Correlation (Cosine) and create a valid 'Gap' vs Shuffle.
"""

import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Any

# -----------------------------------------------------------------------------
# 1. Utils & Data Loading (Same robust loader)
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def load_pt_file(path: str):
    """
    Load .pt file and robustly extract X, Y, CaseIDs.
    Returns: X, Y, CaseIDs(LongTensor or None)
    """
    print(f"📂 Loading {path}...")
    obj = torch.load(path, map_location="cpu")
    
    x, y, case_ids = None, None, None
    
    # Unpacking logic
    if isinstance(obj, dict):
        x = obj.get("x") or obj.get("X")
        y = obj.get("y") or obj.get("Y")
        # Try various keys for case_id
        for k in ["case_ids", "case_id", "case", "slide_id"]:
            if k in obj:
                case_ids = obj[k]
                break
    elif isinstance(obj, (tuple, list)):
        x = obj[0]
        y = obj[1]
        if len(obj) >= 3:
            case_ids = obj[2]
            
    if x is None or y is None:
        raise ValueError(f"❌ Could not find X, Y in {path}")

    # To Tensor & Float
    if not torch.is_tensor(x): x = torch.tensor(x)
    if not torch.is_tensor(y): y = torch.tensor(y)
    x, y = x.float(), y.float()

    # Process Case IDs
    if case_ids is not None:
        # If it's a list of strings, encode them
        if isinstance(case_ids, list) and len(case_ids) > 0 and isinstance(case_ids[0], str):
            uniq = sorted(set(case_ids))
            mapping = {cid: i for i, cid in enumerate(uniq)}
            case_ids = torch.tensor([mapping[c] for c in case_ids], dtype=torch.long)
        elif not torch.is_tensor(case_ids):
            try:
                case_ids = torch.tensor(case_ids, dtype=torch.long)
            except:
                case_ids = None # Give up if complex format
        else:
            case_ids = case_ids.long()
            
    return x, y, case_ids

class SimpleDataset(Dataset):
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i], (self.c[i] if self.c is not None else -1)

# -----------------------------------------------------------------------------
# 2. Case Centering Logic (The Core of "Compass")
# -----------------------------------------------------------------------------
class CaseNormalizer:
    def __init__(self, x_train, y_train, case_train):
        """
        Computes Mean(X) and Mean(Y) per case using TRAIN data only.
        If case info is missing, falls back to Global Mean.
        """
        self.x_stats = {}
        self.y_stats = {}
        self.x_global = x_train.mean(dim=0)
        self.y_global = y_train.mean(dim=0)
        self.use_case = (case_train is not None)

        if self.use_case:
            uniqs = torch.unique(case_train)
            print(f"🔧 Computing stats for {len(uniqs)} cases...")
            for u in uniqs:
                u = u.item()
                mask = (case_train == u)
                self.x_stats[u] = x_train[mask].mean(dim=0)
                self.y_stats[u] = y_train[mask].mean(dim=0)

    def get_mean(self, case_ids, is_y=False):
        """Returns the mean vector for the given batch of case_ids"""
        B = case_ids.shape[0]
        stats = self.y_stats if is_y else self.x_stats
        g_mean = self.y_global if is_y else self.x_global
        device = case_ids.device
        
        if not self.use_case:
            return g_mean.to(device).unsqueeze(0).expand(B, -1)
        
        means = []
        c_cpu = case_ids.detach().cpu().tolist()
        for c in c_cpu:
            if c in stats:
                means.append(stats[c])
            else:
                means.append(g_mean) # Unseen case -> Global mean
        
        return torch.stack(means).to(device)

    def normalize(self, x, y, case_ids):
        """ Returns (x - mu_x), (y - mu_y) """
        mu_x = self.get_mean(case_ids, is_y=False)
        mu_y = self.get_mean(case_ids, is_y=True) if y is not None else None
        
        x_centered = x - mu_x
        y_centered = (y - mu_y) if y is not None else None
        return x_centered, y_centered, mu_x, mu_y

# -----------------------------------------------------------------------------
# 3. Model: Residual MLP (Shallow & Robust)
# -----------------------------------------------------------------------------
class CompassMLP(nn.Module):
    def __init__(self, dim, hidden=1024, depth=2, dropout=0.2):
        super().__init__()
        layers = []
        in_d = dim
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(in_d, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = hidden
        layers.append(nn.Linear(in_d, dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize close to identity/zero for stability
        with torch.no_grad():
            self.net[-1].weight.data *= 0.1
            
    def forward(self, x):
        # Residual connection is key for preserving features
        return x + self.net(x)

# -----------------------------------------------------------------------------
# 4. Evaluation Engine (Gap Watcher)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_compass(model, loader, normalizer, device):
    model.eval()
    
    # Storage
    Y_gt_all, Y_pred_all, Y_shuf_pred_all = [], [], []
    
    for x, y, c in loader:
        x, y, c = x.to(device), y.to(device), c.to(device)
        
        # 1. Normalize (Center)
        x_c, y_c, mu_x, mu_y = normalizer.normalize(x, y, c)
        
        # 2. Forward Normal
        pred_c = model(x_c)
        pred = pred_c + mu_y # Restore global position
        
        # 3. Forward Shuffle (Validation of "Learning")
        # Shuffle x within batch, but keep case_ids/mu_y logic for "what if" scenario
        perm = torch.randperm(x.size(0))
        x_shuf = x[perm]
        x_shuf_c, _, _, _ = normalizer.normalize(x_shuf, None, c[perm]) # Center using shuffled source
        pred_shuf_c = model(x_shuf_c)
        pred_shuf = pred_shuf_c + mu_y # Add back ORIGINAL target mean (Target hasn't moved)
        
        Y_gt_all.append(y.cpu())
        Y_pred_all.append(pred.cpu())
        Y_shuf_pred_all.append(pred_shuf.cpu())

    Y = torch.cat(Y_gt_all)
    P = torch.cat(Y_pred_all)
    S = torch.cat(Y_shuf_pred_all)
    
    # Metrics
    cos_norm = F.cosine_similarity(P, Y, dim=1).mean().item()
    cos_shuf = F.cosine_similarity(S, Y, dim=1).mean().item()
    
    mse = F.mse_loss(P, Y).item()
    
    # Variance Ratio (Are we collapsing?)
    var_p = P.var(dim=0).mean().item()
    var_y = Y.var(dim=0).mean().item()
    var_ratio = var_p / (var_y + 1e-8)
    
    return {
        "cos": cos_norm,
        "cos_shuf": cos_shuf,
        "gap": cos_norm - cos_shuf,
        "mse": mse,
        "var_ratio": var_ratio
    }

# -----------------------------------------------------------------------------
# 5. Main Training Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs_compass")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4) # Slightly lower LR for stability
    parser.add_argument("--w_cos", type=float, default=2.0) # Emphasize direction
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    xtr, ytr, ctr = load_pt_file(args.train_pt)
    xte, yte, cte = load_pt_file(args.test_pt)
    
    print(f"📊 Train: {xtr.shape}, Test: {xte.shape}")
    
    # Init Normalizer (Train stats only)
    normalizer = CaseNormalizer(xtr, ytr, ctr)
    
    # Dataloaders
    dl_train = DataLoader(SimpleDataset(xtr, ytr, ctr), batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(SimpleDataset(xte, yte, cte), batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = CompassMLP(dim=xtr.shape[1], depth=3, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_gap = -1.0
    
    print("🚀 Starting Compass Training...")
    print(f"   Goal: Maximize GAP (Normal - Shuffle) and keep VarRatio > 0.3")
    
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = 0
        
        for x, y, c in dl_train:
            x, y, c = x.to(device), y.to(device), c.to(device)
            
            # Learn the Delta: (x - mu_x) -> (y - mu_y)
            x_c, y_c, _, _ = normalizer.normalize(x, y, c)
            
            pred_c = model(x_c)
            
            # Loss: Cosine + MSE on the centered data
            cos_loss = 1.0 - F.cosine_similarity(pred_c, y_c, dim=1).mean()
            mse_loss = F.mse_loss(pred_c, y_c)
            
            # Norm penalty (Prevent shrinking to 0)
            norm_loss = F.mse_loss(pred_c.norm(dim=1), y_c.norm(dim=1))
            
            loss = (args.w_cos * cos_loss) + mse_loss + (0.1 * norm_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
        # Eval
        metrics = evaluate_compass(model, dl_test, normalizer, device)
        
        # Logging
        is_best = metrics['gap'] > best_gap
        if is_best:
            best_gap = metrics['gap']
            torch.save(model.state_dict(), f"{args.out_dir}/best_compass.pt")
            
        print(f"Ep {epoch:03d} | Loss {loss_sum/len(dl_train):.4f} | "
              f"GAP: {metrics['gap']:.4f} "
              f"(Cos {metrics['cos']:.4f} vs Shuf {metrics['cos_shuf']:.4f}) | "
              f"VarRat: {metrics['var_ratio']:.3f} | "
              f"{'🌟 BEST' if is_best else ''}")
        
        # Early Stop Condition based on USER GOAL
        if metrics['gap'] > 0.05 and metrics['var_ratio'] > 0.3:
            print(f"\n✅ PASS CRITERIA MET at Epoch {epoch}!")
            print("   -> The Compass is pointing correctly and preserving variance.")
            print("   -> You can proceed to Step 2 (Diffusion) now.")
            # break (Optionally uncomment to auto-stop)

    print(f"\n🏁 Finished. Best Gap: {best_gap:.4f}")
    with open(f"{args.out_dir}/final_metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
    
'''

python 01_train_compass.py \
  --train_pt ./data/ts2bs_raw_dataset_512_train.pt \
  --test_pt ./data/ts2bs_raw_dataset_512_test.pt \
  --out_dir runs_compass_test \
  --w_cos 3.0 \
  --batch_size 2048

'''