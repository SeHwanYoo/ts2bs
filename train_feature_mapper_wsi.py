import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
import json
import random
import wandb 
from collections import defaultdict
import numpy as np
import wandb 

wandb.login(key="46aac2559a9feff8fff0ccf5e1c65a911aa3bd50")

# ==========================================================
# [설정]
# ==========================================================
TS_ROOT = "/mnt/d/workspace/dataset/TS_tokens"
BS_ROOT = "/mnt/d/workspace/dataset/BS_tokens"

TS_JSON = "./ts_train.json"
BS_JSON = "./bs_train.json"
TS_TEST_JSON = "./ts_test.json"

OUT_DIR = "./ckpt_feature_mapper_wsi"
DECODER_PATH = "checkpoints_token_decoder/decoder_model_epoch49.pth" 

# 배치 사이즈 = 한 슬라이드에서 볼 패치 개수 (메모리 허용 내 최대치 추천)
BATCH_SIZE = 256  
LR = 1e-4
EPOCHS = 200
DEVICE = "cuda"
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# 시드 고정 (재현성)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==========================================================
# 1. Slide-Aware Dataset & Sampler (핵심 기술 🔥)
# ==========================================================
class WSIDataset(Dataset):
    def __init__(self, root_dir, json_path, max_per_slide=1000):
        self.root_dir = root_dir
        with open(json_path, "r") as f:
            self.rel_paths = json.load(f)
            
        # 환자(Slide) ID별로 인덱스 그룹화
        self.slide_indices = defaultdict(list)
        for idx, path in enumerate(self.rel_paths):
            # 경로 구조: SlideID/patch_x_y.pt
            slide_id = os.path.dirname(path)
            self.slide_indices[slide_id].append(idx)
            
        self.slide_ids = list(self.slide_indices.keys())
        print(f"✅ Loaded {len(self.rel_paths)} patches from {len(self.slide_ids)} slides.")
        
        # Subsampling
        if max_per_slide is not None:
            new_rel_paths = []
            new_slide_indices = defaultdict(list)
            for slide_id, indices in self.slide_indices.items():
                if len(indices) > max_per_slide:
                    indices = random.sample(indices, max_per_slide)
                for idx in indices:
                    new_slide_indices[slide_id].append(len(new_rel_paths))
                    new_rel_paths.append(self.rel_paths[idx])
            self.rel_paths = new_rel_paths
            self.slide_indices = new_slide_indices
            print(f"🔪 Subsampled to {len(self.rel_paths)} patches after limiting {max_per_slide} per slide.")
            wandb.log({"subsampled_patches": len(self.rel_paths)})

    def __len__(self): return len(self.rel_paths)
    
    def __getitem__(self, idx):
        rel_path = self.rel_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        try:
            # [1536, 14, 14] -> [196, 1536]
            feat = torch.load(full_path, map_location="cpu")
            return feat.flatten(1).transpose(0, 1).float()
        except:
            return torch.zeros(196, 1536).float() # 에러 처리

class SlideBatchSampler(Sampler):
    """ 
    랜덤하게 섞지 않고, '같은 슬라이드'에 있는 패치들을 묶어서 배치로 내보냄.
    이게 있어야 WSI-level Loss 계산 가능!
    """
    def __init__(self, slide_indices, batch_size):
        self.slide_indices = slide_indices
        self.batch_size = batch_size
        self.batches = []
        
        for slide_id, indices in slide_indices.items():
            # 슬라이드 내에서는 순서 섞기 (Patch Random)
            random.shuffle(indices)
            
            # 배치 단위로 자르기
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]
                # 너무 작은 자투리 배치는 학습 불안정하므로 스킵 (선택사항)
                if len(batch) > batch_size // 2: 
                    self.batches.append(batch)
        
        # 슬라이드 순서는 섞음 (Slide Random)
        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

# ==========================================================
# 2. Models
# ==========================================================
class FeatureMapper(nn.Module):
    """ TS -> BS 변환기 (Linear Projection + Residual) """
    def __init__(self, dim=1536):
        super().__init__()
        # 단순할수록 WSI 선형성을 잘 보존함
        self.net = nn.Linear(dim, dim) 
    def forward(self, x):
        return x + self.net(x)

class Discriminator(nn.Module):
    """ Patch 또는 Slide 벡터가 Real인지 Fake인지 판별 """
    def __init__(self, dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

# 디코더 (시각화용)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.GroupNorm(32, dim), nn.ReLU(True), nn.Conv2d(dim, dim, 3, 1, 1), nn.GroupNorm(32, dim))
    def forward(self, x): return x + self.block(x)

class TokenDecoder(nn.Module):
    def __init__(self, in_channels=1536, out_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 512, 1); self.first_norm = nn.GroupNorm(32, 512)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.ReLU(True), ResBlock(256))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(32, 128), nn.ReLU(True), ResBlock(128))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(32, 64), nn.ReLU(True), ResBlock(64))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(32, 32), nn.ReLU(True), ResBlock(32))
        self.final = nn.Sequential(nn.Conv2d(32, out_channels, 3, 1, 1), nn.Tanh())
    def forward(self, x):
        x = F.relu(self.first_norm(self.proj(x)))
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        out = self.final(x)
        return F.interpolate(out, size=(256, 256), mode='bilinear')

# ==========================================================
# 3. Main Loop
# ==========================================================
def main():
    wandb.init(project="WSI-Level-Feature-Mapper")
    
    # 1. 데이터셋 & 샘플러
    print("📦 Loading Datasets...")
    ds_ts = WSIDataset(TS_ROOT, TS_JSON, max_per_slide=500)
    ds_bs = WSIDataset(BS_ROOT, BS_JSON, max_per_slide=500)
    ds_test = WSIDataset(TS_ROOT, TS_TEST_JSON, max_per_slide=500) # Test용

    sampler_ts = SlideBatchSampler(ds_ts.slide_indices, BATCH_SIZE)
    sampler_bs = SlideBatchSampler(ds_bs.slide_indices, BATCH_SIZE) # Unpaired Sampling

    loader_ts = DataLoader(ds_ts, batch_sampler=sampler_ts, num_workers=4, pin_memory=True)
    loader_bs = DataLoader(ds_bs, batch_sampler=sampler_bs, num_workers=4, pin_memory=True)
    # Test는 그냥 랜덤 로더 (시각화용)
    loader_test = DataLoader(ds_test, batch_size=4, shuffle=True, num_workers=2)

    bs_iter = iter(loader_bs)

    # 2. 모델
    mapper = FeatureMapper().to(DEVICE)
    patch_disc = Discriminator().to(DEVICE)
    wsi_disc = Discriminator().to(DEVICE) # [NEW] Slide Discriminator

    # 시각화용 디코더
    decoder = None
    if os.path.exists(DECODER_PATH):
        decoder = TokenDecoder().to(DEVICE)
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
        decoder.eval()
        print("✅ Decoder Loaded.")

    opt_G = optim.Adam(mapper.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(list(patch_disc.parameters()) + list(wsi_disc.parameters()), lr=LR, betas=(0.5, 0.999))
    
    criterion_GAN = nn.MSELoss()

    print("🔥 Start WSI-level Training...")
    
    for epoch in range(1, EPOCHS+1):
        mapper.train()
        loop = tqdm(loader_ts, desc=f"Ep {epoch}", ncols=100)
        
        for ts_batch in loop:
            # ts_batch: [B, 196, 1536] (한 슬라이드의 패치들)
            ts_batch = ts_batch.to(DEVICE)
            
            # BS 배치 (다른 슬라이드의 패치들)
            try:
                bs_batch = next(bs_iter).to(DEVICE)
            except:
                bs_iter = iter(loader_bs)
                bs_batch = next(bs_iter).to(DEVICE)

            # --------------------
            # Train Discriminators
            # --------------------
            opt_D.zero_grad()
            
            fake_bs_batch = mapper(ts_batch)
            
            # [Patch Level]
            # 각 패치(196개 평균)가 BS스러운가?
            ts_patch_vec = ts_batch.mean(dim=1) # [B, 1536]
            bs_patch_vec = bs_batch.mean(dim=1)
            fake_patch_vec = fake_bs_batch.detach().mean(dim=1)
            
            loss_D_patch = (criterion_GAN(patch_disc(bs_patch_vec), torch.ones_like(patch_disc(bs_patch_vec))) +
                            criterion_GAN(patch_disc(fake_patch_vec), torch.zeros_like(patch_disc(fake_patch_vec)))) / 2

            # [WSI Level] 🔥 핵심
            # 배치 전체(Slide)의 평균 벡터가 BS스러운가?
            # [B, 196, 1536] -> [B, 1536] (Patch Mean) -> [1, 1536] (Slide Mean)
            real_wsi_vec = bs_patch_vec.mean(dim=0, keepdim=True) # Real BS Slide
            fake_wsi_vec = fake_patch_vec.mean(dim=0, keepdim=True) # Fake BS Slide
            
            loss_D_wsi = (criterion_GAN(wsi_disc(real_wsi_vec), torch.ones_like(wsi_disc(real_wsi_vec))) +
                          criterion_GAN(wsi_disc(fake_wsi_vec), torch.zeros_like(wsi_disc(fake_wsi_vec)))) / 2
            
            loss_D = loss_D_patch + loss_D_wsi
            loss_D.backward()
            opt_D.step()

            # --------------------
            # Train Mapper (G)
            # --------------------
            opt_G.zero_grad()
            
            # Regenerate fake (for grad)
            fake_bs_batch = mapper(ts_batch)
            fake_patch_vec = fake_bs_batch.mean(dim=1)
            fake_wsi_vec = fake_patch_vec.mean(dim=0, keepdim=True)
            
            # 1. GAN Loss (Patch & WSI)
            loss_G_patch = criterion_GAN(patch_disc(fake_patch_vec), torch.ones_like(patch_disc(fake_patch_vec)))
            loss_G_wsi = criterion_GAN(wsi_disc(fake_wsi_vec), torch.ones_like(wsi_disc(fake_wsi_vec)))
            
            # 2. Structural Consistency (Cosine with TS)
            # "Fake Slide의 방향성은 원본 TS Slide와 같아야 한다" (구조 보존)
            ts_wsi_vec = ts_patch_vec.mean(dim=0, keepdim=True)
            loss_struct = 1 - F.cosine_similarity(fake_wsi_vec, ts_wsi_vec).mean()
            
            # 3. Distribution Matching
            loss_stats = F.mse_loss(fake_patch_vec.mean(0), bs_patch_vec.mean(0)) + \
                         F.mse_loss(fake_patch_vec.std(0), bs_patch_vec.std(0))

            loss_G = loss_G_patch + loss_G_wsi + (10.0 * loss_struct) + (10.0 * loss_stats)
            
            loss_G.backward()
            opt_G.step()
            
            # --------------------
            # Metric (증명용)
            # --------------------
            # "Fake가 TS보다 BS에 얼마나 가까워졌나?" (L2 Distance)
            # (수치가 작아질수록 BS에 가까워진 것)
            with torch.no_grad():
                dist_to_ts = F.mse_loss(fake_wsi_vec, ts_wsi_vec).item()
                dist_to_bs = F.mse_loss(fake_wsi_vec, real_wsi_vec).item()
                
            wandb.log({
                "Loss/G": loss_G.item(), "Loss/Struct": loss_struct.item(),
                "Dist/To_TS": dist_to_ts, "Dist/To_BS": dist_to_bs
            })
            loop.set_postfix(G=loss_G.item(), TS_dist=f"{dist_to_ts:.4f}", BS_dist=f"{dist_to_bs:.4f}")

        # ====================
        # Save & Visualize
        # ====================
        if epoch % 10 == 0:
            torch.save(mapper.state_dict(), os.path.join(OUT_DIR, f"mapper_ep{epoch}.pth"))
            
            if decoder is not None:
                mapper.eval()
                with torch.no_grad():
                    # Test Set에서 4장 가져오기
                    ts_sample = next(iter(loader_test)).to(DEVICE) # [4, 196, 1536]
                    fake_sample = mapper(ts_sample)
                    
                    def to_img(feat):
                        B, N, C = feat.shape
                        H = int(N**0.5)
                        return decoder(feat.transpose(1, 2).reshape(B, C, H, H))

                    rec_ts = to_img(ts_sample)
                    fake_bs = to_img(fake_sample)
                    
                    grid = torch.cat([rec_ts, fake_bs], dim=2) # 위아래 말고 옆으로 붙임 (비교용)
                    img_path = os.path.join(OUT_DIR, f"vis_ep{epoch}.png")
                    save_image(make_grid(grid, nrow=1, normalize=True), img_path)
                    
                    wandb.log({"Val": wandb.Image(img_path, caption=f"Ep{epoch}: TS(Left) -> FakeBS(Right)")})

if __name__ == "__main__":
    
    wandb.init(project="WSI-Level-Feature-Mapper")
    
    main()