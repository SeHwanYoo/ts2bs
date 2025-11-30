import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
# import wandb
import torchvision.models as models

# ==========================================
# [설정]
# ==========================================
DIR_IMG = "/mnt/j/brain/BS"
DIR_TOKEN = "/mnt/d/workspace/dataset/BS_tokens"
OUTPUT_DIR = "./checkpoints_token_decoder_all"
PROJECT_NAME = "BS-Token-Decoder"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

# ==========================================
# 1. Decoder Model (14x14 -> 256x256)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GroupNorm(32, dim), nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GroupNorm(32, dim)
        )
    def forward(self, x): return x + self.block(x)

class TokenDecoder(nn.Module):
    def __init__(self, in_channels=1536, out_channels=3):
        super().__init__()
        
        # 1. Projection: 1536 -> 512
        self.proj = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.first_norm = nn.GroupNorm(32, 512)
        
        # 2. Upsampling (14 -> 28 -> 56 -> 112 -> 224)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.ReLU(True), ResBlock(256))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(32, 128), nn.ReLU(True), ResBlock(128))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(32, 64), nn.ReLU(True), ResBlock(64))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(32, 32), nn.ReLU(True), ResBlock(32))
        
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [B, 1536, 14, 14]
        x = F.relu(self.first_norm(self.proj(x)))
        
        x = self.up1(x) # 28
        x = self.up2(x) # 56
        x = self.up3(x) # 112
        x = self.up4(x) # 224
        
        out = self.final(x) # 224
        
        # 224 -> 256 (Resize)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        return out

# ==========================================
# 2. VGG Loss (선명하게 만들기용)
# ==========================================
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        # x, y: -1~1 -> 0~1 -> Normalize
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        x = self.normalize(x)
        y = self.normalize(y)
        return F.mse_loss(self.vgg(x), self.vgg(y))

# ==========================================
# 3. Dataset
# ==========================================
class TokenImageDataset(Dataset):
    def __init__(self, img_dir, token_dir, transform=None):
        self.img_dir = img_dir
        self.token_dir = token_dir
        self.transform = transform
        
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True) + 
                                glob.glob(os.path.join(img_dir, "**/*.jpg"), recursive=True))
        
        # self.img_paths = sorted(glob.glob(os.path.join(img_dir, "TCGA-02-0003-01A-01-BS1.0156cf95-2119-4a30-9894-2c0315c954d1_401_190/*.png"), recursive=True) + 
        #                         glob.glob(os.path.join(img_dir, "TCGA-02-0003-01A-01-BS1.0156cf95-2119-4a30-9894-2c0315c954d1_401_190/*.jpg"), recursive=True))
        
        self.valid_pairs = []
        
        print("🔍 Pairing files...")
        for p in tqdm(self.img_paths):
            rel = os.path.relpath(p, img_dir)
            # token_path = os.path.join(token_dir, os.path.splitext(rel)[0] + ".pt")
            token_path = p.replace(img_dir, token_dir).replace('.png', '.pt').replace('.jpg', '.pt')
            if os.path.exists(token_path):
                self.valid_pairs.append((p, token_path))
        print(f"✅ Found {len(self.valid_pairs)} pairs.")

    def __len__(self): return len(self.valid_pairs)
    def __getitem__(self, idx):
        ip, tp = self.valid_pairs[idx]
        img = Image.open(ip).convert("RGB")
        if self.transform: img = self.transform(img)
        token = torch.load(tp, map_location='cpu') # [1536, 14, 14]
        return {"img": img, "token": token}

# ==========================================
# 4. Main
# ==========================================
def main():
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")
    accelerator.init_trackers(project_name=PROJECT_NAME)
    
    # Transform (Decoder Target: 256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = TokenImageDataset(DIR_IMG, DIR_TOKEN, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    model = TokenDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    crit_l1 = nn.L1Loss()
    crit_vgg = VGGLoss().to(accelerator.device)
    
    model, optimizer, dataloader, crit_vgg = accelerator.prepare(model, optimizer, dataloader, crit_vgg)
    
    print("🚀 Start Training...")
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            img = batch["img"]
            token = batch["token"]
            
            recon = model(token)
            
            loss_l1 = crit_l1(recon, img)
            loss_vgg = crit_vgg(recon, img)
            loss = loss_l1 + (0.1 * loss_vgg)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            if global_step % 50 == 0:
                accelerator.log({"loss": loss.item(), "l1": loss_l1.item(), "vgg": loss_vgg.item()}, step=global_step)
            global_step += 1
            
        # Visualization
        if (epoch+1) % 10 == 0:
            
            if accelerator.is_main_process:
                with torch.no_grad():
                    model.eval()
                    s_img = img[:4]
                    s_token = token[:4]
                    pred = model(s_token)
                    
                    def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)
                    
                    viz = []
                    for k in range(s_img.size(0)):
                        row = torch.cat([denorm(s_img[k]), denorm(pred[k])], dim=2)
                        # viz.append(wandb.Image(row.permute(1,2,0).cpu().numpy(), caption=f"GT vs Pred (Ep{epoch})"))
                        
                        # save images in output folder 
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        save_path = os.path.join(OUTPUT_DIR, f"epoch{epoch}_sample{k}.png")
                        grid_img = row.permute(1,2,0).cpu().numpy() * 255
                        Image.fromarray(grid_img.astype('uint8')).save(save_path)
                        
                        
                    accelerator.log({"Results": viz}, step=global_step)
                
                # 체크포인트 저장
                accelerator.save_state(os.path.join(OUTPUT_DIR, f"ckpt_{epoch}"))
                
                # 2. Inference용 가벼운 모델 파일 따로 저장 (.pth)
                # Unwrap을 해서 DDP 껍데기를 벗기고 저장해야 나중에 편함
                unwrapped_model = accelerator.unwrap_model(model)
                model_path = os.path.join(OUTPUT_DIR, f"decoder_model_epoch{epoch}.pth")
                
                torch.save(unwrapped_model.state_dict(), model_path)
                print(f"💾 Saved model weights to {model_path}")
                
                
    print("🏁 Training Complete.")
    # 마지막 에폭 모델 저장
    final_path = os.path.join(OUTPUT_DIR, "decoder_model_final.pth")
    torch.save(accelerator.unwrap_model(model).state_dict(), final_path)
    print(f"💾 Saved Final Model to {final_path}")

if __name__ == "__main__":
    main()