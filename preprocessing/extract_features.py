import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from huggingface_hub import login

# ==========================================
# [설정] 모델별 상세 스펙 (HuggingFace ID 등)
# ==========================================
MODEL_CONFIGS = {
    "uni": {
        "hf_id": "MahmoodLab/uni",
        "arch": "vit_large_patch16_224",
        "img_size": 224,
        "mean": (0.5, 0.5, 0.5), # UNI는 보통 0.5 normalize 권장인 경우가 많음 (확인 필요시 조정)
        "std": (0.5, 0.5, 0.5),
    },
    "virchow2": {
        "hf_id": "paige-ai/Virchow2",
        "arch": "hf_hub:paige-ai/Virchow2", # timm이 HF Hub 지원
        "img_size": 224, # Virchow는 224 or 512 (보통 224 패치 사용시)
        "mean": (0.485, 0.456, 0.406), # ImageNet Stat
        "std": (0.229, 0.224, 0.225),
    },
    "conch": {
        "hf_id": "MahmoodLab/CONCH", 
        "script_path": None, # CONCH는 별도 라이브러리 필요할 수 있음 (아래 설명 참조)
        "img_size": 224,
        "mean": (0.48145466, 0.4578275, 0.40821073), # CLIP standard
        "std": (0.26862954, 0.26130258, 0.27577711),
    }
}

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 재귀적으로 이미지 파일 검색
        print(f"🔍 Scanning files in {root_dir}...")
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.image_paths.append(os.path.join(root, file))
        print(f"   👉 Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), img_path # Dummy for error handling

def get_model_and_transform(model_name):
    print(f"🚀 Loading Model: {model_name.upper()}...")
    cfg = MODEL_CONFIGS[model_name]
    
    # 1. Transform 설정
    transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
    ])
    
    # 2. Model 로드
    model = None
    
    if model_name == "uni":
        # UNI: timm을 통해 로드 (hf_hub 지원)
        # 만약 로컬 가중치가 있다면 pretrained_cfg 등을 수정해야 함
        model = timm.create_model("hf_hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        
    elif model_name == "virchow2":
        # Virchow2: timm + HF Hub
        # 주의: timm 최신 버전 필요
        model = timm.create_model("hf_hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        
    elif model_name == "conch":
        # CONCH는 timm으로 바로 안 될 수 있음. open_clip이나 별도 로더 필요.
        # 여기서는 timm 호환이 된다고 가정하거나, 혹은 사용자 환경에 conch 라이브러리가 있다고 가정.
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            model, _ = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/CONCH")
        except ImportError:
            print("🚨 [Error] CONCH requires 'conch' library installed or specific loader.")
            print("   -> Fallback: Attempting generic ViT loading (Might fail for CONCH specifics)")
            # CONCH가 ViT-B/16 기반이므로 구조만 가져올 수도 있으나, 가중치 매핑이 다름.
            # *실제 사용시에는 MahmoodLab의 공식 conch repo 코드를 참조해야 함*
            raise NotImplementedError("CONCH loading requires custom library installation.")

    # 공통: Evaluation 모드 & Head 제거 (Feature Extraction 용)
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(0) # Remove classification head
    
    model.eval()
    model.cuda()
    return model, transform

def main():
    parser = argparse.ArgumentParser(description="Extract Features using Foundation Models")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing patch images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save .pt files")
    parser.add_argument("--model", type=str, default="uni", choices=["uni", "virchow2", "conch"], help="Model to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    # 출력 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 & 데이터셋 준비
    model, transform = get_model_and_transform(args.model)
    dataset = PatchDataset(args.input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    
    print("🎬 Start Extraction...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            images = images.cuda()
            
            # 모델별 Forward 방식 차이 처리
            if args.model == "conch":
                # CONCH는 visual encoder만 사용
                features = model.encode_image(images, proj_contrast=False, normalize=False)
            else:
                # UNI, Virchow2 (timm base)
                features = model(images)
            
            # 저장 (CPU로 이동)
            features = features.cpu()
            
            for i, path in enumerate(paths):
                # 원본 경로 구조 파싱
                rel_path = os.path.relpath(path, args.input_dir) # e.g., Case1/patch_01.png
                save_rel_path = os.path.splitext(rel_path)[0] + ".pt" # e.g., Case1/patch_01.pt
                save_path = os.path.join(args.output_dir, save_rel_path)
                
                # 하위 폴더 생성
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 저장
                torch.save(features[i].clone(), save_path)

    print("✅ Extraction Complete!")

if __name__ == "__main__":
    main()