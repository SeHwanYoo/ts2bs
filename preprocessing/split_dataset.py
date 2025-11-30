import os
import glob
import json
import random
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# [설정] 스캔할 토큰 폴더 경로 (여기가 Root)
# ==========================================
TS_ROOT = "/mnt/d/workspace/dataset/TS_tokens"
BS_ROOT = "/mnt/d/workspace/dataset/BS_tokens"

TEST_RATIO = 0.2  # 20%는 Test셋
SEED = 42         # 랜덤 시드 고정

def get_patient_id(rel_path):
    """
    상대 경로에서 환자 ID 추출
    예: 'TCGA-02-0003-01A-TS1/patch_0.pt' -> 'TCGA-02-0003'
    """
    # 1. 상위 폴더명(Slide ID) 추출
    slide_id = os.path.dirname(rel_path) 
    # 2. Slide ID에서 Patient ID 추출
    parts = slide_id.split('-')
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return slide_id # 예외 처리

def scan_and_group(root_dir):
    print(f"🔍 Scanning {root_dir} ...")
    # .pt 파일 재귀 탐색
    files = glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True)
    
    patient_map = defaultdict(list)
    
    for fpath in tqdm(files):
        # [핵심] 절대 경로 -> 상대 경로 변환
        # 예: /mnt/d/.../Slide1/p.pt -> Slide1/p.pt
        rel_path = os.path.relpath(fpath, root_dir)
        
        # 환자 ID 별로 묶기
        pid = get_patient_id(rel_path)
        patient_map[pid].append(rel_path)
        
    print(f"   👉 Found {len(files)} tokens from {len(patient_map)} patients.")
    return patient_map

def split_and_save(data_map, prefix):
    # 1. 환자 리스트 섞기
    patients = sorted(list(data_map.keys()))
    random.seed(SEED)
    random.shuffle(patients)
    
    # 2. Train/Test 환자 나누기
    split_idx = int(len(patients) * (1 - TEST_RATIO))
    train_pids = patients[:split_idx]
    test_pids = patients[split_idx:]
    
    # 3. 파일 리스트로 펼치기 (Flatten)
    train_files = []
    for pid in train_pids:
        train_files.extend(data_map[pid])
        
    test_files = []
    for pid in test_pids:
        test_files.extend(data_map[pid])
        
    # 4. 저장
    print(f"\n💾 Saving {prefix} splits...")
    with open(f"{prefix}_train.json", "w") as f:
        json.dump(train_files, f, indent=2)
    print(f"   - Train: {len(train_files)} files ({len(train_pids)} patients)")
    
    with open(f"{prefix}_test.json", "w") as f:
        json.dump(test_files, f, indent=2)
    print(f"   - Test : {len(test_files)} files ({len(test_pids)} patients)")

def main():
    print("🚀 Starting Portable Dataset Split...")
    
    # TS 처리
    ts_data = scan_and_group(TS_ROOT)
    split_and_save(ts_data, "ts")
    
    # BS 처리
    bs_data = scan_and_group(BS_ROOT)
    split_and_save(bs_data, "bs")
    
    print("\n🎉 Done! JSON files contain RELATIVE PATHS only.")

if __name__ == "__main__":
    main()