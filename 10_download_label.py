import pandas as pd
import os
from glob import glob

# ==========================================
# 1. 설정
# ==========================================
FEATURE_DIR = 'D:/workspace/dataset/brain/BS_CONCH' 
# 방금 PowerShell로 다운받은 그 파일
CLINICAL_FILE = 'D:/workspace/ts2bs_20251213/idh_data.txt' 
OUTPUT_CSV = 'D:/workspace/dataset/brain/final_idh_labels.csv'

# ==========================================
# 2. 파일 로드 (무조건 읽음)
# ==========================================
if not os.path.exists(CLINICAL_FILE):
    print(f"❌ '{CLINICAL_FILE}' 파일이 없습니다! 1단계 PowerShell 명령어를 실행해주세요.")
    exit()

print("📖 텍스트 파일 읽는 중...")

# cBioPortal 데이터는 '#'으로 시작하는 주석이 4줄 있음. 그거 건너뛰고 읽기.
try:
    df = pd.read_csv(CLINICAL_FILE, sep='\t', comment='#')
except:
    # 혹시 에러나면 헤더 무시하고 다시 시도
    df = pd.read_csv(CLINICAL_FILE, sep='\t', comment='#', header=0, on_bad_lines='skip')

# ==========================================
# 3. IDH 컬럼 & 환자 ID 찾기 (지능형 탐색)
# ==========================================
# 컬럼명 대문자로 통일
df.columns = [str(c).upper().strip() for c in df.columns]

# 1) 환자 ID 컬럼 찾기 (SAMPLE_ID or PATIENT_ID)
id_col = None
for col in df.columns:
    if "PATIENT" in col or "SAMPLE" in col:
        # 값이 TCGA로 시작하는지 확인
        if df[col].astype(str).str.contains("TCGA").any():
            id_col = col
            break
if not id_col:
    print("❌ 환자 ID 컬럼을 못 찾았습니다.")
    exit()

# 2) IDH 정보 컬럼 찾기
# Ceccarelli 2016 데이터셋의 핵심 컬럼명 후보들
candidates = ['IDH/CODELESS SUBTYPE', 'IDH1 MUTATION', 'IDH STATUS', 'IDH1 STATUS', 'SUBTYPE']
target_col = None

# 후보군에서 정확히 일치하는 것 찾기
for cand in candidates:
    if cand in df.columns:
        target_col = cand
        break

# 없으면 'IDH' 글자 들어간 거 아무거나 찾기
if not target_col:
    for col in df.columns:
        if "IDH" in col:
            target_col = col
            break

print(f"✅ 매칭 준비 완료!")
print(f" - ID 컬럼: {id_col}")
print(f" - IDH 컬럼: {target_col}")

# ==========================================
# 4. 매칭 및 저장
# ==========================================
# 검색 속도를 위해 딕셔너리로 변환
# 키: TCGA-02-0003 (앞 3자리만 사용)
df['short_id'] = df[id_col].astype(str).apply(lambda x: "-".join(x.split("-")[:3]))
ref_dict = df.set_index('short_id')[target_col].to_dict()

# 내 환자 리스트
pt_files = glob(os.path.join(FEATURE_DIR, "**/*.pt"), recursive=True)
results = []
match_cnt = 0
missing_cnt = 0

for f in pt_files:
    pid = "-".join(os.path.basename(f).split("-")[:3]) # TCGA-02-0003
    
    val = ref_dict.get(pid, "N/A")
    val_str = str(val).upper()
    
    label = -1
    
    # === 라벨링 규칙 (Ceccarelli 2016) ===
    # WT -> 0
    # Mutant / Codel / Non-codel -> 1
    
    if "WT" in val_str or "WILDTYPE" in val_str:
        label = 0
    elif "MUT" in val_str or "CODEL" in val_str:
        label = 1
    elif "NOS" in val_str or val_str == "N/A" or val_str == "NAN":
        label = -1
    else:
        # IDH 컬럼이 확실하다면, WT 아니면 다 Mutant로 간주
        if target_col and "IDH" in target_col:
            label = 1
        else:
            label = -1

    if label != -1:
        results.append({'patient_id': pid, 'label': label, 'raw_value': val})
        match_cnt += 1
    else:
        missing_cnt += 1

# 결과 저장
df_res = pd.DataFrame(results)
if not df_res.empty:
    # 중복 제거 (혹시 파일이 여러개라 중복됐을까봐)
    df_res = df_res.drop_duplicates(subset=['patient_id'])
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*30)
    print(f"🎉 성공했습니다 형님!")
    print(f" - 매칭 성공: {len(df_res)}명")
    print(f" - 매칭 실패: {93 - len(df_res)}명")
    print(f"💾 파일 저장됨: {OUTPUT_CSV}")
    print("="*30)
    print(df_res['label'].value_counts())
else:
    print("❌ 매칭된 데이터가 없습니다. txt 파일 내용을 확인해주세요.")