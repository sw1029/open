
import pandas as pd
import os

# 데이터 경로 설정
input_csv_path = 'train/train.csv'
output_base_dir = 'data'

# CSV 파일 읽기
print(f"Reading data from {input_csv_path}...")
df = pd.read_csv(input_csv_path)

# '영업장명_메뉴명' 컬럼이 없는 경우를 대비한 예외 처리
if '영업장명_메뉴명' not in df.columns:
    raise ValueError(" '영업장명_메뉴명' column not found in the CSV file.")

# '영업장명'과 '메뉴명' 분리
print("Splitting '영업장명_메뉴명' column...")
df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

# 영업장명으로 그룹화
grouped_by_store = df.groupby('영업장명')

print("Processing and saving data...")
# 각 영업장별로 처리
for store_name, store_group in grouped_by_store:
    # 영업장명으로 디렉토리 경로 생성
    store_dir = os.path.join(output_base_dir, store_name)
    os.makedirs(store_dir, exist_ok=True)
    
    # 해당 영업장의 메뉴별로 그룹화
    grouped_by_menu = store_group.groupby('메뉴명')
    
    # 각 메뉴별로 CSV 파일 저장
    for menu_name, menu_group in grouped_by_menu:
        # 파일명으로 사용하기 부적절한 문자 제거
        safe_menu_name = "".join([c for c in menu_name if c.isalnum() or c in (' ', '-')]).rstrip()
        output_csv_path = os.path.join(store_dir, f"{safe_menu_name}.csv")
        
        # 필요한 컬럼(영업일자, 매출수량)만 선택하여 저장
        menu_group[['영업일자', '매출수량']].to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print("Data processing complete.")
