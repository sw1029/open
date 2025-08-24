
import pandas as pd
import os
import glob

# 기본 경로 설정
base_dir = 'data'

# data 디렉토리 내의 모든 하위 디렉토리(영업장) 목록을 가져옴
store_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

print("Calculating correlations for each store...")

# 각 영업장별로 상관관계 계산
for store_name in store_dirs:
    store_path = os.path.join(base_dir, store_name)
    menu_csv_files = glob.glob(os.path.join(store_path, '*.csv'))
    
    # 메뉴가 2개 미만이면 상관관계를 계산할 수 없음
    if len(menu_csv_files) < 2:
        print(f"Skipping '{store_name}' as it has fewer than 2 menu items.")
        continue

    # 모든 메뉴 데이터를 병합할 DataFrame 초기화
    merged_df = pd.DataFrame()

    # 각 메뉴 CSV 파일을 읽어와 병합
    for menu_file in menu_csv_files:
        menu_name = os.path.splitext(os.path.basename(menu_file))[0]
        try:
            menu_df = pd.read_csv(menu_file)
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file {menu_file}")
            continue

        # 날짜를 인덱스로 설정하고 매출수량 컬럼의 이름을 메뉴명으로 변경
        menu_df = menu_df.set_index('영업일자')
        menu_df = menu_df.rename(columns={'매출수량': menu_name})
        
        if merged_df.empty:
            merged_df = menu_df
        else:
            # outer join을 사용하여 모든 날짜를 포함
            merged_df = merged_df.join(menu_df, how='outer')

    # 병합 후 없는 값(NaN)은 0으로 채움 (매출이 없음을 의미)
    merged_df = merged_df.fillna(0)

    # 메뉴 간의 매출수량 상관계수 계산
    correlation_matrix = merged_df.corr()

    # 결과를 CSV 파일로 저장
    output_path = os.path.join(base_dir, f"{store_name}.csv")
    correlation_matrix.to_csv(output_path, encoding='utf-8-sig')
    print(f"Successfully saved correlation matrix for '{store_name}' to {output_path}")

print("Correlation analysis complete.")
