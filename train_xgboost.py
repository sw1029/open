import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import glob
import os

print("XGBoost-based demand forecasting script with correlation features started.")

# --- SMAPE 평가 지표 함수 정의 ---
def smape(y_true, y_pred, eps: float = 1e-8):
    """SMAPE metric with a small epsilon in the denominator to avoid ``0/0``."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(numerator / denominator) * 100

# --- 1. 데이터 로딩 ---
print("Step 1: Loading data...")
train_df = pd.read_csv('train/train.csv')
test_files = glob.glob('test/*.csv')
test_df_list = []
for file in test_files:
    temp_df = pd.read_csv(file)
    test_id = os.path.splitext(os.path.basename(file))[0]
    temp_df['test_id'] = test_id
    temp_df['영업일자'] = pd.to_datetime(temp_df['영업일자'])
    temp_df['submission_date'] = [f"{test_id}+{i+1}일" for i in range(len(temp_df))]
    test_df_list.append(temp_df)
test_df = pd.concat(test_df_list, ignore_index=True)
sample_submission_df = pd.read_csv('sample_submission.csv')

# --- 2. 피처 엔지니어링 ---
print("Step 2: Feature Engineering...")

# 전체 데이터프레임 생성 (나중에 lag 피처 생성에 사용)
combined_df = pd.concat([train_df, test_df], ignore_index=True)

def create_features(df):
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['year'] = df['영업일자'].dt.year
    df['month'] = df['영업일자'].dt.month
    df['day'] = df['영업일자'].dt.day
    df['dayofweek'] = df['영업일자'].dt.dayofweek
    df['weekofyear'] = df['영업일자'].dt.isocalendar().week.astype(int)
    return df

combined_df = create_features(combined_df)

for col in ['영업장명', '메뉴명']:
    le = LabelEncoder()
    le.fit(combined_df[col].unique())
    combined_df[col+'_encoded'] = le.transform(combined_df[col])

combined_df = combined_df.sort_values(by=['영업장명_메뉴명', '영업일자'])

print("Creating lag and rolling features...")
lags = [1, 7, 14, 28] # 1일전 lag 추가
for lag in lags:
    combined_df[f'lag_{lag}'] = combined_df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
combined_df['rolling_mean_7'] = combined_df.groupby('영업장명_메뉴명')['매출수량'].transform(lambda x: x.shift(1).rolling(7).mean())

# --- 2.1. 상관관계 피처 추가 ---
print("Creating correlation features...")
corr_files = glob.glob('data/*.csv')
corr_matrices = {os.path.basename(f).replace('.csv', ''): pd.read_csv(f, index_col=0) for f in corr_files}

best_buddy_map = {}
for store, corr_matrix in corr_matrices.items():
    for menu in corr_matrix.columns:
        # 자기 자신을 제외하고 가장 상관관계가 높은 메뉴 찾기
        best_buddy = corr_matrix[menu].drop(menu).idxmax()
        best_buddy_map[(store, menu)] = best_buddy

# 각 메뉴의 "최고의 짝꿍" 메뉴명을 매핑
combined_df['best_buddy'] = combined_df.set_index(['영업장명', '메뉴명']).index.map(best_buddy_map.get)

# 1일 전 판매량 데이터를 담은 임시 데이터프레임 생성
lag1_sales_df = combined_df[['영업일자', '영업장명', '메뉴명', 'lag_1']].copy()
lag1_sales_df.rename(columns={'lag_1': 'buddy_lag_1_sales'}, inplace=True)

# 원본 데이터에 "최고의 짝꿍"의 1일 전 판매량을 병합
combined_df = pd.merge(
    combined_df, 
    lag1_sales_df, 
    left_on=['영업일자', '영업장명', 'best_buddy'], 
    right_on=['영업일자', '영업장명', '메뉴명'], 
    how='left',
    suffixes=('', '_buddy')
)

combined_df.drop(columns=['메뉴명_buddy', 'best_buddy'], inplace=True)


# --- 3. 모델 훈련 및 검증 ---
print("Step 3: Training and validating XGBoost model...")

# 학습 데이터와 테스트 데이터 다시 분리 (기존 코드와 동일)
train_df = combined_df[combined_df['매출수량'].notna()]
test_df = combined_df[combined_df['매출수량'].isna()].copy() # .copy() 추가

features = [
    'year', 'month', 'day', 'dayofweek', 'weekofyear',
    '영업장명_encoded', '메뉴명_encoded',
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7',
    'buddy_lag_1_sales'
]
target = '매출수량'

# train_df 내에서 검증 데이터 분리 (기존 코드와 동일)
cutoff_date = train_df['영업일자'].max() - pd.Timedelta(days=7)
train_split_df = train_df[train_df['영업일자'] <= cutoff_date]
val_split_df = train_df[train_df['영업일자'] > cutoff_date]

X_train, y_train = train_split_df[features], train_split_df[target]
X_val, y_val = val_split_df[features], val_split_df[target]
X_test = test_df[features]

# 데이터 분리 후, 각 피처 셋에 대해 결측치 처리
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print(f"Training data size: {len(X_train)}, Validation data size: {len(X_val)}, Test data size: {len(X_test)}")

print(f"Training data size: {len(X_train)}, Validation data size: {len(X_val)}")

# y_train에 로그 변환 적용
y_train_log = np.log1p(np.maximum(0, y_train))
y_val_log = np.log1p(np.maximum(0, y_val))

xgb_reg = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

xgb_reg.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=100) # y_train_log, y_val_log 사용

val_preds_log = xgb_reg.predict(X_val)
val_preds = np.expm1(val_preds_log) # expm1로 역변환
val_preds[val_preds < 0] = 0
smape_score = smape(y_val, val_preds) # 원본 y_val과 비교
print(f"\nValidation SMAPE Score with Corr-Feature: {smape_score:.4f}\n")

# --- 4. 전체 데이터로 재학습 및 최종 예측 ---
print("Step 4: Retraining on full data and creating submission file...")
X_train_full, y_train_full = train_df[features], train_df[target]

# 최종 예측 시에도 동일하게 적용
y_train_full_log = np.log1p(np.maximum(0, y_train_full)) # 로그 변환 및 음수 값 클리핑

final_params = xgb_reg.get_params()
final_params['n_estimators'] = xgb_reg.best_iteration
if 'early_stopping_rounds' in final_params:
    del final_params['early_stopping_rounds']

xgb_reg_final = xgb.XGBRegressor(**final_params)
xgb_reg_final.fit(X_train_full, y_train_full_log, verbose=False) # y_train_full_log 사용

X_test = test_df[features]
predictions_log = xgb_reg_final.predict(X_test)
predictions = np.expm1(predictions_log) # 역변환
predictions[predictions < 0] = 0
test_df.loc[:, '매출수량'] = predictions.astype(int)


submission_df = (
    test_df.pivot_table(index='submission_date', columns='영업장명_메뉴명', values='매출수량')
    .reset_index()
)
submission_df = sample_submission_df[['영업일자']].merge(
    submission_df, left_on='영업일자', right_on='submission_date', how='left'
)
submission_df.drop(columns=['submission_date'], inplace=True)
submission_df = submission_df.reindex(columns=sample_submission_df.columns)
submission_df.fillna(0, inplace=True)

output_path = 'xgboost_submission_with_corr_feature.csv'
submission_df.to_csv(output_path, index=False)

print(f"Submission file created successfully at: {output_path}")
