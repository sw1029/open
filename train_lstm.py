import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("PyTorch LSTM-based demand forecasting script started.")

# --- 0. 설정 및 SMAPE 함수 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 하이퍼파라미터
SEQUENCE_LENGTH = 14
PREDICT_LENGTH = 7  # 한 번에 예측할 타임스텝 수
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # 에포크 수 증가
HIDDEN_SIZE = 128 # 모델 용량 증가
NUM_LAYERS = 2
PATIENCE = 10 # 조기 종료를 위한 patience

def smape(y_true, y_pred, eps: float = 1e-8):
    """Symmetric mean absolute percentage error.

    Adds a small ``eps`` to the denominator to avoid ``0/0`` situations that
    can produce ``nan`` values during evaluation.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(numerator / denominator) * 100


class SMAPELoss(nn.Module):
    """SMAPE loss that avoids division by zero.

    The denominator uses the sum of absolute values with a small epsilon
    to stabilize training when both prediction and target are zero.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + self.eps
        return (numerator / denominator).mean()

def get_future_date_str(date_str, days_to_add):
    """ 'TEST_00+1일' 형식의 문자열 날짜를 days_to_add 만큼 더한 문자열을 반환 """
    try:
        parts = date_str.replace('일','').split('+')
        test_id = parts[0]
        day_num = int(parts[1])
        return f"{test_id}+{day_num + days_to_add}일"
    except (IndexError, ValueError):
         # 표준 날짜 형식이 들어올 경우 처리 (예: '2023-01-01')
        return str(pd.to_datetime(date_str) + pd.Timedelta(days=days_to_add))

# --- 1. 데이터 로딩 및 피처 엔지니어링 ---
print("Step 1: Loading and feature engineering...")
train_df = pd.read_csv('train/train.csv')
test_df = pd.concat([pd.read_csv(f) for f in glob.glob('test/*.csv')], ignore_index=True)
sample_submission_df = pd.read_csv('sample_submission.csv')

def create_features_train(df):
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    # train 데이터의 '영업일자'는 datetime으로 변환
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['dayofweek'] = df['영업일자'].dt.dayofweek
    df['month'] = df['영업일자'].dt.month
    return df

def create_features_test(df):
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    # test 데이터의 '영업일자'는 문자열로 주어지므로 실제 날짜로 변환
    df['영업일자'] = pd.to_datetime(df['영업일자'], errors='coerce')
    # 변환된 날짜를 이용해 요일과 월을 계산하고, 파싱 실패 시 -1로 채움
    df['dayofweek'] = df['영업일자'].dt.dayofweek.fillna(-1).astype(int)
    df['month'] = df['영업일자'].dt.month.fillna(-1).astype(int)
    return df

def load_calendar_features(df, holiday_path='holidays.csv', event_path='events.csv'):
    """
    Load holiday, season and event information. If corresponding files are
    missing, default values are used.
    """
    df['영업일자'] = pd.to_datetime(df['영업일자'])

    if os.path.exists(holiday_path):
        holiday_df = pd.read_csv(holiday_path)
        holiday_dates = pd.to_datetime(holiday_df['date']).dt.strftime('%Y-%m-%d').tolist()
        df['is_holiday'] = df['영업일자'].dt.strftime('%Y-%m-%d').isin(holiday_dates).astype(int)
    else:
        df['is_holiday'] = 0

    if os.path.exists(event_path):
        event_df = pd.read_csv(event_path)
        event_dates = pd.to_datetime(event_df['date']).dt.strftime('%Y-%m-%d').tolist()
        df['is_event'] = df['영업일자'].dt.strftime('%Y-%m-%d').isin(event_dates).astype(int)
    else:
        df['is_event'] = 0

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    df['season'] = df['month'].apply(month_to_season)
    return df

# train_df와 test_df에 각각 다른 피처 엔지니어링 함수 적용
train_df = create_features_train(train_df)
test_df = create_features_test(test_df)
# 공휴일/계절/이벤트 정보 추가
train_df = load_calendar_features(train_df)
test_df = load_calendar_features(test_df)

# 이후 처리를 위해 test_df의 '영업일자'를 문자열로 명시적 변환
train_df['영업일자'] = train_df['영업일자'].astype(str)
test_df['영업일자'] = test_df['영업일자'].astype(str)


train_df['source'] = 'train'
test_df['source'] = 'test'
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.loc[combined_df['source'] == 'test', '매출수량'] = np.nan
print(f"DEBUG after create_features: NaNs count = {combined_df['매출수량'].isna().sum()}")

for col in ['영업장명', '메뉴명']:
    le = LabelEncoder()
    combined_df[col+'_encoded'] = le.fit_transform(combined_df[col])
print(f"DEBUG after label_encoding: NaNs count = {combined_df['매출수량'].isna().sum()}")

combined_df = combined_df.sort_values(by=['영업장명_메뉴명', '영업일자'])
print(f"DEBUG after sort_values: NaNs count = {combined_df['매출수량'].isna().sum()}")

lags = [1, 7, 14]
for lag in lags:
    combined_df[f'lag_{lag}'] = combined_df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
print(f"DEBUG after lag_creation: NaNs count = {combined_df['매출수량'].isna().sum()}")

corr_files = glob.glob('data/*.csv')
corr_matrices = {os.path.basename(f).replace('.csv', ''): pd.read_csv(f, index_col=0) for f in corr_files}
best_buddy_map = { (store, menu): corr_matrix[menu].drop(menu).idxmax() for store, corr_matrix in corr_matrices.items() for menu in corr_matrix.columns }
combined_df['best_buddy'] = combined_df.set_index(['영업장명', '메뉴명']).index.map(best_buddy_map.get)
print(f"DEBUG after best_buddy: NaNs count = {combined_df['매출수량'].isna().sum()}")

lag1_sales_df = combined_df[['영업일자', '영업장명', '메뉴명', 'lag_1']].rename(columns={'lag_1': 'buddy_lag_1_sales'})
combined_df = pd.merge(combined_df, lag1_sales_df, left_on=['영업일자', '영업장명', 'best_buddy'], right_on=['영업일자', '영업장명', '메뉴명'], how='left', suffixes=('', '_buddy'))
print(f"DEBUG after merge: NaNs count = {combined_df['매출수량'].isna().sum()}")

# 피처 생성 과정에서 생긴 NaN만 0으로 채움 (예측 대상인 매출수량의 NaN은 유지)
cols_to_fill = [f'lag_{lag}' for lag in lags] + ['buddy_lag_1_sales']
for col in cols_to_fill:
    # 경고를 해결하고 더 안전한 방식으로 NaN 값을 채움
    combined_df[col] = combined_df[col].fillna(0)

# 수정이 잘 적용되었는지 확인하기 위한 디버그 코드
print(f"DEBUG after targeted fillna: NaNs count = {combined_df['매출수량'].isna().sum()}")

# --- 2. 데이터 전처리 ---
print("Step 2: Scaling and creating sequences...")
features_to_scale = ['dayofweek', 'month', '영업장명_encoded', '메뉴명_encoded', 'lag_1', 'lag_7', 'lag_14', 'buddy_lag_1_sales', 'is_holiday', 'season', 'is_event']
target_col = '매출수량'

# 로그 변환 적용 (매출수량이 NaN이 아닌 경우에만)
combined_df.loc[combined_df[target_col].notna(), target_col] = combined_df.loc[combined_df[target_col].notna(), target_col].apply(lambda x: np.log1p(x) if x > 0 else 0)

# 학습 데이터에 기반해 스케일러를 학습시키고, 동일한 스케일을 테스트 데이터에도 적용
scaler = MinMaxScaler()
scaler.fit(combined_df[combined_df['source'] == 'train'][features_to_scale])
combined_df[features_to_scale] = scaler.transform(combined_df[features_to_scale])

scalers = {}
for item_id in tqdm(combined_df['영업장명_메뉴명'].unique(), desc="Scaling target by item"):
    scaler = MinMaxScaler()
    item_sales = combined_df.loc[combined_df['영업장명_메뉴명'] == item_id, target_col].values.reshape(-1, 1)
    
    # 학습 데이터에만 fit_transform 적용 (NaN이 아닌 값만 사용)
    train_sales = item_sales[~np.isnan(item_sales).squeeze()]
    if len(train_sales) > 0:
        scaler.fit(train_sales.reshape(-1, 1))
        
        # 전체 데이터에 transform 적용 (NaN 포함)
        combined_df.loc[combined_df['영업장명_메뉴명'] == item_id, target_col] = scaler.transform(item_sales).flatten()
        scalers[item_id] = scaler

def create_sequences(data, features, target, seq_length, predict_length):
    xs, ys, item_ids = [], [], []
    for item_id, group in data.groupby('영업장명_메뉴명'):
        feature_data = group[features].values
        target_data = group[target].values
        for i in range(len(group) - seq_length - predict_length + 1):
            xs.append(feature_data[i:i+seq_length])
            ys.append(target_data[i+seq_length:i+seq_length+predict_length])
            item_ids.append(item_id)
    return np.array(xs), np.array(ys), np.array(item_ids)

features = features_to_scale
train_data = combined_df[combined_df['매출수량'].notna()]
X, y, item_ids = create_sequences(train_data, features, target_col, SEQUENCE_LENGTH, PREDICT_LENGTH)

X_train, X_val, y_train, y_val, item_ids_train, item_ids_val = \
    X[:int(len(X)*0.9)], X[int(len(X)*0.9):], \
    y[:int(len(y)*0.9)], y[int(len(y)*0.9):], \
    item_ids[:int(len(item_ids)*0.9)], item_ids[int(len(item_ids)*0.9):]

# --- 3. PyTorch Dataset 및 DataLoader ---
print("Step 3: Creating PyTorch Datasets and Dataloaders...")
class SalesDataset(Dataset):
    def __init__(self, X, y, item_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.item_ids = item_ids
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.item_ids[idx]

train_dataset = SalesDataset(X_train, y_train, item_ids_train)
val_dataset = SalesDataset(X_val, y_val, item_ids_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. LSTM 모델 정의 ---
print("Step 4: Defining LSTM model...")
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        return out

model = LSTMModel(input_size=len(features), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=PREDICT_LENGTH).to(DEVICE)
criterion = nn.SmoothL1Loss()
smape_loss_fn = SMAPELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# --- 5. 모델 훈련 및 검증 ---
print("Step 5: Training and validating model...")
best_val_smape = float('inf')
patience_counter = 0

epoch_iterator = tqdm(range(NUM_EPOCHS), desc="Training Epochs")
for epoch in epoch_iterator:
    model.train()
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) + smape_loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss, all_preds, all_labels, all_item_ids = 0, [], [], []
    with torch.no_grad():
        for inputs, labels, batch_item_ids in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels) + smape_loss_fn(outputs, labels)
            val_loss += batch_loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_item_ids.append(np.repeat(batch_item_ids, PREDICT_LENGTH))
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss) # 스케줄러 step

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_item_ids = np.concatenate(all_item_ids, axis=0)

    all_preds_flat = all_preds.reshape(-1, 1)
    all_labels_flat = all_labels.reshape(-1, 1)

    # 역변환 로직
    all_preds_unscaled = np.zeros_like(all_preds_flat)
    all_labels_unscaled = np.zeros_like(all_labels_flat)

    for i in range(len(all_preds_flat)):
        item_id = all_item_ids[i]
        if item_id in scalers:
            pred_unscaled = scalers[item_id].inverse_transform(all_preds_flat[i].reshape(-1, 1))
            label_unscaled = scalers[item_id].inverse_transform(all_labels_flat[i].reshape(-1, 1))

            pred_original = np.expm1(pred_unscaled)
            label_original = np.expm1(label_unscaled)

            pred_original[pred_original < 0] = 0
            label_original[label_original < 0] = 0

            all_preds_unscaled[i] = pred_original
            all_labels_unscaled[i] = label_original
        else:
            all_preds_unscaled[i] = np.expm1(all_preds_flat[i])
            all_labels_unscaled[i] = np.expm1(all_labels_flat[i])
            all_preds_unscaled[i][all_preds_unscaled[i] < 0] = 0
            all_labels_unscaled[i][all_labels_unscaled[i] < 0] = 0
    val_smape = smape(all_labels_unscaled, all_preds_unscaled)

    epoch_iterator.set_postfix(val_loss=f"{val_loss:.6f}", val_smape=f"{val_smape:.4f}")

    if val_smape < best_val_smape:
        best_val_smape = val_smape
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        tqdm.write(f"Epoch {epoch+1}: Validation SMAPE improved to {val_smape:.4f}. Saving model...")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs of no improvement.")
            break



# --- 6. 최종 예측 및 제출 ---
print("Step 6: Predicting and creating submission file with recursive forecasting (including buddy feature)...")

# 가장 좋았던 모델 가중치 로드
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

# 전체 데이터프레임 복사하여 예측용으로 사용
recursive_df = combined_df.copy()

# 예측 대상이 되는 날짜들을 순서대로 가져옴
prediction_dates = sorted(recursive_df[recursive_df['매출수량'].isna()]['영업일자'].unique())

# 예측 대상 인덱스 저장 (최종 결과 추출용)
test_indices = recursive_df[recursive_df['매출수량'].isna()].index

with torch.no_grad():
    for start_idx in tqdm(range(0, len(prediction_dates), PREDICT_LENGTH), desc="Recursive Prediction by Date"):
        current_dates = prediction_dates[start_idx:start_idx + PREDICT_LENGTH]

        batch_item_ids = recursive_df[recursive_df['영업일자'].isin(current_dates)]['영업장명_메뉴명'].unique()
        batch_predictions = {}
        for item_id in batch_item_ids:
            item_history = recursive_df[(recursive_df['영업장명_메뉴명'] == item_id) & (recursive_df['영업일자'] < current_dates[0])]
            sequence_data = item_history.tail(SEQUENCE_LENGTH)
            if len(sequence_data) < SEQUENCE_LENGTH or sequence_data[target_col].isna().all():
                buddy_id = recursive_df.loc[
                    recursive_df['영업장명_메뉴명'] == item_id, 'best_buddy'
                ].iloc[0]
                init_val = np.nan
                if pd.notna(buddy_id):
                    buddy_history = recursive_df[
                        (recursive_df['영업장명_메뉴명'] == buddy_id)
                        & (recursive_df['영업일자'] < current_dates[0])
                    ]
                    buddy_sales = buddy_history[target_col].dropna()
                    if not buddy_sales.empty:
                        init_val = buddy_sales.iloc[-1]
                if np.isnan(init_val):
                    overall_mean = recursive_df[target_col].dropna().mean()
                    init_val = overall_mean if not np.isnan(overall_mean) else 0.1
                predicted_seq = np.repeat(init_val, len(current_dates))
            else:
                input_features = sequence_data[features].values
                input_tensor = torch.tensor([input_features], dtype=torch.float32).to(DEVICE)
                prediction_scaled = model(input_tensor).cpu().numpy()[0]
                predicted_seq = prediction_scaled[:len(current_dates)]
            batch_predictions[item_id] = predicted_seq

        for offset, current_date in enumerate(current_dates):
            day_predictions = {item_id: preds[offset] for item_id, preds in batch_predictions.items()}

            for item_id, pred_val in day_predictions.items():
                idx_to_update = recursive_df[
                    (recursive_df['영업일자'] == current_date) &
                    (recursive_df['영업장명_메뉴명'] == item_id)
                ].index
                if not idx_to_update.empty:
                    recursive_df.loc[idx_to_update, '매출수량'] = pred_val

            for item_id, pred_val in day_predictions.items():
                for lag_days in [1, 7, 14]:
                    future_date = get_future_date_str(current_date, lag_days)
                    future_idx = recursive_df.index[
                        (recursive_df['영업일자'] == future_date) &
                        (recursive_df['영업장명_메뉴명'] == item_id)
                    ]
                    if not future_idx.empty:
                        recursive_df.loc[future_idx[0], f'lag_{lag_days}'] = pred_val

            next_day = get_future_date_str(current_date, 1)
            next_day_rows_idx = recursive_df[recursive_df['영업일자'] == next_day].index
            for idx in next_day_rows_idx:
                buddy_item_id = recursive_df.loc[idx, 'best_buddy']
                if pd.notna(buddy_item_id) and buddy_item_id in day_predictions:
                    recursive_df.loc[idx, 'buddy_lag_1_sales'] = day_predictions[buddy_item_id]


# --- 4. 최종 결과 처리 ---
# 예측된 값들의 스케일을 원래대로 복원
# target_scaler 대신 scalers 딕셔너리를 사용하여 품목별 역변환 수행
# recursive_df.loc[test_indices, '매출수량'] = predicted_values_unscaled.flatten() # 이 라인은 아래 for 루프에서 처리됨

# submission_df를 recursive_df.loc[test_indices]로 초기화하여 사용
submission_df_for_inverse = recursive_df.loc[test_indices].copy()

for item_id in tqdm(submission_df_for_inverse['영업장명_메뉴명'].unique(), desc="Inverse transforming predictions"):
    if item_id in scalers:
        item_indices = submission_df_for_inverse[submission_df_for_inverse['영업장명_메뉴명'] == item_id].index
        predicted_values_scaled = submission_df_for_inverse.loc[item_indices, '매출수량'].values.reshape(-1, 1)
        
        # 스케일러 역변환
        predicted_values_unscaled = scalers[item_id].inverse_transform(predicted_values_scaled)
        
        # 로그 변환 역변환
        predicted_values_original = np.expm1(predicted_values_unscaled)
        
        # 음수 값은 0으로 처리
        predicted_values_original[predicted_values_original < 0] = 0
        
        submission_df_for_inverse.loc[item_indices, '매출수량'] = predicted_values_original.flatten()
    else:
        # 스케일러가 없는 품목은 로그 변환만 역변환
        item_indices = submission_df_for_inverse[submission_df_for_inverse['영업장명_메뉴명'] == item_id].index
        predicted_values_scaled = submission_df_for_inverse.loc[item_indices, '매출수량'].values

        # 로그 변환 역변환
        predicted_values_original = np.expm1(predicted_values_scaled)

        # 음수 값은 0으로 처리
        predicted_values_original[predicted_values_original < 0] = 0

        submission_df_for_inverse.loc[item_indices, '매출수량'] = predicted_values_original

# 최종적으로 recursive_df에 역변환된 값을 반영
recursive_df.loc[test_indices, '매출수량'] = submission_df_for_inverse['매출수량']

# --- 5. 제출 파일 생성 ---
submission_df = (
    recursive_df.loc[test_indices]
    .pivot_table(index='영업일자', columns='영업장명_메뉴명', values='매출수량')
    .reset_index()
)
final_submission = sample_submission_df[['영업일자']].merge(
    submission_df, on='영업일자', how='left'
)

# 결측치 확인: 존재하면 상세 정보 출력 후 실행 중단
na_counts = final_submission.isna().sum()
if na_counts.sum() > 0:
    print("Missing values detected in final submission columns:")
    print(na_counts[na_counts > 0])
    for col in final_submission.columns:
        if col != '영업일자' and na_counts[col] > 0:
            missing_dates = final_submission.loc[final_submission[col].isna(), '영업일자']
            print(f"Column '{col}' has missing dates: {missing_dates.tolist()}")
    raise ValueError("NaNs found in final submission. Aborting.")

final_submission = final_submission[sample_submission_df.columns]

# 제출 직전에만 반올림하여 정수로 변환
value_columns = final_submission.columns.drop('영업일자')
final_submission[value_columns] = np.round(final_submission[value_columns]).astype(int)

final_submission.to_csv("lstm_submission_recursive_full.csv", index=False)
print("Submission file created successfully at: lstm_submission_recursive_full.csv")
