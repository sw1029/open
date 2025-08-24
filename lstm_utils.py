import os
import glob
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import holidays


# ---------------------------------------------------------------------------
# Utility functions and classes reused by LSTM training scripts
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smape(y_true, y_pred, eps: float = 1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(numerator / denominator) * 100


class SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + self.eps
        return (numerator / denominator).mean()


def get_future_date_str(date_str: str, days_to_add: int, mapping: Dict[str, str]):
    try:
        parts = date_str.replace('일', '').split('+')
        test_id = parts[0]
        day_num = int(parts[1])
        return f"{test_id}+{day_num + days_to_add}일"
    except (IndexError, ValueError):
        base = mapping.get(str(pd.to_datetime(date_str).date()))
        if base:
            parts = base.replace('일', '').split('+')
            test_id = parts[0]
            day_num = int(parts[1])
            return f"{test_id}+{day_num + days_to_add}일"
        future_date = pd.to_datetime(date_str) + pd.Timedelta(days=days_to_add)
        return f"TEST_{future_date.strftime('%Y-%m-%d')}"


def create_features_train(df: pd.DataFrame) -> pd.DataFrame:
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['dayofweek'] = df['영업일자'].dt.dayofweek
    df['month'] = df['영업일자'].dt.month
    return df


def create_features_test(df: pd.DataFrame) -> pd.DataFrame:
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'], errors='coerce')
    df['dayofweek'] = df['영업일자'].dt.dayofweek.fillna(-1).astype(int)
    df['month'] = df['영업일자'].dt.month.fillna(-1).astype(int)
    return df


def load_calendar_features(df: pd.DataFrame, event_path: str = 'events.csv') -> pd.DataFrame:
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    years = df['영업일자'].dt.year.unique()
    kr_holidays = holidays.KR(years=years, expand=True, observed=True)
    df['is_holiday'] = df['영업일자'].dt.date.isin(kr_holidays).astype(int)

    if os.path.exists(event_path):
        event_df = pd.read_csv(event_path)
        event_dates = pd.to_datetime(event_df['date']).dt.strftime('%Y-%m-%d').tolist()
        df['is_event'] = df['영업일자'].dt.strftime('%Y-%m-%d').isin(event_dates).astype(int)
    else:
        df['is_event'] = 0

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3

    df['season'] = df['month'].apply(month_to_season)
    return df


def create_sequences(data: pd.DataFrame, features, target, seq_length, predict_length):
    xs, ys, item_ids = [], [], []
    for item_id, group in data.groupby('영업장명_메뉴명'):
        feature_data = group[features].values
        target_data = group[target].values
        for i in range(len(group) - seq_length - predict_length + 1):
            xs.append(feature_data[i:i+seq_length])
            ys.append(target_data[i+seq_length:i+seq_length+predict_length])
            item_ids.append(item_id)
    return np.array(xs), np.array(ys), np.array(item_ids)


def time_warp(arr: np.ndarray, scale: float) -> np.ndarray:
    """Resample array along time axis by a scaling factor while keeping length."""
    n = arr.shape[0]
    idx = np.arange(n)
    warped_idx = idx * scale
    if arr.ndim == 1:
        return np.interp(idx, warped_idx, arr, left=arr[0], right=arr[-1])
    warped = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        warped[:, col] = np.interp(idx, warped_idx, arr[:, col], left=arr[0, col], right=arr[-1, col])
    return warped


def augment_sequences(X: np.ndarray, y: np.ndarray, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment sequences for specific stores using noise, scaling and time warping."""
    target_keywords = ['담하', '미라시아']
    aug_X, aug_y, aug_ids = [], [], []
    for x_seq, y_seq, item_id in zip(X, y, item_ids):
        if any(keyword in item_id for keyword in target_keywords):
            noise_x = x_seq + np.random.normal(0, 0.01, x_seq.shape)
            noise_y = y_seq + np.random.normal(0, 0.01, y_seq.shape)
            aug_X.append(noise_x)
            aug_y.append(noise_y)
            aug_ids.append(item_id)

            scale_factor = np.random.uniform(0.9, 1.1)
            aug_X.append(x_seq * scale_factor)
            aug_y.append(y_seq * scale_factor)
            aug_ids.append(item_id)

            warp_factor = np.random.uniform(0.8, 1.2)
            aug_X.append(time_warp(x_seq, warp_factor))
            aug_y.append(time_warp(y_seq, warp_factor))
            aug_ids.append(item_id)

    if aug_X:
        X = np.concatenate([X, np.array(aug_X)], axis=0)
        y = np.concatenate([y, np.array(aug_y)], axis=0)
        item_ids = np.concatenate([item_ids, np.array(aug_ids)], axis=0)
    return X, y, item_ids


class SalesDataset(Dataset):
    def __init__(self, X, y, item_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.item_ids = item_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.item_ids[idx]


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attn_score = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_score(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        out = self.fc(context)
        out = self.activation(out)
        return out


def prepare_datasets(sequence_length: int, predict_length: int, batch_size: int):
    """Load data, apply feature engineering and create train/val dataloaders."""
    logging.info("Preparing datasets with sequence_length=%s predict_length=%s", sequence_length, predict_length)

    train_df = pd.read_csv('train/train.csv')
    test_files = glob.glob('test/TEST_*.csv')
    test_df_list = []
    for file in test_files:
        temp_df = pd.read_csv(file)
        test_id = os.path.splitext(os.path.basename(file))[0]
        temp_df['영업일자'] = pd.to_datetime(temp_df['영업일자'])
        for _, g in temp_df.groupby('영업장명_메뉴명'):
            g = g.sort_values('영업일자')
            past = g.iloc[:-7].copy()
            past['submission_date'] = past['영업일자'].dt.strftime('%Y-%m-%d')
            future = g.tail(7).copy().reset_index(drop=True)
            future['submission_date'] = [f"{test_id}+{i+1}일" for i in range(len(future))]
            future['매출수량'] = np.nan
            past['test_id'] = test_id
            future['test_id'] = test_id
            test_df_list.append(pd.concat([past, future], ignore_index=True))
    test_df = pd.concat(test_df_list, ignore_index=True)

    submission_date_map = test_df.set_index(test_df['영업일자'].astype(str))['submission_date'].to_dict()
    submission_to_date_map = test_df.set_index('submission_date')['영업일자'].astype(str).to_dict()

    expected_test_nans = 7 * test_df[['test_id', '영업장명_메뉴명']].drop_duplicates().shape[0]
    actual_test_nans = test_df['매출수량'].isna().sum()
    if actual_test_nans != expected_test_nans:
        raise ValueError("Unexpected number of NaNs in test data")
    nans_per_item = test_df[test_df['매출수량'].isna()].groupby(['test_id', '영업장명_메뉴명']).size()
    if not (nans_per_item == 7).all():
        raise ValueError("Each test_id/item pair must have exactly 7 NaNs.")

    sample_submission_df = pd.read_csv('sample_submission.csv')

    train_df = create_features_train(train_df)
    test_df = create_features_test(test_df)
    train_df = load_calendar_features(train_df)
    test_df = load_calendar_features(test_df)

    for col in ['영업장명', '메뉴명']:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col + '_encoded'] = le.transform(train_df[col])
        test_df[col + '_encoded'] = le.transform(test_df[col])

    lags = [1, 7, 14, 28]
    train_df = train_df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    for lag in lags:
        train_df[f'lag_{lag}'] = train_df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
    test_df = test_df.sort_values(by=['test_id', '영업장명_메뉴명', '영업일자'])
    for lag in lags:
        test_df[f'lag_{lag}'] = test_df.groupby(['test_id', '영업장명_메뉴명'])['매출수량'].shift(lag)

    corr_files = glob.glob('data/*.csv')
    corr_matrices = {os.path.basename(f).replace('.csv', ''): pd.read_csv(f, index_col=0) for f in corr_files}
    best_buddy_map = {
        (store, menu): corr_matrix[menu].drop(menu).idxmax()
        for store, corr_matrix in corr_matrices.items()
        for menu in corr_matrix.columns
    }
    train_df['best_buddy'] = train_df.set_index(['영업장명', '메뉴명']).index.map(best_buddy_map.get)
    test_df['best_buddy'] = test_df.set_index(['영업장명', '메뉴명']).index.map(best_buddy_map.get)

    lag1_train = train_df[['영업일자', '영업장명', '메뉴명', 'lag_1']].rename(columns={'lag_1': 'buddy_lag_1_sales'})
    train_df = pd.merge(
        train_df,
        lag1_train,
        left_on=['영업일자', '영업장명', 'best_buddy'],
        right_on=['영업일자', '영업장명', '메뉴명'],
        how='left',
        suffixes=('', '_buddy')
    )
    train_df.drop(columns=['메뉴명_buddy'], inplace=True)

    lag1_test = test_df[['test_id', '영업일자', '영업장명', '메뉴명', 'lag_1']].rename(columns={'lag_1': 'buddy_lag_1_sales'})
    test_df = pd.merge(
        test_df,
        lag1_test,
        left_on=['test_id', '영업일자', '영업장명', 'best_buddy'],
        right_on=['test_id', '영업일자', '영업장명', '메뉴명'],
        how='left',
        suffixes=('', '_buddy')
    )
    test_df.drop(columns=['메뉴명_buddy'], inplace=True)

    cols_to_fill = [f'lag_{lag}' for lag in lags] + ['buddy_lag_1_sales']
    train_df[cols_to_fill] = train_df[cols_to_fill].fillna(0)
    test_df[cols_to_fill] = test_df[cols_to_fill].fillna(0)

    train_df['영업일자'] = train_df['영업일자'].astype(str)
    test_df['영업일자'] = test_df['영업일자'].astype(str)

    train_df['source'] = 'train'
    test_df['source'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    combined_nan_count = combined_df['매출수량'].isna().sum()
    expected_test_nans = 7 * test_df[['test_id', '영업장명_메뉴명']].drop_duplicates().shape[0]
    if combined_nan_count != expected_test_nans:
        raise ValueError("Combined dataframe NaN mismatch")

    features_to_scale = ['dayofweek', 'month', '영업장명_encoded', '메뉴명_encoded',
                         'lag_1', 'lag_7', 'lag_14', 'lag_28', 'buddy_lag_1_sales',
                         'is_holiday', 'season', 'is_event']
    target_col = '매출수량'

    combined_df.loc[combined_df[target_col].notna(), target_col] = \
        combined_df.loc[combined_df[target_col].notna(), target_col].apply(lambda x: np.log1p(x) if x > 0 else 0)

    scaler = MinMaxScaler()
    scaler.fit(combined_df[combined_df['source'] == 'train'][features_to_scale])
    combined_df[features_to_scale] = scaler.transform(combined_df[features_to_scale])

    scalers = {}
    for item_id in tqdm(combined_df['영업장명_메뉴명'].unique(), desc="Scaling target by item"):
        scaler_item = MinMaxScaler()
        item_sales = combined_df.loc[combined_df['영업장명_메뉴명'] == item_id, target_col].values.reshape(-1, 1)
        train_sales = item_sales[~np.isnan(item_sales).squeeze()]
        if len(train_sales) > 0:
            scaler_item.fit(train_sales.reshape(-1, 1))
            combined_df.loc[combined_df['영업장명_메뉴명'] == item_id, target_col] = scaler_item.transform(item_sales).flatten()
            scalers[item_id] = scaler_item

    features = features_to_scale
    train_data = combined_df[combined_df['매출수량'].notna()]
    X, y, item_ids = create_sequences(train_data, features, target_col, sequence_length, predict_length)

    # Augment sequences for specified stores to expand training data
    X, y, item_ids = augment_sequences(X, y, item_ids)

    X_train, X_val = X[:int(len(X)*0.9)], X[int(len(X)*0.9):]
    y_train, y_val = y[:int(len(y)*0.9)], y[int(len(y)*0.9):]
    item_ids_train, item_ids_val = item_ids[:int(len(item_ids)*0.9)], item_ids[int(len(item_ids)*0.9):]

    train_dataset = SalesDataset(X_train, y_train, item_ids_train)
    val_dataset = SalesDataset(X_val, y_val, item_ids_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_indices = combined_df[combined_df['매출수량'].isna()].index

    return train_loader, val_loader, scalers, combined_df, features, target_col, sample_submission_df, submission_date_map, submission_to_date_map, test_indices


def predict_and_submit(model: nn.Module, combined_df: pd.DataFrame, scalers: Dict[str, MinMaxScaler],
                        features, target_col: str, sample_submission_df: pd.DataFrame,
                        submission_date_map: Dict[str, str], submission_to_date_map: Dict[str, str],
                        test_indices, sequence_length: int, predict_length: int,
                        output_path: str = "lstm_attention_submission.csv"):
    model.eval()
    recursive_df = combined_df.copy()
    prediction_dates = sorted(
        recursive_df[recursive_df['매출수량'].isna()]['submission_date'].unique(),
        key=lambda x: (x.split('+')[0], int(x.split('+')[1].replace('일', '')))
    )

    test_idx_series = recursive_df[recursive_df['매출수량'].isna()].index

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(prediction_dates), predict_length), desc="Recursive Prediction by Date"):
            current_dates = prediction_dates[start_idx:start_idx + predict_length]
            batch_item_ids = recursive_df[recursive_df['submission_date'].isin(current_dates)]['영업장명_메뉴명'].unique()
            batch_predictions: Dict[str, Any] = {}
            for item_id in batch_item_ids:
                cutoff_date = submission_to_date_map.get(current_dates[0], current_dates[0])
                item_history = recursive_df[
                    (recursive_df['영업장명_메뉴명'] == item_id) &
                    (recursive_df['영업일자'] < cutoff_date)
                ]
                sequence_data = item_history.tail(sequence_length)
                if len(sequence_data) < sequence_length or sequence_data[target_col].isna().all():
                    buddy_id = recursive_df.loc[
                        recursive_df['영업장명_메뉴명'] == item_id, 'best_buddy'
                    ].iloc[0]
                    init_val = np.nan
                    if pd.notna(buddy_id):
                        buddy_history = recursive_df[
                            (recursive_df['영업장명_메뉴명'] == buddy_id) &
                            (recursive_df['영업일자'] < cutoff_date)
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
                    input_tensor = torch.tensor(np.array([input_features]), dtype=torch.float32).to(DEVICE)
                    prediction_scaled = model(input_tensor).cpu().numpy()[0]
                    predicted_seq = prediction_scaled[:len(current_dates)]
                batch_predictions[item_id] = predicted_seq

            for offset, current_date in enumerate(current_dates):
                day_predictions = {item_id: preds[offset] for item_id, preds in batch_predictions.items()}
                for item_id, pred_val in day_predictions.items():
                    idx_to_update = recursive_df[
                        (recursive_df['submission_date'] == current_date) &
                        (recursive_df['영업장명_메뉴명'] == item_id)
                    ].index
                    if not idx_to_update.empty:
                        recursive_df.loc[idx_to_update, '매출수량'] = pred_val
                for item_id, pred_val in day_predictions.items():
                    for lag_days in [1, 7, 14]:
                        future_date = get_future_date_str(current_date, lag_days, submission_date_map)
                        future_idx = recursive_df.index[
                            (recursive_df['submission_date'] == future_date) &
                            (recursive_df['영업장명_메뉴명'] == item_id)
                        ]
                        if not future_idx.empty:
                            recursive_df.loc[future_idx[0], f'lag_{lag_days}'] = pred_val
                next_day = get_future_date_str(current_date, 1, submission_date_map)
                next_day_rows_idx = recursive_df[recursive_df['submission_date'] == next_day].index
                for idx in next_day_rows_idx:
                    buddy_item_id = recursive_df.loc[idx, 'best_buddy']
                    if pd.notna(buddy_item_id) and buddy_item_id in day_predictions:
                        recursive_df.loc[idx, 'buddy_lag_1_sales'] = day_predictions[buddy_item_id]

    submission_df_for_inverse = recursive_df.loc[test_idx_series].copy()
    for item_id in tqdm(submission_df_for_inverse['영업장명_메뉴명'].unique(), desc="Inverse transforming predictions"):
        if item_id in scalers:
            item_indices = submission_df_for_inverse[submission_df_for_inverse['영업장명_메뉴명'] == item_id].index
            predicted_values_scaled = submission_df_for_inverse.loc[item_indices, '매출수량'].values.reshape(-1, 1)
            predicted_values_unscaled = scalers[item_id].inverse_transform(predicted_values_scaled)
            predicted_values_original = np.expm1(predicted_values_unscaled)
            predicted_values_original[predicted_values_original < 0] = 0
            submission_df_for_inverse.loc[item_indices, '매출수량'] = predicted_values_original.flatten()
        else:
            item_indices = submission_df_for_inverse[submission_df_for_inverse['영업장명_메뉴명'] == item_id].index
            predicted_values_scaled = submission_df_for_inverse.loc[item_indices, '매출수량'].values
            predicted_values_original = np.expm1(predicted_values_scaled)
            predicted_values_original[predicted_values_original < 0] = 0
            submission_df_for_inverse.loc[item_indices, '매출수량'] = predicted_values_original

    recursive_df.loc[test_idx_series, '매출수량'] = submission_df_for_inverse['매출수량']

    submission_df = (
        recursive_df.loc[test_idx_series]
        .pivot_table(index='submission_date', columns='영업장명_메뉴명', values='매출수량')
        .reset_index()
    )
    final_submission = sample_submission_df[['영업일자']].merge(
        submission_df, left_on='영업일자', right_on='submission_date', how='left'
    )
    final_submission.drop(columns=['submission_date'], inplace=True)

    value_columns = final_submission.columns.drop('영업일자')
    na_mask = final_submission[value_columns].isna()
    if na_mask.any().any():
        for col in value_columns:
            missing_dates = final_submission.loc[na_mask[col], '영업일자']
            if not missing_dates.empty:
                logging.warning("NaN detected for item '%s' on dates: %s", col, missing_dates.tolist())
    final_submission[value_columns] = final_submission[value_columns].fillna(0)
    final_submission[value_columns] = np.round(final_submission[value_columns]).astype(int)
    final_submission.to_csv(output_path, index=False)
    return output_path
