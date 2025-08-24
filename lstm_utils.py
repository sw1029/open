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
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + self.eps
        loss = numerator / denominator
        if weights is not None:
            loss = loss * weights
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


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


def create_sequences(
    data: pd.DataFrame,
    features,
    target,
    seq_length,
    predict_length,
    future_features=None,
):
    """Create rolling window sequences.

    When ``future_features`` is provided, an additional array containing the
    features for the prediction horizon is returned for each sequence. These
    features should correspond to exogenous variables that are known in
    advance (e.g. day of week, holiday flags).
    """

    xs, ys, future_feats, item_ids, store_idxs, item_idxs = [], [], [], [], [], []
    for item_id, group in data.groupby('영업장명_메뉴명'):
        feature_data = group[features].values
        target_data = group[target].values
        store_idx = group['영업장명_encoded'].iloc[0]
        item_idx = group['메뉴명_encoded'].iloc[0]
        future_feat_data = (
            group[future_features].values if future_features is not None else None
        )
        max_start = len(group) - seq_length - predict_length + 1
        for i in range(max_start):
            xs.append(feature_data[i : i + seq_length])
            ys.append(target_data[i + seq_length : i + seq_length + predict_length])
            if future_features is not None:
                future_feats.append(
                    future_feat_data[
                        i + seq_length : i + seq_length + predict_length
                    ]
                )
            item_ids.append(item_id)
            store_idxs.append(store_idx)
            item_idxs.append(item_idx)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    item_ids_arr = np.array(item_ids)
    store_idx_arr = np.array(store_idxs)
    item_idx_arr = np.array(item_idxs)
    if future_features is not None:
        return (
            xs_arr,
            ys_arr,
            np.array(future_feats),
            item_ids_arr,
            store_idx_arr,
            item_idx_arr,
        )
    return xs_arr, ys_arr, item_ids_arr, store_idx_arr, item_idx_arr


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


def augment_sequences(
    X: np.ndarray,
    y: np.ndarray,
    item_ids: np.ndarray,
    future_feats: np.ndarray | None = None,
    store_idxs: np.ndarray | None = None,
    item_idxs: np.ndarray | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Augment sequences for specific stores using noise, scaling and time warping."""

    target_keywords = ['담하', '미라시아']
    aug_X, aug_y, aug_ids, aug_future, aug_store, aug_item = [], [], [], [], [], []
    for idx, (x_seq, y_seq, item_id) in enumerate(zip(X, y, item_ids)):
        fut_seq = future_feats[idx] if future_feats is not None else None
        store_idx = store_idxs[idx] if store_idxs is not None else None
        item_idx = item_idxs[idx] if item_idxs is not None else None
        if any(keyword in item_id for keyword in target_keywords):
            noise_x = x_seq + np.random.normal(0, 0.01, x_seq.shape)
            noise_y = y_seq + np.random.normal(0, 0.01, y_seq.shape)
            aug_X.append(noise_x)
            aug_y.append(noise_y)
            aug_ids.append(item_id)
            if fut_seq is not None:
                aug_future.append(fut_seq)
            if store_idx is not None:
                aug_store.append(store_idx)
            if item_idx is not None:
                aug_item.append(item_idx)

            scale_factor = np.random.uniform(0.9, 1.1)
            aug_X.append(x_seq * scale_factor)
            aug_y.append(y_seq * scale_factor)
            aug_ids.append(item_id)
            if fut_seq is not None:
                aug_future.append(fut_seq)
            if store_idx is not None:
                aug_store.append(store_idx)
            if item_idx is not None:
                aug_item.append(item_idx)

            warp_factor = np.random.uniform(0.8, 1.2)
            aug_X.append(time_warp(x_seq, warp_factor))
            aug_y.append(time_warp(y_seq, warp_factor))
            aug_ids.append(item_id)
            if fut_seq is not None:
                aug_future.append(fut_seq)
            if store_idx is not None:
                aug_store.append(store_idx)
            if item_idx is not None:
                aug_item.append(item_idx)

    if aug_X:
        X = np.concatenate([X, np.array(aug_X)], axis=0)
        y = np.concatenate([y, np.array(aug_y)], axis=0)
        item_ids = np.concatenate([item_ids, np.array(aug_ids)], axis=0)
        if future_feats is not None:
            future_feats = np.concatenate([future_feats, np.array(aug_future)], axis=0)
        if store_idxs is not None:
            store_idxs = np.concatenate([store_idxs, np.array(aug_store)], axis=0)
        if item_idxs is not None:
            item_idxs = np.concatenate([item_idxs, np.array(aug_item)], axis=0)
    return X, y, item_ids, future_feats, store_idxs, item_idxs


class SalesDataset(Dataset):
    """Dataset returning sequences, targets, ids, future features and embedding indices."""

    def __init__(self, X, y, item_ids, future_feats, store_idxs, item_idxs):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.future_feats = torch.tensor(future_feats, dtype=torch.float32)
        self.item_ids = item_ids
        self.store_idxs = torch.tensor(store_idxs, dtype=torch.long)
        self.item_idxs = torch.tensor(item_idxs, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.item_ids[idx],
            self.future_feats[idx],
            self.store_idxs[idx],
            self.item_idxs[idx],
        )


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


class Encoder(nn.Module):
    """LSTM encoder preceded by a temporal convolution."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        cnn_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("Encoder expects num_layers to be at least 3")
        # Temporal convolution over the feature dimension
        padding = (kernel_size - 1) // 2
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.lstm = nn.LSTM(
            cnn_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor):
        # x expected shape: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, time, channels)
        outputs, (h, c) = self.lstm(x)
        # Combine forward and backward states for each layer
        h = torch.cat((h[0::2], h[1::2]), dim=2)
        c = torch.cat((c[0::2], c[1::2]), dim=2)
        return outputs, (h, c)


class Decoder(nn.Module):
    """LSTM decoder with multi-head attention over encoder outputs."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        num_heads: int,
        decoder_steps: int,
        future_feat_dim: int,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("Decoder expects num_layers to be at least 3")
        self.decoder_steps = decoder_steps
        self.output_size = output_size
        self.future_feat_dim = future_feat_dim
        self.lstm = nn.LSTM(
            output_size + future_feat_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,
        )
        # Expect encoder outputs and hidden states of size ``hidden_size`` (i.e.,
        # twice the base encoder hidden size due to bidirectionality)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        hidden,
        future_feats: torch.Tensor,
        target_len: int | None = None,
        targets: torch.Tensor | None = None,
        scheduled_sampling_prob: float = 0.0,
    ):
        """Decode with optional scheduled sampling.

        During training, with probability ``scheduled_sampling_prob`` the model's
        previous prediction is fed back as the next input instead of the ground
        truth. This helps the model adapt to its own mistakes during inference.
        """
        if target_len is None:
            target_len = self.decoder_steps
        batch_size = encoder_outputs.size(0)

        prev_output = torch.zeros(
            batch_size, 1, self.output_size, device=encoder_outputs.device
        )
        decoder_hidden = hidden
        outputs = []
        for t in range(target_len):
            feat_t = future_feats[:, t].unsqueeze(1)
            decoder_input = torch.cat([prev_output, feat_t], dim=2)
            decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)
            attn_output, _ = self.attn(decoder_output, encoder_outputs, encoder_outputs)
            step_output = self.fc(attn_output)
            step_output = self.activation(step_output)
            outputs.append(step_output)

            if targets is not None and self.training:
                teacher_input = targets[:, t].view(batch_size, 1, self.output_size)
                use_model_pred = (
                    torch.rand(batch_size, 1, 1, device=encoder_outputs.device)
                    < scheduled_sampling_prob
                )
                prev_output = torch.where(
                    use_model_pred,
                    step_output.detach(),
                    teacher_input,
                )
            else:
                prev_output = step_output.detach()

        output = torch.cat(outputs, dim=1).squeeze(-1)
        return output


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with LSTM encoder and attention decoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        num_heads: int,
        decoder_steps: int,
        cnn_channels: int | None = None,
        kernel_size: int = 3,
        future_feat_dim: int = 0,
        num_stores: int = 0,
        num_items: int = 0,
        emb_dim_store: int = 0,
        emb_dim_item: int = 0,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = input_size
        self.store_emb = nn.Embedding(num_stores, emb_dim_store)
        self.item_emb = nn.Embedding(num_items, emb_dim_item)
        total_input_size = input_size + emb_dim_store + emb_dim_item
        self.encoder = Encoder(
            total_input_size,
            hidden_size,
            num_layers,
            cnn_channels,
            kernel_size,
        )
        self.decoder = Decoder(
            hidden_size * 2,
            num_layers,
            output_size,
            num_heads,
            decoder_steps,
            future_feat_dim,
        )
        # Projection for residual connection from the last encoder input.
        self.residual_proj = nn.Linear(total_input_size, decoder_steps)

    def forward(
        self,
        x: torch.Tensor,
        store_idx: torch.Tensor,
        item_idx: torch.Tensor,
        target_len: int | None = None,
        targets: torch.Tensor | None = None,
        scheduled_sampling_prob: float = 0.0,
        future_feats: torch.Tensor | None = None,
    ):
        if future_feats is None:
            raise ValueError("future_feats must be provided for decoder inputs")
        store_emb = self.store_emb(store_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        item_emb = self.item_emb(item_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, store_emb, item_emb], dim=2)
        encoder_outputs, hidden = self.encoder(x)
        outputs = self.decoder(
            encoder_outputs,
            hidden,
            future_feats,
            target_len,
            targets,
            scheduled_sampling_prob,
        )
        # Residual path projecting the last encoder input to decoder outputs.
        residual = self.residual_proj(x[:, -1])
        outputs = outputs + residual[:, : outputs.size(1)]
        return outputs


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

    # Rolling mean features
    train_df['roll7'] = (
        train_df.groupby('영업장명_메뉴명')['매출수량']
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    train_df['roll28'] = (
        train_df.groupby('영업장명_메뉴명')['매출수량']
        .transform(lambda x: x.shift(1).rolling(28).mean())
    )
    test_df['roll7'] = (
        test_df.groupby(['test_id', '영업장명_메뉴명'])['매출수량']
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    test_df['roll28'] = (
        test_df.groupby(['test_id', '영업장명_메뉴명'])['매출수량']
        .transform(lambda x: x.shift(1).rolling(28).mean())
    )

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

    cols_to_fill = [f'lag_{lag}' for lag in lags] + ['buddy_lag_1_sales', 'roll7', 'roll28']
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

    # Numerical features used directly by the model (store/item indices handled via embeddings)
    features_to_scale = [
        'dayofweek',
        'month',
        'lag_1',
        'lag_7',
        'lag_14',
        'lag_28',
        'buddy_lag_1_sales',
        'roll7',
        'roll28',
        'is_holiday',
        'season',
        'is_event',
    ]
    # Features available for future time steps (do not rely on past sales)
    future_features = ['dayofweek', 'month', 'is_holiday', 'season', 'is_event']
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
    (
        X,
        y,
        future_feat_array,
        item_ids,
        store_idxs,
        item_idxs,
    ) = create_sequences(
        train_data,
        features,
        target_col,
        sequence_length,
        predict_length,
        future_features,
    )

    # Augment sequences for specified stores to expand training data
    (
        X,
        y,
        item_ids,
        future_feat_array,
        store_idxs,
        item_idxs,
    ) = augment_sequences(
        X, y, item_ids, future_feat_array, store_idxs, item_idxs
    )

    X_train, X_val = X[: int(len(X) * 0.9)], X[int(len(X) * 0.9) :]
    y_train, y_val = y[: int(len(y) * 0.9)], y[int(len(y) * 0.9) :]
    future_train, future_val = (
        future_feat_array[: int(len(future_feat_array) * 0.9)],
        future_feat_array[int(len(future_feat_array) * 0.9) :],
    )
    item_ids_train, item_ids_val = (
        item_ids[: int(len(item_ids) * 0.9)],
        item_ids[int(len(item_ids) * 0.9) :],
    )
    store_train, store_val = (
        store_idxs[: int(len(store_idxs) * 0.9)],
        store_idxs[int(len(store_idxs) * 0.9) :],
    )
    item_train, item_val = (
        item_idxs[: int(len(item_idxs) * 0.9)],
        item_idxs[int(len(item_idxs) * 0.9) :],
    )

    train_dataset = SalesDataset(
        X_train, y_train, item_ids_train, future_train, store_train, item_train
    )
    val_dataset = SalesDataset(
        X_val, y_val, item_ids_val, future_val, store_val, item_val
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_indices = combined_df[combined_df['매출수량'].isna()].index
    item_weights = {}
    for item_id in combined_df['영업장명_메뉴명'].unique():
        store = item_id.split('_')[0]
        item_weights[item_id] = 2.0 if store in ['담하', '미라시아'] else 1.0

    return (
        train_loader,
        val_loader,
        scalers,
        combined_df,
        features,
        future_features,
        target_col,
        sample_submission_df,
        submission_date_map,
        submission_to_date_map,
        test_indices,
        item_weights,
    )


def predict_and_submit(
    model: nn.Module,
    combined_df: pd.DataFrame,
    scalers: Dict[str, MinMaxScaler],
    features,
    future_features,
    target_col: str,
    sample_submission_df: pd.DataFrame,
    submission_date_map: Dict[str, str],
    submission_to_date_map: Dict[str, str],
    test_indices,
    sequence_length: int,
    predict_length: int,
    output_path: str = "lstm_attention_submission.csv",
):
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
                    future_inputs = recursive_df[
                        (recursive_df['영업장명_메뉴명'] == item_id)
                        & (recursive_df['submission_date'].isin(current_dates))
                    ][future_features].values
                    input_tensor = torch.tensor(
                        np.array([input_features]), dtype=torch.float32
                    ).to(DEVICE)
                    future_tensor = torch.tensor(
                        np.array([future_inputs]), dtype=torch.float32
                    ).to(DEVICE)
                    store_tensor = torch.tensor(
                        [sequence_data['영업장명_encoded'].iloc[0]], dtype=torch.long
                    ).to(DEVICE)
                    item_tensor = torch.tensor(
                        [sequence_data['메뉴명_encoded'].iloc[0]], dtype=torch.long
                    ).to(DEVICE)
                    prediction_scaled = model(
                        input_tensor,
                        store_tensor,
                        item_tensor,
                        target_len=len(current_dates),
                        future_feats=future_tensor,
                    ).cpu().numpy()[0]
                    predicted_seq = prediction_scaled[: len(current_dates)]
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
