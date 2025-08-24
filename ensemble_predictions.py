import pandas as pd
import numpy as np
from typing import Optional, Dict


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric mean absolute percentage error with small epsilon."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(numerator / denominator) * 100


def ensemble_predictions(
    lstm_path: str = "lstm_attention_submission.csv",
    gru_path: str = "gru_submission.csv",
    output_path: str = "ensemble_submission.csv",
    w_lstm: float = 0.5,
    w_gru: float = 0.5,
    target_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create an ensemble of LSTM and GRU predictions.

    Parameters
    ----------
    lstm_path, gru_path : str
        Paths to LSTM and GRU prediction csv files.
    output_path : str
        File path to save the ensembled predictions.
    w_lstm, w_gru : float
        Weights for LSTM and GRU predictions. Weighted average is
        normalized by the sum of weights.
    target_path : Optional[str]
        Path to ground truth CSV for SMAPE evaluation. The ground truth
        should have the same structure and ``영업일자`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ensembled predictions.
    """
    lstm_df = pd.read_csv(lstm_path)
    gru_df = pd.read_csv(gru_path)

    merged = pd.merge(lstm_df, gru_df, on="영업일자", suffixes=("_lstm", "_gru"))

    result = merged[["영업일자"]].copy()
    pred_cols = [c for c in lstm_df.columns if c != "영업일자"]
    total_weight = w_lstm + w_gru
    for col in pred_cols:
        result[col] = (
            merged[f"{col}_lstm"] * w_lstm + merged[f"{col}_gru"] * w_gru
        ) / total_weight

    result.to_csv(output_path, index=False)

    if target_path is not None:
        truth_df = pd.read_csv(target_path)
        # Align columns in case of different ordering
        truth_df = truth_df[["영업일자"] + pred_cols]
        joined = pd.merge(truth_df, result, on="영업일자", suffixes=("_true", "_pred"))
        smape_scores: Dict[str, float] = {}
        for col in pred_cols:
            smape_scores[col] = smape(joined[f"{col}_true"], joined[f"{col}_pred"])
        print(f"Mean SMAPE: {np.mean(list(smape_scores.values())):.4f}")
    return result


if __name__ == "__main__":
    ensemble_predictions()
