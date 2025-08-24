import optuna
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lstm_utils import (
    DEVICE,
    SMAPELoss,
    LSTMAttention,
    prepare_datasets,
    predict_and_submit,
    smape,
)

SEQUENCE_LENGTH = 14
PREDICT_LENGTH = 7
BATCH_SIZE = 64
OPTUNA_EPOCHS = 10
RETRAIN_EPOCHS = 30

# Track the best SMAPE observed during Optuna trials
best_smape = float("inf")

(
    train_loader,
    val_loader,
    scalers,
    combined_df,
    features,
    target_col,
    sample_submission_df,
    submission_date_map,
    submission_to_date_map,
    test_indices,
) = prepare_datasets(SEQUENCE_LENGTH, PREDICT_LENGTH, BATCH_SIZE)


def objective(trial):
    global best_smape
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = LSTMAttention(
        input_size=len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=PREDICT_LENGTH,
    ).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    smape_loss_fn = SMAPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(OPTUNA_EPOCHS):
        model.train()
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) + smape_loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels, all_item_ids = [], [], []
    with torch.no_grad():
        for inputs, labels, batch_item_ids in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_item_ids.append(np.repeat(batch_item_ids, PREDICT_LENGTH))

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_item_ids = np.concatenate(all_item_ids, axis=0)
    all_preds_flat = all_preds.reshape(-1, 1)
    all_labels_flat = all_labels.reshape(-1, 1)
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

    smape_value = smape(all_labels_unscaled, all_preds_unscaled)
    if smape_value < best_smape:
        best_smape = smape_value
        model_path = f"best_model_trial_{trial.number}.pth"
        torch.save(model.state_dict(), model_path)
        trial.set_user_attr("best_model_path", model_path)
    return smape_value


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

best_trial = study.best_trial
best_params = best_trial.params
best_model = LSTMAttention(
    input_size=len(features),
    hidden_size=best_params["hidden_size"],
    num_layers=best_params["num_layers"],
    output_size=PREDICT_LENGTH,
).to(DEVICE)
best_model_path = best_trial.user_attrs.get("best_model_path")
if best_model_path:
    best_model.load_state_dict(torch.load(best_model_path))
criterion = nn.SmoothL1Loss()
smape_loss_fn = SMAPELoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])

for epoch in tqdm(range(RETRAIN_EPOCHS), desc="Training best model"):
    best_model.train()
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels) + smape_loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(best_model.state_dict(), "best_model.pth")

predict_and_submit(
    best_model,
    combined_df,
    scalers,
    features,
    target_col,
    sample_submission_df,
    submission_date_map,
    submission_to_date_map,
    test_indices,
    SEQUENCE_LENGTH,
    PREDICT_LENGTH,
)
