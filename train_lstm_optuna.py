import optuna
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lstm_utils import (
    DEVICE,
    FUTURE_FEATURES,
    SMAPELoss,
    Seq2Seq,
    prepare_datasets,
    predict_and_submit,
    smape,
)

SEQUENCE_LENGTH = 14
PREDICT_LENGTH = 7
BATCH_SIZE = 64
OPTUNA_EPOCHS = 10
RETRAIN_EPOCHS = 30
PATIENCE = 10

SCHEDULED_SAMPLING_PROB = 0.1


def update_sampling_prob(epoch: int) -> None:
    """Linearly increase scheduled sampling probability with epoch."""
    global SCHEDULED_SAMPLING_PROB
    SCHEDULED_SAMPLING_PROB = min(0.5, 0.1 + 0.02 * epoch)

# Progressive training lengths and their epoch ratios
STAGE_LENGTHS = [1, 3, PREDICT_LENGTH]
STAGE_RATIOS = [1, 1, 1]

# Track the best SMAPE observed during Optuna trials
best_smape = float("inf")


def objective(trial):
    global best_smape
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_heads = trial.suggest_int("num_heads", 1, 8)
    decoder_steps = trial.suggest_int("decoder_steps", PREDICT_LENGTH, 14)

    # Prepare datasets for the first stage to obtain feature information
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
        item_weights,
    ) = prepare_datasets(SEQUENCE_LENGTH, STAGE_LENGTHS[0], BATCH_SIZE)

    model = Seq2Seq(
        input_size=len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        decoder_steps=decoder_steps,
        output_size=1,
        future_feat_dim=len(FUTURE_FEATURES),
    ).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    smape_loss_fn = SMAPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_ratio = sum(STAGE_RATIOS)
    epochs_per_stage = [max(1, OPTUNA_EPOCHS * r // total_ratio) for r in STAGE_RATIOS]
    while sum(epochs_per_stage) < OPTUNA_EPOCHS:
        for i in range(len(epochs_per_stage)):
            if sum(epochs_per_stage) < OPTUNA_EPOCHS:
                epochs_per_stage[i] += 1
            else:
                break

    for stage_idx, curr_len in enumerate(STAGE_LENGTHS):
        if stage_idx > 0:
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
                item_weights,
            ) = prepare_datasets(SEQUENCE_LENGTH, curr_len, BATCH_SIZE)

        for epoch in range(epochs_per_stage[stage_idx]):
            update_sampling_prob(epoch)
            model.train()
            for inputs, labels, batch_item_ids, future_feats in train_loader:
                inputs, labels, future_feats = (
                    inputs.to(DEVICE),
                    labels.to(DEVICE),
                    future_feats.to(DEVICE),
                )
                weights = (
                    torch.tensor([item_weights[item] for item in batch_item_ids], dtype=torch.float32)
                    .unsqueeze(1)
                    .to(DEVICE)
                )
                optimizer.zero_grad()
                outputs = model(
                    inputs,
                    future_feats,
                    curr_len,
                    labels,
                    SCHEDULED_SAMPLING_PROB,
                )
                loss = (
                    criterion(outputs, labels) * weights
                    + smape_loss_fn(outputs, labels, weights)
                ).mean()
                loss.backward()
                optimizer.step()

    model.eval()
    val_loss, all_preds, all_labels, all_item_ids = 0, [], [], []
    with torch.no_grad():
        for inputs, labels, batch_item_ids, future_feats in val_loader:
            inputs, labels, future_feats = (
                inputs.to(DEVICE),
                labels.to(DEVICE),
                future_feats.to(DEVICE),
            )
            weights = (
                torch.tensor([item_weights[item] for item in batch_item_ids], dtype=torch.float32)
                .unsqueeze(1)
                .to(DEVICE)
            )
            outputs = model(inputs, future_feats, STAGE_LENGTHS[-1])
            batch_loss = (
                criterion(outputs, labels) * weights
                + smape_loss_fn(outputs, labels, weights)
            ).mean()
            val_loss += batch_loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_item_ids.append(np.repeat(batch_item_ids, STAGE_LENGTHS[-1]))

    val_loss /= len(val_loader)

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
study.optimize(objective, n_trials=50)

best_trial = study.best_trial
best_params = best_trial.params

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
    item_weights,
) = prepare_datasets(SEQUENCE_LENGTH, PREDICT_LENGTH, BATCH_SIZE)

best_model = Seq2Seq(
    input_size=len(features),
    hidden_size=best_params["hidden_size"],
    num_layers=best_params["num_layers"],
    num_heads=best_params["num_heads"],
    decoder_steps=PREDICT_LENGTH,
    output_size=1,
    future_feat_dim=len(FUTURE_FEATURES),
).to(DEVICE)
best_model_path = best_trial.user_attrs.get("best_model_path")
if best_model_path:
    best_model.load_state_dict(torch.load(best_model_path))
criterion = nn.SmoothL1Loss(reduction="none")
smape_loss_fn = SMAPELoss(reduction="none")
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])

best_val_smape = float("inf")
patience_counter = 0

for epoch in tqdm(range(RETRAIN_EPOCHS), desc="Curriculum training"):
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
        item_weights,
    ) = prepare_datasets(SEQUENCE_LENGTH, PREDICT_LENGTH, BATCH_SIZE)

    update_sampling_prob(epoch)
    best_model.train()
    for inputs, labels, batch_item_ids, future_feats in train_loader:
        inputs, labels, future_feats = (
            inputs.to(DEVICE),
            labels.to(DEVICE),
            future_feats.to(DEVICE),
        )
        weights = (
            torch.tensor([item_weights[item] for item in batch_item_ids], dtype=torch.float32)
            .unsqueeze(1)
            .to(DEVICE)
        )
        optimizer.zero_grad()
        outputs = best_model(
            inputs,
            future_feats,
            PREDICT_LENGTH,
            labels,
            SCHEDULED_SAMPLING_PROB,
        )
        l1_loss = criterion(outputs, labels) * weights
        smape_loss = smape_loss_fn(outputs, labels, weights)
        loss = (l1_loss + smape_loss).mean()
        loss.backward()
        optimizer.step()

    best_model.eval()
    val_loss, all_preds, all_labels, all_item_ids = 0, [], [], []
    with torch.no_grad():
        for inputs, labels, batch_item_ids, future_feats in val_loader:
            inputs, labels, future_feats = (
                inputs.to(DEVICE),
                labels.to(DEVICE),
                future_feats.to(DEVICE),
            )
            weights = (
                torch.tensor([item_weights[item] for item in batch_item_ids], dtype=torch.float32)
                .unsqueeze(1)
                .to(DEVICE)
            )
            outputs = best_model(inputs, future_feats, PREDICT_LENGTH)
            batch_loss = (
                criterion(outputs, labels) * weights
                + smape_loss_fn(outputs, labels, weights)
            ).mean()
            val_loss += batch_loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_item_ids.append(np.repeat(batch_item_ids, PREDICT_LENGTH))

    val_loss /= len(val_loader)

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

    val_smape = smape(all_labels_unscaled, all_preds_unscaled)

    if val_smape < best_val_smape:
        best_val_smape = val_smape
        patience_counter = 0
        torch.save(best_model.state_dict(), "best_lstm_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break


(
    _,
    _,
    scalers,
    combined_df,
    features,
    target_col,
    sample_submission_df,
    submission_date_map,
    submission_to_date_map,
    test_indices,
    _,
) = prepare_datasets(SEQUENCE_LENGTH, PREDICT_LENGTH, BATCH_SIZE)

best_model.load_state_dict(torch.load("best_lstm_model.pth"))
final_predict_length = PREDICT_LENGTH
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
    final_predict_length,
)
