import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lstm_utils import (
    DEVICE,
    SMAPELoss,
    Seq2Seq,
    prepare_datasets,
    predict_and_submit,
    smape,
)


SEQUENCE_LENGTH = 14
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 60
HIDDEN_SIZE = 512
NUM_LAYERS = 3
PATIENCE = 10
DEFAULT_PREDICT_LENGTH = 7
DEFAULT_NUM_HEADS = 4

SCHEDULED_SAMPLING_PROB = 0.1


def update_sampling_prob(epoch: int) -> None:
    """Linearly increase scheduled sampling probability with epoch."""
    global SCHEDULED_SAMPLING_PROB
    SCHEDULED_SAMPLING_PROB = min(0.5, 0.1 + 0.02 * epoch)


def get_alpha(epoch: int) -> float:
    """Compute L1 loss weight based on epoch.

    The weight starts at 0.5 for the first 20% of epochs and then
    linearly decays to 0 by the final epoch.
    """
    warmup_epochs = int(0.2 * NUM_EPOCHS)
    if epoch < warmup_epochs:
        return 0.5
    decay_epochs = max(1, NUM_EPOCHS - warmup_epochs)
    progress = (epoch - warmup_epochs) / decay_epochs
    return max(0.0, 0.5 * (1 - progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help="Number of attention heads in the decoder.",
    )
    parser.add_argument(
        "--decoder_steps",
        type=int,
        default=DEFAULT_PREDICT_LENGTH,
        help="Final number of time steps for the decoder to predict.",
    )
    parser.add_argument(
        "--cnn_channels",
        type=int,
        default=None,
        help="Output channels of initial Conv1d block.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size of initial Conv1d block.",
    )
    args = parser.parse_args()
    final_predict_length = args.decoder_steps
    num_heads = args.num_heads
    cnn_channels = args.cnn_channels
    kernel_size = args.kernel_size

    logging.basicConfig(level=logging.INFO)
    print("PyTorch LSTM-based demand forecasting script started.")

    curriculum_lengths = [1, 3, final_predict_length]
    base_stage_epochs = NUM_EPOCHS // len(curriculum_lengths)
    stage_epochs = [base_stage_epochs] * len(curriculum_lengths)
    stage_epochs[-1] += NUM_EPOCHS - sum(stage_epochs)

    (
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
    ) = prepare_datasets(SEQUENCE_LENGTH, curriculum_lengths[0], BATCH_SIZE)

    if cnn_channels is None:
        cnn_channels = len(features)

    model = Seq2Seq(
        input_size=len(features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        num_heads=num_heads,
        decoder_steps=final_predict_length,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        future_feat_dim=len(future_features),
    ).to(DEVICE)
    criterion = nn.SmoothL1Loss(reduction="none")
    smape_loss_fn = SMAPELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, verbose=True
    )

    best_val_smape = float("inf")
    stage_logs = []
    total_epoch = 0

    for stage_idx, (curr_len, epochs) in enumerate(
        zip(curriculum_lengths, stage_epochs), 1
    ):
        if stage_idx > 1:
            (
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
            ) = prepare_datasets(SEQUENCE_LENGTH, curr_len, BATCH_SIZE)
        logging.info(
            "Starting curriculum stage %s with predict_length=%s for %s epochs",
            stage_idx,
            curr_len,
            epochs,
        )
        epoch_iterator = tqdm(
            range(epochs),
            desc=f"Stage {stage_idx} (predict_length={curr_len})",
        )
        stage_best_smape = float("inf")
        patience_counter = 0
        for epoch in epoch_iterator:
            update_sampling_prob(epoch)
            alpha = get_alpha(total_epoch)
            model.train()
            for inputs, labels, batch_item_ids, future_feats in train_loader:
                inputs, labels, future_feats = inputs.to(DEVICE), labels.to(DEVICE), future_feats.to(DEVICE)
                weights = (
                    torch.tensor([item_weights[item] for item in batch_item_ids], dtype=torch.float32)
                    .unsqueeze(1)
                    .to(DEVICE)
                )
                optimizer.zero_grad()
                outputs = model(
                    inputs,
                    curr_len,
                    labels,
                    SCHEDULED_SAMPLING_PROB,
                    future_feats,
                )
                l1_loss = (criterion(outputs, labels) * weights).mean()
                smape_loss = smape_loss_fn(outputs, labels, weights).mean()
                loss = alpha * l1_loss + (1 - alpha) * smape_loss
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
                    outputs = model(inputs, curr_len, future_feats=future_feats)
                    batch_l1 = (criterion(outputs, labels) * weights).mean()
                    batch_smape = smape_loss_fn(outputs, labels, weights).mean()
                    batch_loss = alpha * batch_l1 + (1 - alpha) * batch_smape
                    val_loss += batch_loss.item()
                    all_preds.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_item_ids.append(np.repeat(batch_item_ids, curr_len))

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
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
                    pred_unscaled = scalers[item_id].inverse_transform(
                        all_preds_flat[i].reshape(-1, 1)
                    )
                    label_unscaled = scalers[item_id].inverse_transform(
                        all_labels_flat[i].reshape(-1, 1)
                    )
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
            epoch_iterator.set_postfix(
                val_loss=f"{val_loss:.6f}",
                val_smape=f"{val_smape:.4f}",
                ss_prob=f"{SCHEDULED_SAMPLING_PROB:.2f}"
            )
            logging.info(
                "Stage %s Epoch %s: ss_prob=%.2f val_smape=%.4f",
                stage_idx,
                epoch + 1,
                SCHEDULED_SAMPLING_PROB,
                val_smape,
            )
            total_epoch += 1

            if val_smape < stage_best_smape:
                stage_best_smape = val_smape
                patience_counter = 0
                if stage_idx == len(curriculum_lengths) and val_smape < best_val_smape:
                    best_val_smape = val_smape
                    torch.save(model.state_dict(), "best_lstm_model.pth")
                    tqdm.write(
                        f"Stage {stage_idx} Epoch {epoch+1}: Validation SMAPE improved to {val_smape:.4f}. Saving model..."
                    )
                else:
                    tqdm.write(
                        f"Stage {stage_idx} Epoch {epoch+1}: Validation SMAPE improved to {val_smape:.4f}"
                    )
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info(
                        "\nEarly stopping triggered in stage %s after %s epochs of no improvement.",
                        stage_idx,
                        PATIENCE,
                    )
                    break

        stage_logs.append((curr_len, stage_best_smape))
        logging.info(
            "Stage %s completed with best val_smape=%.4f", stage_idx, stage_best_smape
        )

    for length, smape_val in stage_logs:
        logging.info(
            "Curriculum stage with predict_length=%s achieved best val_smape=%.4f",
            length,
            smape_val,
        )

    model.load_state_dict(torch.load("best_lstm_model.pth"))
    predict_and_submit(
        model,
        combined_df,
        scalers,
        features,
        future_features,
        target_col,
        sample_submission_df,
        submission_date_map,
        submission_to_date_map,
        test_indices,
        SEQUENCE_LENGTH,
        final_predict_length,
    )


if __name__ == "__main__":
    main()
