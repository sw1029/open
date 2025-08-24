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
# Default prediction length used if not provided via CLI
PREDICT_LENGTH = 7
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
HIDDEN_SIZE = 128
NUM_LAYERS = 2
PATIENCE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument(
        "--decoder_steps",
        type=int,
        default=PREDICT_LENGTH,
        help="Number of decoder time steps",
    )
    args = parser.parse_args()

    print("PyTorch LSTM-based demand forecasting script started.")

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
    ) = prepare_datasets(SEQUENCE_LENGTH, args.decoder_steps, BATCH_SIZE)

    model = Seq2Seq(
        input_size=len(features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=args.num_heads,
        decoder_steps=args.decoder_steps,
        output_size=1,
    ).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    smape_loss_fn = SMAPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, verbose=True
    )

    best_val_smape = float("inf")
    patience_counter = 0

    epoch_iterator = tqdm(range(NUM_EPOCHS), desc="Training Epochs")
    for epoch in epoch_iterator:
        model.train()
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            decoder_inputs = torch.zeros(
                (labels.size(0), args.decoder_steps, 1), device=DEVICE
            )
            optimizer.zero_grad()
            outputs = model(inputs, decoder_inputs)
            loss = criterion(outputs, labels) + smape_loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, all_preds, all_labels, all_item_ids = 0, [], [], []
        with torch.no_grad():
            for inputs, labels, batch_item_ids in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                decoder_inputs = torch.zeros(
                    (labels.size(0), args.decoder_steps, 1), device=DEVICE
                )
                outputs = model(inputs, decoder_inputs)
                batch_loss = criterion(outputs, labels) + smape_loss_fn(outputs, labels)
                val_loss += batch_loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_item_ids.append(np.repeat(batch_item_ids, args.decoder_steps))

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
        epoch_iterator.set_postfix(val_loss=f"{val_loss:.6f}", val_smape=f"{val_smape:.4f}")

        if val_smape < best_val_smape:
            best_val_smape = val_smape
            patience_counter = 0
            torch.save(model.state_dict(), "best_lstm_model.pth")
            tqdm.write(
                f"Epoch {epoch+1}: Validation SMAPE improved to {val_smape:.4f}. Saving model..."
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"\nEarly stopping triggered after {PATIENCE} epochs of no improvement."
                )
                break

    model.load_state_dict(torch.load("best_lstm_model.pth"))
    predict_and_submit(
        model,
        combined_df,
        scalers,
        features,
        target_col,
        sample_submission_df,
        submission_date_map,
        submission_to_date_map,
        test_indices,
        SEQUENCE_LENGTH,
        args.decoder_steps,
    )


if __name__ == "__main__":
    main()
