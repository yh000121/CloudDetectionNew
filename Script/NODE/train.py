import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=20,
    device='cpu',
    patience=5,
    model_save_path=None
):
    """
    Train the NODE model with validation, learning rate scheduling, and early stopping.

    Args:
        model (nn.Module): The NODE model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion: Loss function.
        optimizer: Optimizer (e.g., Adam).
        scheduler: Learning rate scheduler (e.g., ReduceLROnPlateau).
        num_epochs (int): Maximum number of epochs.
        device (str): Compute device ('cpu' or 'cuda').
        patience (int): Number of epochs with no improvement before stopping.
        model_save_path (str): File path to save the best model.

    Returns:
        dict: Training history containing losses, accuracies, and F1 scores.
    """
    model.to(device)
    best_val_loss = float('inf')
    no_improve = 0

    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(1, num_epochs + 1):
        # ----- Training phase -----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ----- Validation phase -----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_samples += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = val_loss_sum / val_samples
        val_acc = val_correct / val_samples
        val_f1 = f1_score(all_targets, all_preds, average='macro')

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}"
        )

    return history