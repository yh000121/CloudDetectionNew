import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss ceases to improve.

    Args:
        patience (int): How many epochs to wait after last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, current_loss):
        """
        Call this at the end of each epoch.

        Args:
            current_loss (float): The current validation loss.

        Returns:
            bool: True if training should stop early, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        # Check if loss improved
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics: accuracy, precision, recall, F1.

    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.

    Returns:
        tuple: (accuracy, precision, recall, f1), all floats.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1
