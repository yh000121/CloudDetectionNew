import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import NODEModel
from train import train_model
from evaluate import evaluate_model_metrics_from_loader, evaluate_all_images


def main():
    # Configuration and hyperparameters
    input_dim = 3
    num_classes = 3
    num_blocks = 3
    num_trees = 5
    tree_depth = 3
    dropout = 0.2
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 50
    batch_size = 64
    test_size = 0.2
    val_size = 0.1
    patience = 5
    random_state = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths for data and outputs
    base_dir = os.path.join('..', '..')
    datasets_dir = os.path.join(base_dir, 'datasets')
    results_dir = os.path.join(base_dir, 'results')
    model_save_path = os.path.join(results_dir, 'node_model_best.pth')
    csv_path = os.path.join(datasets_dir, 'node_training_data.csv')
    os.makedirs(results_dir, exist_ok=True)

    # Load CSV and sample up to 70k per class
    df = pd.read_csv(csv_path)
    df = df.groupby('label', group_keys=False).apply(
        lambda g: g.sample(n=70000, random_state=random_state) if len(g) >= 70000 else g
    ).reset_index(drop=True)
    X = df[['feature_0', 'feature_1', 'feature_2']].values
    y = df['label'].values

    # Split into train+val and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Split train+val into train and validation sets
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac,
        random_state=random_state, stratify=y_trainval
    )

    # Create DataLoaders
    def make_loader(X_arr, y_arr, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    # Initialize model, criterion, optimizer, and scheduler
    model = NODEModel(input_dim, num_blocks, num_trees, tree_depth, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Train the model
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=num_epochs,
        device=device,
        patience=patience,
        model_save_path=model_save_path
    )

    # Plot training and validation loss curves
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'],   label='Val   Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.show()

    # Plot training and validation accuracy curves
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'],   label='Val   Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'acc_curve.png'))
    plt.show()

    # Evaluate on the test set and generate classification images
    print('\n=== Test Set Evaluation & Classification Images ===')
    evaluate_model_metrics_from_loader(
        model, test_loader, device,
        generate_images=True,
        datasets_dir=datasets_dir
    )

if __name__ == '__main__':
    main()