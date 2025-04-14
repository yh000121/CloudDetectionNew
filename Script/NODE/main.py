import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from model import NODEModel
from train import train_model

def main():
    # Hyperparameters
    input_dim = 3              # Number of PCA features from CSV
    num_classes = 3            # Number of classes (e.g., clear, ice, cloud)
    num_blocks = 3             # Number of NODEBlocks to stack
    num_trees = 5              # Number of trees per NODEBlock
    tree_depth = 3             # Depth of each decision tree
    dropout = 0.2              # Dropout rate
    learning_rate = 0.001      # Learning rate
    num_epochs = 20            # Number of training epochs
    batch_size = 64            # Batch size
    test_size = 0.2            # Proportion of data used for test set
    random_state = 42          # Random state for reproducibility

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set dataset directory and CSV file path (adjust relative path as needed)
    datasets_dir = os.path.join("..", "..", "datasets")
    csv_path = os.path.join(datasets_dir, "node_training_data.csv")
    print("Loading CSV data from:", csv_path)

    # Load CSV data generated from preprocessing
    df = pd.read_csv(csv_path)

    # Check and extract feature columns and label column
    feature_columns = ['feature_0', 'feature_1', 'feature_2']
    if not set(feature_columns).issubset(df.columns):
        raise ValueError("CSV file does not contain expected feature columns.")

    # Limit each class to at most 70,000 samples by grouping and sampling
    df_sampled = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=70000, random_state=random_state) if len(x) >= 70000 else x
    ).reset_index(drop=True)
    print("After sampling, dataset shape:", df_sampled.shape)
    print("Sample counts per class:\n", df_sampled['label'].value_counts())

    # Extract features and labels from the sampled DataFrame
    X = df_sampled[feature_columns].values   # Shape: (total_samples, 3)
    y = df_sampled['label'].values             # Shape: (total_samples,)

    # Split data into training and test sets (stratify to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

    # Create TensorDataset and DataLoader for training and testing
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the NODE model (input_dim is 3 because of PCA features)
    model = NODEModel(input_dim, num_blocks, num_trees, tree_depth, num_classes, dropout)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model using the training DataLoader
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Save the trained model's state_dict in the results folder
    results_dir = os.path.join("..", "..", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_save_path = os.path.join(results_dir, "node_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to:", model_save_path)

    # Evaluate the model on the test set using a DataLoader-based evaluation function
    from evaluate import evaluate_model_metrics_from_loader
    print("\nEvaluating on the test set:")
    evaluate_model_metrics_from_loader(model, test_loader, device=device)

if __name__ == "__main__":
    main()
