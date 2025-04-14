import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from model import NODEModel


def evaluate_model_metrics_from_loader(model, data_loader, device='cpu'):
    """
    Evaluate the model on a provided DataLoader (e.g., test set) and print overall accuracy,
    classification report, and confusion matrix.
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, digits=4)
    cm = confusion_matrix(all_targets, all_preds)

    print("Evaluation Accuracy: {:.4f}".format(acc))
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return acc, report, cm


def evaluate_all_images(model, datasets_dir, device='cpu', generate_images=True):
    """
    Evaluate and generate classification result images for all images from PCA-transformed data.
    For each image, classify each pixel. If generate_images is True, save the classification result
    image; if False, only output numerical summaries.
    """
    pca_data_path = os.path.join(datasets_dir, "pca_transformed_data.npy")
    try:
        pca_data = np.load(pca_data_path)
    except FileNotFoundError:
        print("PCA-transformed data not found at:", pca_data_path)
        return

    # Create subfolder for classification images if generating images
    results_dir = os.path.join("..", "..", "results", "NODE")
    if generate_images and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.to(device)
    model.eval()

    num_images = pca_data.shape[0]
    for i in range(num_images):
        image = pca_data[i]  # shape: (height, width, n_components)
        h, w, n_components = image.shape
        # Flatten image to (num_pixels, n_components)
        X = image.reshape(-1, n_components)
        X_tensor = torch.from_numpy(X).float().to(device)

        with torch.no_grad():
            outputs = model(X_tensor)
            _, preds = torch.max(outputs, dim=1)
        pred_image = preds.cpu().numpy().reshape(h, w)

        if generate_images:
            plt.figure(figsize=(10, 8))
            plt.imshow(pred_image, cmap="viridis")
            plt.title(f"Classification Result for Image {i}")
            cbar = plt.colorbar(ticks=[0, 1, 2])
            cbar.set_label("Class Label")
            plt.xlabel("Column")
            plt.ylabel("Row")
            plt.tight_layout()

            image_save_path = os.path.join(results_dir, f"image_{i}_classification.png")
            plt.savefig(image_save_path)
            print(f"Image {i} classification result saved to: {image_save_path}")
            plt.close()
        else:
            unique, counts = np.unique(pred_image, return_counts=True)
            print(f"Image {i} prediction distribution: {dict(zip(unique, counts))}")


if __name__ == "__main__":
    # For standalone testing of evaluate_all_images, reconstruct model and load saved state.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_dir = os.path.join("..", "..", "datasets")
    results_dir = os.path.join("..", "..", "results")

    # Reconstruct the model (must match the training parameters)
    input_dim = 3
    num_classes = 3
    num_blocks = 3
    num_trees = 5
    tree_depth = 3
    dropout = 0.2
    model = NODEModel(input_dim, num_blocks, num_trees, tree_depth, num_classes, dropout)

    model_save_path = os.path.join(results_dir, "node_model.pth")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("Loaded model from:", model_save_path)
    else:
        print("Model file not found at", model_save_path)
        exit(1)

    # For debugging, you can set generate_images to False
    evaluate_all_images(model, datasets_dir, device=device, generate_images=True)
