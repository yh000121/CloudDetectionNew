import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter

def load_data(datasets_dir):
    """
    Load PCA-transformed data and pixel indices from the given directory.

    Returns:
        pca_data (np.ndarray): Data array with shape (num_images, height, width, 3).
        pixel_indices (list): List of pixel indices (tuples).
    """
    pca_data_path = os.path.join(datasets_dir, "pca_transformed_data.npy")
    pixel_index_path = os.path.join(datasets_dir, "pixel_index.npy")

    pca_data = np.load(pca_data_path)
    pixel_indices = np.load(pixel_index_path, allow_pickle=True).tolist()
    return pca_data, pixel_indices


def perform_kmeans(pca_data, n_clusters=3):
    """
    Flatten PCA data from multiple images and perform K-means clustering.

    Returns:
        cluster_labels (np.ndarray): Cluster labels for each pixel (flattened).
        image_shapes (list): List of (height, width) for each image.
    """
    num_images = pca_data.shape[0]
    image_shapes = []
    flattened_features = []

    for i in range(num_images):
        img = pca_data[i]
        h, w, _ = img.shape
        image_shapes.append((h, w))
        flattened_features.append(img.reshape(-1, 3))

    flattened_features = np.concatenate(flattened_features, axis=0)
    print(f"Running KMeans on {flattened_features.shape[0]} pixels with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flattened_features)
    return cluster_labels, image_shapes


def save_individual_cluster_images(cluster_labels, image_shapes, image_folders, results_dir, mapping):
    """
    Reshape cluster labels back to image dimensions and save as images with semantic coloring.
    mapping: dict from cluster label to class index
    """
    kmeans_results_dir = os.path.join(results_dir, "kmeans")
    os.makedirs(kmeans_results_dir, exist_ok=True)

    # Define semantic class names and colormap
    class_names = ['Sea', 'Ice', 'Cloud']
    cmap = ListedColormap(['blue', 'lightblue', 'white'])

    start = 0
    for i, (h, w) in enumerate(image_shapes):
        end = start + h * w
        labels_img = cluster_labels[start:end].reshape(h, w)

        # Map raw cluster labels → semantic class indices
        mapped = np.vectorize(mapping.get)(labels_img)

        # Create figure with colorbar and class labels
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(mapped, cmap=cmap)
        ax.set_title(f"{image_folders[i]} Classification Map")  # 图标题

        # 设置 colorbar 并用 class_names 作为标签
        cbar = fig.colorbar(im, ax=ax, ticks=range(len(class_names)))
        cbar.ax.set_yticklabels(class_names)
        ax.axis('off')

        # Save figure
        filename = os.path.join(kmeans_results_dir, f"{image_folders[i]}_clusters.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved {image_folders[i]}'s clustering result to {filename}.")
        start = end


def evaluate_kmeans(cluster_labels, datasets_dir):
    """
    Evaluate K-means clustering by finding an optimal one-to-one mapping between clusters and manual labels.
    Returns the optimal mapping.
    """
    import os
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score
    from scipy.optimize import linear_sum_assignment

    manual_labels_path = os.path.join(datasets_dir, "all_labels.npy")
    manual_labels = np.load(manual_labels_path, allow_pickle=True)

    valid_mask = (manual_labels != -1)
    valid_manual_labels = manual_labels[valid_mask]
    valid_cluster_labels = cluster_labels[valid_mask]
    total = valid_manual_labels.shape[0]
    if total == 0:
        print("No valid manual labels found (all are -1).")
        return {}

    cm = confusion_matrix(valid_cluster_labels, valid_manual_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    optimal_mapping = {row: col for row, col in zip(row_ind, col_ind)}

    mapped_predicted_labels = np.array([optimal_mapping.get(cl, -1) for cl in valid_cluster_labels])

    accuracy = accuracy_score(valid_manual_labels, mapped_predicted_labels)
    precision = precision_score(valid_manual_labels, mapped_predicted_labels, average='macro')
    f1 = f1_score(valid_manual_labels, mapped_predicted_labels, average='macro')
    new_cm = confusion_matrix(valid_manual_labels, mapped_predicted_labels)
    report = classification_report(valid_manual_labels, mapped_predicted_labels, digits=4)

    print("Optimal cluster to manual label mapping (via Hungarian algorithm):")
    for cl, label in optimal_mapping.items():
        print(f"  Cluster {cl} -> Label {label}")
    print(f"Evaluation on {total} labeled pixels:")
    print(f"  Accuracy = {accuracy:.4f}")
    print(f"  Precision (macro-average) = {precision:.4f}")
    print(f"  F1 Score (macro-average) = {f1:.4f}")
    print("Confusion Matrix:")
    print(new_cm)
    print("Classification Report:")
    print(report)

    return optimal_mapping


def main():
    datasets_dir = os.path.join("..", "datasets")
    results_dir = os.path.join("..", "results")

    image_folders_path = os.path.join(datasets_dir, "image_folders.npy")
    if os.path.exists(image_folders_path):
        image_folders = np.load(image_folders_path, allow_pickle=True).tolist()
    else:
        image_folders = ["image_001", "image_002", "image_003", "image_004", "image_005"]
        print("Using default image folder names.")

    pca_data, _ = load_data(datasets_dir)
    cluster_labels, image_shapes = perform_kmeans(pca_data, n_clusters=3)
    optimal_mapping = evaluate_kmeans(cluster_labels, datasets_dir)
    if not optimal_mapping:
        print("Skipping image generation due to lack of mapping.")
    else:
        save_individual_cluster_images(cluster_labels, image_shapes, image_folders, results_dir, optimal_mapping)

    kmeans_results_dir = os.path.join(results_dir, "kmeans")
    os.makedirs(kmeans_results_dir, exist_ok=True)
    cluster_labels_path = os.path.join(kmeans_results_dir, "kmeans_clustered.npy")
    np.save(cluster_labels_path, cluster_labels)
    print(f"KMeans cluster labels saved to {cluster_labels_path}.")


if __name__ == "__main__":
    main()
