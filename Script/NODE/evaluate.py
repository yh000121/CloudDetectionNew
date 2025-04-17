import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import NODEModel
from metrics import compute_metrics
from interpret import compute_leaf_importance, compute_shap_values


def evaluate_model_metrics_from_loader(model, data_loader, device='cpu', generate_images=False, datasets_dir=None):
    """
    Evaluate the model on a provided DataLoader (e.g., test set), print metrics,
    and optionally generate classification images.

    Args:
        model (nn.Module): Trained NODEModel.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        device (str): Compute device.
        generate_images (bool): Whether to generate classification images.
        datasets_dir (str): Directory containing pca_transformed_data.npy if generating images.

    Returns:
        tuple: (all_targets, all_preds)
    """
    model.to(device)
    model.eval()
    all_preds, all_targets = [], []
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

    if generate_images:
        if datasets_dir is None:
            raise ValueError('datasets_dir must be provided when generate_images=True')
        evaluate_all_images(model, datasets_dir, device, generate_images=True)

    return all_targets, all_preds


def evaluate_all_images(model, datasets_dir, device='cpu', generate_images=True):
    """
    Generate classification result images for all PCA-transformed inputs.
    """
    pca_path = os.path.join(datasets_dir, 'pca_transformed_data.npy')
    try:
        pca_data = np.load(pca_path)
    except FileNotFoundError:
        print(f"Missing PCA data at: {pca_path}")
        return

    class_names = ['Ice', 'Cloud', 'Sea']
    results_dir = os.path.join('..', '..', 'results', 'NODE')
    os.makedirs(results_dir, exist_ok=True)

    model.to(device)
    model.eval()
    for idx in range(pca_data.shape[0]):
        img = pca_data[idx]
        h, w, c = img.shape
        X = img.reshape(-1, c)
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, preds = torch.max(outputs, 1)
        pred_img = preds.cpu().numpy().reshape(h, w)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(pred_img, cmap='viridis')
        plt.title(f'Image {idx} Classification')
        cbar = plt.colorbar(im, ticks=list(range(len(class_names))))
        cbar.ax.set_yticklabels(class_names)
        plt.tight_layout()
        out_path = os.path.join(results_dir, f'image_{idx}_classification.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved classification image: {out_path}")


def evaluate_interpretation(model, data_loader, background_data, test_data, device='cpu'):
    """
    Interpret the model via leaf importance and SHAP for up to three classes.
    Saves leaf importance bar chart and SHAP summary plots to results directory.
    """
    # 1. 准备结果目录
    base_results = os.path.join('..', '..', 'results', 'NODE')
    os.makedirs(base_results, exist_ok=True)
    shap_dir = os.path.join(base_results, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    # 2. 计算叶子重要性
    print("Computing leaf importance...")
    leaf_imp = compute_leaf_importance(model, data_loader, device)
    labels, values = [], []
    for b, trees in leaf_imp.items():
        for t, imp in enumerate(trees):
            top5 = sorted(imp, reverse=True)[:5]
            print(f"Block {b}, Tree {t}, Top-5 leaf importances: {top5}")
            labels.append(f"B{b}_T{t}")
            values.append(max(imp))

    # 3. 绘制并保存叶子重要性柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xlabel('Block_Tree')
    plt.ylabel('Max Leaf Importance')
    plt.title('Leaf Importance per Tree')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    leaf_bar_path = os.path.join(base_results, 'leaf_importance_bar.png')
    plt.savefig(leaf_bar_path)
    plt.close()
    print(f"Saved leaf importance bar chart to: {leaf_bar_path}")

    # 4. SHAP 分析（保持原来逻辑）
    print("Computing SHAP values...")
    shap_vals = compute_shap_values(model, background_data, test_data, device)

    class_names = ['Ice', 'Cloud', 'Sea']
    feature_names = [f'feature_{i}' for i in range(test_data.shape[1])]
    df_test = pd.DataFrame(test_data, columns=feature_names)

    for i in range(min(len(shap_vals), len(class_names))):
        class_shap = shap_vals[i]
        n = class_shap.shape[0]
        df_plot = df_test.iloc[:n]
        cls = class_names[i]
        print(f"SHAP summary for class {cls}")
        shap.summary_plot(class_shap, df_plot, feature_names=feature_names, show=False)
        plot_path = os.path.join(shap_dir, f"shap_summary_{cls}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary plot for class {cls} to: {plot_path}")



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.join('..', '..')
    datasets_dir = os.path.join(base_dir, 'datasets')
    results_dir = os.path.join(base_dir, 'results')
    model_path = os.path.join(results_dir, 'node_model_best.pth')
    csv_path = os.path.join(datasets_dir, 'node_training_data.csv')

    # Load and sample data
    df = pd.read_csv(csv_path)
    df = df.groupby('label', group_keys=False).apply(
        lambda g: g.sample(n=70000, random_state=42) if len(g)>=70000 else g
    ).reset_index(drop=True)
    X = df[['feature_0','feature_1','feature_2']].values
    y = df['label'].values

    # Split data
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.125, random_state=42, stratify=y_trval
    )

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    ), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=64, shuffle=False)

    # Load model
    model = NODEModel(input_dim=3, num_blocks=3, num_trees=5, tree_depth=3, num_classes=3)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded model for evaluation.')
    else:
        print('Model file not found.')
        exit(1)

    # Evaluate and generate images
    print('=== Test Evaluation and Images ===')
    evaluate_model_metrics_from_loader(
        model, test_loader, device,
        generate_images=True, datasets_dir=datasets_dir
    )
    # Interpretation
    print('=== Interpretation on Validation ===')
    evaluate_interpretation(model, val_loader, X_train[:100], X_val[:100], device)
