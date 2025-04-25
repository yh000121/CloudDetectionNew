import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import NODEModel
from interpret import compute_leaf_importance, compute_shap_values, print_top_leaf_rules, print_overall_top_leaf_rules
from matplotlib.colors import ListedColormap



def evaluate_metrics(model, data_loader, device='cpu'):
    """
    Evaluate model on data_loader and print accuracy, classification report, confusion matrix.
    """
    model.to(device).eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    return all_targets, all_preds


def generate_classification_images(model, datasets_dir, device='cpu'):
    """
    For each PCA-transformed image, predict per-pixel class and save a colored segmentation map.

    Color mapping:
        0 (Sea)   -> blue
        1 (Ice)   -> lightblue
        2 (Cloud) -> white
    """
    # Load PCA-transformed data (shape: n_images × H × W × 3)
    pca_path = os.path.join(datasets_dir, 'pca_transformed_data.npy')
    pca_data = np.load(pca_path)

    # Define class names and corresponding colors
    class_names = ['Sea', 'Ice', 'Cloud']
    cmap = ListedColormap(['blue', 'lightblue', 'white'])

    # Prepare output directory
    out_dir = os.path.join('..', '..', 'results', 'NODE', 'images')
    os.makedirs(out_dir, exist_ok=True)

    # Set model to evaluation mode and move to device
    model.to(device)
    model.eval()

    # Loop over each image
    for idx in range(pca_data.shape[0]):
        img = pca_data[idx]           # numpy array of shape (H, W, 3)
        H, W, C = img.shape
        # Flatten to (H*W, 3) for batch prediction
        flat = img.reshape(-1, C)
        inputs = torch.from_numpy(flat).float().to(device)

        # Predict class logits and take argmax
        with torch.no_grad():
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()

        # Reshape predictions back to (H, W)
        pred_map = preds.reshape(H, W)

        # Plot segmentation map with custom colormap
        plt.figure(figsize=(6, 6))
        im = plt.imshow(pred_map, cmap=cmap, vmin=0, vmax=2)
        plt.title(f"Image {idx} Classification")
        cbar = plt.colorbar(im, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(class_names)
        plt.axis('off')

        # Save the figure
        save_path = os.path.join(out_dir, f'image_{idx}_pred.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved segmentation map: {save_path}")

def evaluate_interpretation(model, data_loader, background_data, test_data, device='cpu'):


    print("Computing leaf importance...")
    leaf_imp = compute_leaf_importance(model, data_loader, device)
    for b_idx, trees in leaf_imp.items():
        for t_idx, imp in enumerate(trees):
            top5 = sorted(imp, reverse=True)[:5]
            print(f"Block {b_idx}, Tree {t_idx}, top-5 leaf importances: {top5}")


    print("Computing SHAP values...")
    shap_vals = compute_shap_values(model, background_data, test_data, device)

    feature_names = ['PC1','PC2','PC3']
    df_test = pd.DataFrame(test_data, columns=feature_names)
    class_names = ['Ice','Cloud','Sea']
    shap_dir = os.path.join('..','..','results','NODE','shap')
    os.makedirs(shap_dir, exist_ok=True)

    saved_paths = []
    for i, cls in enumerate(class_names):
        if i >= len(shap_vals):
            break
        vals = shap_vals[i]
        df_sub = df_test.iloc[:vals.shape[0]]

        print(f"Saving SHAP beeswarm for {cls} ...")
        shap.summary_plot(
            vals,
            df_sub,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Summary for {cls}")
        outp = os.path.join(shap_dir, f"shap_beeswarm_{cls}.png")
        plt.savefig(outp, bbox_inches='tight')
        plt.close()
        print(f"  -> saved to {outp}")
        saved_paths.append(outp)


    print("Combining individual plots into one big image...")
    imgs = [Image.open(p) for p in saved_paths]
    widths  = [img.width  for img in imgs]
    heights = [img.height for img in imgs]
    max_width = max(widths)
    total_height = sum(heights)
    mosaic = Image.new('RGB', (max_width, total_height), color=(255,255,255))
    y_offset = 0
    for img in imgs:
        mosaic.paste(img, (0, y_offset))
        y_offset += img.height
    big_path = os.path.join(shap_dir, 'shap_beeswarm_all.png')
    mosaic.save(big_path)
    print(f"Saved combined big image to: {big_path}")



if __name__ == "__main__":
    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = os.path.join('..','..')
    datasets_dir = os.path.join(base, 'datasets')
    results_dir = os.path.join(base, 'results')
    os.makedirs(os.path.join(results_dir,'NODE'), exist_ok=True)

    # load best model
    model_path = os.path.join(results_dir, 'node_model_best.pth')
    model = NODEModel(input_dim=3, num_blocks=3, num_trees=5,
                      tree_depth=3, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded model from", model_path)

    # load CSV, sample and split
    df = pd.read_csv(os.path.join(datasets_dir, 'node_training_data.csv'))
    df = df.groupby('label', group_keys=False)\
           .apply(lambda g: g.sample(n=70000, random_state=42)
                                 if len(g)>=70000 else g)\
           .reset_index(drop=True)
    X = df[['feature_0','feature_1','feature_2']].values
    y = df['label'].values
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.125, random_state=42, stratify=y_trval)

    # build loaders
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test,dtype=torch.float32),
        torch.tensor(y_test,dtype=torch.long)
    ), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val,dtype=torch.float32),
        torch.tensor(y_val,dtype=torch.long)
    ), batch_size=64, shuffle=False)

    # 1) evaluate metrics & images
    # print("\n=== Test Evaluation ===")
    # evaluate_metrics(model, test_loader, device)
    # print("\n=== Generate Segmentation Maps ===")
    # generate_classification_images(model, datasets_dir, device)

    # 2) interpret
    # use a small background & test sample for SHAP to speed up
    # bg = X_train[:200]
    # ts = X_val[:200]
    # print("\n=== Interpretation ===")
    # evaluate_interpretation(model, val_loader, bg, ts, device)


    # print_top_leaf_rules(
    #     model,
    #     val_loader,
    #     device=device,
    #     top_k=2,
    #     feature_names=['PC1','PC2','PC3']
    # )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        ),
        batch_size=64, shuffle=False
    )

    print_overall_top_leaf_rules(
        model,
        val_loader,
        device='cuda',
        top_k=5,
        feature_names=['PC1', 'PC2', 'PC3']
    )