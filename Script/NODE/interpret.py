import numpy as np
import torch
import shap
from tqdm import tqdm


def compute_leaf_importance(model, data_loader, device='cpu'):
    """
    Compute importance of each leaf in every tree of the NODE model.
    Importance is measured by the average probability of reaching each leaf
    multiplied by the leaf's output weight, aggregated across all data.

    Args:
        model (nn.Module): Trained NODEModel instance.
        data_loader (DataLoader): DataLoader yielding input features (batch, dim).
        device (str): Device for computation.

    Returns:
        dict: Nested mapping block->tree->leaf importance array.
    """
    model.to(device)
    model.eval()

    # Prepare structure to accumulate leaf probabilities
    importance = {}
    # We assume model.blocks is a list of NODEBlock
    for b_idx, block in enumerate(model.blocks):
        importance[b_idx] = []
        for t_idx in range(block.num_trees):
            num_leaves = block.trees[t_idx].num_leaves
            importance[b_idx].append(np.zeros(num_leaves))

    total_samples = 0
    # Iterate over dataset to accumulate leaf_probs
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="Leaf importance batches"):
            inputs = inputs.to(device)
            # Forward through each block's trees
            for b_idx, block in enumerate(model.blocks):
                # Compute raw decisions for each tree in this block
                # We need to extract leaf_probs per tree
                for t_idx, tree in enumerate(block.trees):
                    decisions = torch.sigmoid(tree.decision_fn(inputs))  # (batch, depth)
                    # compute leaf_probs same as in forward
                    depth = tree.tree_depth
                    batch_size = decisions.size(0)
                    device = decisions.device
                    # create codes
                    codes = torch.tensor(
                        [[int(b) for b in format(i, f'0{depth}b')] for i in range(tree.num_leaves)],
                        dtype=torch.float32, device=device
                    )
                    d_exp = decisions.unsqueeze(1).expand(batch_size, tree.num_leaves, depth)
                    c_exp = codes.unsqueeze(0).expand(batch_size, tree.num_leaves, depth)
                    leaf_probs = d_exp * c_exp + (1 - d_exp) * (1 - c_exp)
                    leaf_probs = leaf_probs.prod(dim=2).cpu().numpy()  # (batch, num_leaves)
                    # weighted by leaf output
                    leaf_vals = tree.leaf_values.detach().cpu().numpy()  # (num_leaves,)
                    # accumulate
                    importance[b_idx][t_idx] += (leaf_probs * leaf_vals).sum(axis=0)
            total_samples += inputs.size(0)

    # Normalize by total samples
    for b_idx in importance:
        for t_idx in range(len(importance[b_idx])):
            importance[b_idx][t_idx] /= total_samples
            importance[b_idx][t_idx] = importance[b_idx][t_idx].tolist()

    return importance


def compute_shap_values(model, background_data, test_data, device='cpu', nsamples=100):
    """
    Compute SHAP values for the NODE model using KernelExplainer.

    Args:
        model (nn.Module): Trained NODEModel.
        background_data (np.ndarray): Background dataset for SHAP, shape (M, dim).
        test_data (np.ndarray): Data to explain, shape (N, dim).
        device (str): Device.
        nsamples (int): Number of samples for approximation.

    Returns:
        list of np.ndarray: SHAP values for each class.
    """
    # Wrap model.predict_proba style function
    def f(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        return torch.softmax(logits, dim=1).cpu().detach().numpy()

    # Initialize KernelExplainer with background data
    explainer = shap.KernelExplainer(f, background_data)
    # Compute SHAP values for test data
    shap_values = explainer.shap_values(test_data, nsamples=nsamples)
    return shap_values
