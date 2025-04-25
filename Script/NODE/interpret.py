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


import numpy as np
import torch


def print_leaf_rule(tree, leaf_idx, feature_names=None):
    """
    将单个叶子节点的决策路径打印成人可读的 if-then 规则。

    Args:
        tree (ObliviousDecisionTree): 一棵软决策树实例。
        leaf_idx (int): 叶子节点索引（0 ~ 2^D-1）。
        feature_names (list of str): 每个特征的名称，长度等于输入维度。
                                     默认 ['PC1','PC2','PC3']。
    """
    D = tree.tree_depth
    # 默认特征名
    if feature_names is None:
        feature_names = [f'PC{i + 1}' for i in range(tree.decision_fn.in_features)]
    # 1) 二进制编码
    code = format(leaf_idx, f'0{D}b')  # e.g. '101'
    bits = [int(c) for c in code]
    # 2) 决策参数
    W = tree.decision_fn.weight.detach().cpu().numpy()  # shape (D, in_dim)
    B = tree.decision_fn.bias.detach().cpu().numpy()  # shape (D,)
    print(f"\n>>> Leaf {leaf_idx} 路径 (code={code}) 规则：")
    for level, bit in enumerate(bits):
        op = '>' if bit == 1 else '≤'
        # 拼接 w·x + b
        terms = " + ".join(f"{W[level, k]:.3f}*{feature_names[k]}"
                           for k in range(W.shape[1]))
        print(f"  Level {level + 1}: ({terms} + ({B[level]:+.3f})) {op} 0")
    val = tree.leaf_values[leaf_idx].item()
    print(f"  => 输出 weight = leaf_values[{leaf_idx}] = {val:.4f}")


def print_top_leaf_rules(model, data_loader, device='cpu', top_k=3, feature_names=None):
    """
    计算所有 Block/Tree 的叶子重要性，并打印每棵树中最重要的前 top_k 条规则。

    Args:
        model (nn.Module): 已训练的 NODEModel。
        data_loader (DataLoader): 用于 compute_leaf_importance 的数据 loader。
        device (str): 'cpu' 或 'cuda'。
        top_k (int): 每棵树打印前 top_k 个重要叶子。
        feature_names (list of str): 特征名称列表，长度 = model 输入维度。
    """
    from interpret import compute_leaf_importance

    model.to(device)
    model.eval()
    # 1) 计算叶子重要性
    leaf_imp = compute_leaf_importance(model, data_loader, device)
    # 2) 遍历每个 Block 和 Tree
    for b_idx, tree_list in leaf_imp.items():
        for t_idx, imp_list in enumerate(tree_list):
            # 找到前 top_k 大的叶子索引
            top_idxs = sorted(range(len(imp_list)),
                              key=lambda i: imp_list[i],
                              reverse=True)[:top_k]
            vals = [imp_list[i] for i in top_idxs]
            print(f"\nBlock {b_idx}, Tree {t_idx} 最重要的 {top_k} 个叶子：")
            for rank, (leaf_i, score) in enumerate(zip(top_idxs, vals), 1):
                print(f"  {rank}. Leaf {leaf_i} (importance={score:.4f})")
                # 打印该叶子的 if-then 规则
                print_leaf_rule(model.blocks[b_idx].trees[t_idx], leaf_i, feature_names)



def print_overall_top_leaf_rules(model, data_loader, device='cpu', top_k=5, feature_names=None):
    """
    1) 计算所有叶子的 importance；
    2) 按 importance 排序，选出前 top_k 条；
    3) 打印它们的 Block/Tree/Leaf 索引和人可读的 if–then 规则。

    Args:
        model (nn.Module): 已训练的 NODEModel。
        data_loader (DataLoader): 用于计算 importance 的数据 loader。
        device (str): 'cpu' 或 'cuda'。
        top_k (int): 要展示的全局最重要叶子数量。
        feature_names (list of str): 特征名列表，默认为 ['PC1','PC2','PC3']。
    """
    model.to(device).eval()
    # 1) 计算叶子重要性
    leaf_imp = compute_leaf_importance(model, data_loader, device)
    # 2) 收集所有 (importance, block, tree, leaf_idx)
    all_leaves = []
    for b_idx, trees in leaf_imp.items():
        for t_idx, imp_list in enumerate(trees):
            for leaf_idx, score in enumerate(imp_list):
                all_leaves.append((score, b_idx, t_idx, leaf_idx))
    # 排序并取前 top_k
    top_leaves = sorted(all_leaves, key=lambda x: x[0], reverse=True)[:top_k]

    print(f"\n=== 全局 Top {top_k} 重要叶子规则 ===")
    for rank, (score, b_idx, t_idx, leaf_idx) in enumerate(top_leaves, 1):
        print(f"\n{rank}. Block {b_idx}, Tree {t_idx}, Leaf {leaf_idx}, importance = {score:.4f}")
        # 3) 打印可读规则
        print_leaf_rule(
            model.blocks[b_idx].trees[t_idx],
            leaf_idx,
            feature_names=feature_names
        )


