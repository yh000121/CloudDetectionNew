import torch
import torch.nn as nn
import torch.nn.functional as F


class ObliviousDecisionTree(nn.Module):
    """
    Oblivious Decision Tree Module.

    This module implements a soft decision tree:
      - Computes decision probabilities via a linear layer + sigmoid.
      - Generates binary codes for leaves and computes reach probabilities.
      - Outputs a weighted sum of leaf values.

    Args:
        input_dim (int): Dimension of input features.
        tree_depth (int): Depth of each decision tree.
    """

    def __init__(self, input_dim, tree_depth):
        super(ObliviousDecisionTree, self).__init__()
        self.tree_depth = tree_depth
        self.num_leaves = 2 ** tree_depth
        self.decision_fn = nn.Linear(input_dim, tree_depth)
        self.leaf_values = nn.Parameter(torch.randn(self.num_leaves))

    def forward(self, x):
        # Compute soft decisions (probabilities between 0 and 1)
        decisions = torch.sigmoid(self.decision_fn(x))  # shape: (batch, depth)
        batch_size, _ = decisions.shape
        device = x.device

        # Generate binary codes for leaves
        leaf_codes = torch.tensor(
            [[int(b) for b in format(i, f'0{self.tree_depth}b')] for i in range(self.num_leaves)],
            dtype=torch.float32, device=device
        )  # shape: (num_leaves, depth)

        # Vectorize probability computation for all leaves
        d_exp = decisions.unsqueeze(1).expand(batch_size, self.num_leaves, self.tree_depth)
        c_exp = leaf_codes.unsqueeze(0).expand(batch_size, self.num_leaves, self.tree_depth)
        leaf_probs = d_exp * c_exp + (1 - d_exp) * (1 - c_exp)
        leaf_probs = leaf_probs.prod(dim=2)  # shape: (batch, num_leaves)

        # Weighted sum of leaf outputs
        out = (leaf_probs * self.leaf_values).sum(dim=1, keepdim=True)  # shape: (batch,1)
        return out


class NODEBlock(nn.Module):
    """
    NODEBlock Module.

    Aggregates multiple ObliviousDecisionTrees, applies attention gating,
    normalizes, applies dropout, and projects back to input dimension.

    Args:
        input_dim (int): Dimension of input features.
        num_trees (int): Number of trees in the block.
        tree_depth (int): Depth of each tree.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_dim, num_trees, tree_depth, dropout=0.2):
        super(NODEBlock, self).__init__()
        self.num_trees = num_trees
        # Ensemble of soft decision trees
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(input_dim, tree_depth) for _ in range(num_trees)
        ])
        # Attention gating to weight tree outputs dynamically
        self.attn_fc = nn.Linear(num_trees, num_trees)
        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(num_trees)
        self.dropout = nn.Dropout(dropout)
        # Project fused output back to input dimension
        self.fc = nn.Linear(num_trees, input_dim)

    def forward(self, x):
        # Collect outputs from each tree: list of (batch,1)
        tree_outs = [tree(x) for tree in self.trees]
        # Concatenate to (batch, num_trees)
        tree_outs = torch.cat(tree_outs, dim=1)

        # Attention gating: compute weights and fuse
        attn_scores = self.attn_fc(tree_outs)               # (batch, num_trees)
        attn_weights = F.softmax(attn_scores, dim=1)        # normalized weights
        gated = tree_outs * attn_weights                    # element-wise weighting

        # Normalize and regularize
        normed = self.bn(gated)
        dropped = self.dropout(normed)

        # Project back to input dimension
        out = self.fc(dropped)  # (batch, input_dim)
        return out


class NODEModel(nn.Module):
    """
    NODEModel Module.

    Stacks multiple NODEBlocks with residual connections and a final classifier.

    Args:
        input_dim (int): Dimension of input features.
        num_blocks (int): Number of NODEBlocks.
        num_trees (int): Trees per block.
        tree_depth (int): Depth of each tree.
        num_classes (int): Number of classes.
        dropout (float): Dropout probability inside blocks.
    """

    def __init__(self, input_dim, num_blocks, num_trees, tree_depth, num_classes, dropout=0.2):
        super(NODEModel, self).__init__()
        # Stack of NODEBlocks
        self.blocks = nn.ModuleList([
            NODEBlock(input_dim, num_trees, tree_depth, dropout) for _ in range(num_blocks)
        ])
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        # Residual connections over blocks
        for block in self.blocks:
            x = x + block(x)
        # Classifier
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Example: test model shapes
    model = NODEModel(input_dim=3, num_blocks=2, num_trees=4, tree_depth=3, num_classes=3)
    sample = torch.randn(5, 3)
    out = model(sample)
    print("Output shape:", out.shape)  # Expect (5, 3)
