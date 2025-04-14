import torch
import torch.nn as nn


class ObliviousDecisionTree(nn.Module):
    """
    Oblivious Decision Tree Module.

    This module implements a soft decision tree:
      - It computes decision probabilities using a linear layer followed by a sigmoid.
      - Generates binary codes for each leaf (representing decision paths).
      - Computes the probability of reaching each leaf by taking the product of decisions.
      - Produces output as a weighted sum of leaf values.

    Args:
        input_dim (int): Dimension of the input features.
        tree_depth (int): Depth of the tree (number of decision nodes).
    """

    def __init__(self, input_dim, tree_depth):
        super(ObliviousDecisionTree, self).__init__()
        self.tree_depth = tree_depth
        self.num_leaves = 2 ** tree_depth  # Total number of leaves

        # Linear layer to compute decision scores
        self.decision_fn = nn.Linear(input_dim, tree_depth)
        # Each leaf has a learnable output parameter
        self.leaf_values = nn.Parameter(torch.randn(self.num_leaves))

    def forward(self, x):
        # Compute decision probabilities (values between 0 and 1)
        decisions = torch.sigmoid(self.decision_fn(x))  # shape: (batch_size, tree_depth)
        batch_size = x.size(0)
        device = x.device

        # Generate binary codes for all leaves (e.g., for tree_depth=3, 8 leaves)
        leaf_codes = torch.tensor(
            [[int(b) for b in format(i, f'0{self.tree_depth}b')]
             for i in range(self.num_leaves)],
            dtype=torch.float32, device=device
        )  # shape: (num_leaves, tree_depth)

        # Expand dimensions for vectorized computation
        decisions_expanded = decisions.unsqueeze(1).expand(batch_size, self.num_leaves, self.tree_depth)
        leaf_codes_expanded = leaf_codes.unsqueeze(0).expand(batch_size, self.num_leaves, self.tree_depth)

        # For each decision node, use decision probability or (1 - probability) based on leaf code
        leaf_probabilities = decisions_expanded * leaf_codes_expanded + (1 - decisions_expanded) * (
                    1 - leaf_codes_expanded)
        # Multiply probabilities along the decision path to get leaf reaching probability
        leaf_probabilities = leaf_probabilities.prod(dim=2)  # shape: (batch_size, num_leaves)

        # Compute final output as weighted sum of leaf outputs
        out = (leaf_probabilities * self.leaf_values).sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
        return out


class NODEBlock(nn.Module):
    """
    NODEBlock Module.

    This module aggregates multiple ObliviousDecisionTrees:
      - It collects outputs from several trees.
      - Normalizes the concatenated outputs with BatchNorm.
      - Applies Dropout for regularization.
      - Projects the aggregated output back to the input dimension using a fully connected layer.

    Args:
        input_dim (int): Dimension of input features.
        num_trees (int): Number of decision trees in the block.
        tree_depth (int): Depth of each decision tree.
        dropout (float): Dropout rate.
    """

    def __init__(self, input_dim, num_trees, tree_depth, dropout=0.2):
        super(NODEBlock, self).__init__()
        self.trees = nn.ModuleList([ObliviousDecisionTree(input_dim, tree_depth) for _ in range(num_trees)])
        self.bn = nn.BatchNorm1d(num_trees)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_trees, input_dim)

    def forward(self, x):
        # Get output from each tree; each tree outputs a scalar per sample.
        tree_outputs = [tree(x) for tree in self.trees]  # list of tensors (batch_size, 1)
        # Concatenate outputs along the feature dimension: shape becomes (batch_size, num_trees)
        tree_outputs = torch.cat(tree_outputs, dim=1)
        tree_outputs = self.bn(tree_outputs)
        tree_outputs = self.dropout(tree_outputs)
        # Project back to input dimension
        out = self.fc(tree_outputs)  # shape: (batch_size, input_dim)
        return out


class NODEModel(nn.Module):
    """
    NODEModel Module.

    This module builds the overall NODE model by stacking multiple NODEBlocks using residual connections,
    followed by a final classifier that outputs class logits.

    Args:
        input_dim (int): Dimension of input features.
        num_blocks (int): Number of NODEBlocks to stack.
        num_trees (int): Number of trees in each NODEBlock.
        tree_depth (int): Depth of each decision tree.
        num_classes (int): Number of classes for classification.
        dropout (float): Dropout rate in NODEBlocks.
    """

    def __init__(self, input_dim, num_blocks, num_trees, tree_depth, num_classes, dropout=0.2):
        super(NODEModel, self).__init__()
        self.blocks = nn.ModuleList([NODEBlock(input_dim, num_trees, tree_depth, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Apply stacked NODEBlocks with residual connections
        for block in self.blocks:
            x = x + block(x)
        # Final classification layer
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Example testing of the NODEModel
    model = NODEModel(input_dim=10, num_blocks=3, num_trees=5, tree_depth=3, num_classes=3, dropout=0.2)
    print(model)
    dummy_input = torch.randn(4, 10)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected output shape: (4, 3)
