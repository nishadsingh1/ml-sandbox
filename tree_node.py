import numpy as np

class TreeNode:
    """
    Represents a node in a DecisionTree.
    """

    def __init__(self, left, right, split_rule, label, distribution=None):
        """
        Represents a node in a DecisionTree.

        Parameters
        ----------
        left : TreeNode
            The left child of this node.
        right : TreeNode
            Ther ight child of this node.
        split_rule : SplitRule
            The SplitRule that corresponds to this node.
        label : int
            If not none, the Node is a leaf node. If datapoints arrive at this
            node at inference time, assign them this label.
            This should be set to argmax_{label_i}(self.distribution[label_i])
        distribution : np.array, optional
            A (k, 1) numpy array. The array represents the belief distribution over
            the correct classification of labels that arrive at this node.
            Of the form [p_1, p_2, ...] such that sum(p_i) = 1.
        """
        self.left = left
        self.right = right
        self.split_rule = split_rule
        self.label = label
        self.distribution = {}

    def is_leaf_node(self):
        """
        Returns
        -------
        bool
            Returns true iff this node is a leaf node.
        """
        return self.label is not None

    def convert_to_leaf(self, labels, k):
        """
        Makes input node a leaf node by computing its belief distribution and label.

        Parameters
        ----------
        node : TreeNode
            The node to conver to a leaf
        labels : np.array
            An (n,) numpy array. Each entry represents the label for its corresponding datapoint.
        split_rule : SplitRule
            The SplitRule to use in separating the data and labels.
        k : int
            The number of possible labels.
        """
        assert np.max(labels) <= k

        counts = np.zeros(k)
        max_count = 0.0
        mode_label = None
        for l in labels:
            counts[l] += 1.0
            if counts[l] > max_count:
                max_count = counts[l]
                mode_label = l

        self.distribution = np.multiply(float(1) / len(labels), counts).reshape(
            (1, k))
        self.label = mode_label
