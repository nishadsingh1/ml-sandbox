class SplitRule:
    """
    Represents a rule that determines the direction a datapoint goes when
    arriving at a TreeNode within a DecisionTree.
    """

    def __init__(self, feature_index, threshold, is_categorical=False):
        """
        Parameters
        ----------
        feature_index : int
            The index of the feature corresponding to this rule.
        threshold : float
            The threshold value for the rule's split.
        is_categorical : bool
            True iff this split rule applies to a categorical feature.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_categorical = is_categorical

    def split_right(self, datapoint):
        """
        Returns
        -------
        bool
            Returns true if the datapoint should split right according to this split rule.
        """

        # If the feature is categorical, datapoint belongs in right child iff its feature value
        # matches the threshold value exactly.
        if self.is_categorical:
            return datapoint[self.feature_index] == self.threshold
        else:
            # If the feature is real-valued, datapoint belongs in right child iff its
            # feature value > threshold value.
            return datapoint[self.feature_index] > self.threshold
