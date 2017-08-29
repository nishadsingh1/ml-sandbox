import random
import numpy as np
from model import Model
from decision_tree import DecisionTree

DEFAULT_FRACTION_FEATURES = 0.8
DEFAULT_FRACTION_DATA = 0.9
DEFAULT_NUM_TREES = 5
DEFAULT_MAX_DEPTH = 15
DEFAULT_MAX_NUM_THRESHOLDS = 5


class RandomForest(Model):
    """
    Implements a Random Forest model. Uses feature and data bagging and an argmax over the sum of constituent
    DecisionTree belief distributions for prediction.
    """

    def __init__(self, max_depth = DEFAULT_MAX_DEPTH, num_trees = DEFAULT_NUM_TREES, \
                    fraction_features = DEFAULT_FRACTION_FEATURES, fraction_data = DEFAULT_FRACTION_DATA, \
                    cat_feature_indices = set(), max_num_thresholds = DEFAULT_MAX_NUM_THRESHOLDS):
        """
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of any leaf node in each DecisionTree within this RandomForest.
        num_trees : int, optional
            The number of DecisionTrees to exist in this RandomForest.
        fraction_features : float, optional
            The fraction of features to permit each DecisionTree to consider for splitting. The actual
            features will be sampled uniformly at random without replacement for each DecisionTree.
        fraction_data : float, optional
            The fraction of datapoints to supply each DecisionTree for training. The actual datapoints
            will be sampled uniformly at random with replacement for each DecisionTree.
        cat_feature_indices : set(int), optional
            The set of indices of features that are categorical (as opposed to numerical or real-valued).
        max_num_thresholds : int, optional
            The maximum number of possible thresholds to consider splitting on for any given feature.
        """
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.fraction_features = fraction_features
        self.fraction_data = fraction_data
        self.cat_feature_indices = cat_feature_indices
        self.max_num_thresholds = max_num_thresholds
        self.trees = []

    def train(self, data, labels):
        """
        Trains the Random Forest by creating and training its constituent trees.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.
        labels : np.array
            An (n,) numpy array. Each entry represents the label for its corresponding datapoint.
        """
        n, f = data.shape
        num_features = int(self.fraction_features * f)
        num_datapoints = int(self.fraction_data * n)
        k = np.max(labels) + 1
        for i in range(self.num_trees):
            subset_of_data_indices = np.random.choice(
                n, num_datapoints, replace=True)
            data_for_tree, labels_for_tree = data[
                subset_of_data_indices], labels[subset_of_data_indices]

            ignore_feature_indices = set(
                np.random.choice(f, f - num_features, replace=False))
            tree = DecisionTree(
                max_depth=self.max_depth,
                ignore_feature_indices=ignore_feature_indices,
                cat_feature_indices=self.cat_feature_indices,
                max_num_thresholds=self.max_num_thresholds)
            tree.train(data_for_tree, labels_for_tree, None, k)
            self.trees.append(tree)

    def predict(self, data):
        """
        Generates predictions for each input datapoint by argmaxing over the sum of the belief
        distributions over the true label reported by each of the Random Forests' trees.

        TODO: Parameterize the aggregation function and/or allow for feedback for tree weighting

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.

        Returns
        -------
        np.array:
            An (n, 1) numpy array. Each entry represents the predicted classification for the corresponding
            datapoint.
        """
        summed_distributions = None
        for tree in self.trees:
            distribution = tree.predict_distribution(data)
            if summed_distributions is None:
                summed_distributions = distribution
            else:
                summed_distributions = np.add(summed_distributions,
                                              distribution)
        argmaxed = np.argmax(summed_distributions, axis=1)
        return np.argmax(summed_distributions, axis=1).reshape((len(data), 1))
