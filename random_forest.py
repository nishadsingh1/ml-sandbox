import random
import numpy as np
from model import Model
from decision_tree import DecisionTree

DEFAULT_FRACTION_FEATURES = 0.8
DEFAULT_FRACTION_DATA = 0.9
DEFAULT_NUM_TREES = 5
DEFAULT_MAX_HEIGHT = 15


class RandomForest(Model):
    def __init__(self, max_height = DEFAULT_MAX_HEIGHT, num_trees = DEFAULT_NUM_TREES, \
                    fraction_features = DEFAULT_FRACTION_FEATURES, fraction_data = DEFAULT_FRACTION_DATA, \
                    cat_feature_indices = []):
        self.max_height = max_height
        self.num_trees = num_trees
        self.fraction_features = float(fraction_features)
        self.fraction_data = float(fraction_data)
        self.trees = []
        self.cat_feature_indices = cat_feature_indices

    def train(self, data, labels):
        """
        Trains the Random Forest by creating and training its constituent trees.
        Makes use of feature and data bagging within each decision tree.
        """
        n, f = data.shape
        num_features = int(self.fraction_features * f)
        num_datapoints = int(self.fraction_data * n)
        max_label = np.max(labels)
        for i in range(self.num_trees):
            subset_of_data_indices = np.random.choice(
                n, num_datapoints, replace=True)
            data_for_tree, labels_for_tree = data[
                subset_of_data_indices], labels[subset_of_data_indices]

            feature_indices = set(
                np.random.choice(f, num_features, replace=False))
            tree = DecisionTree(
                self.max_height,
                feature_indices,
                cat_feature_indices=self.cat_feature_indices)
            tree.train(data_for_tree, labels_for_tree, max_label)
            self.trees.append(tree)

    def predict(self, data):
        """
        Generates predictions for each input datapoint by argmaxing over the sum of the belief
        distributions over the true label reported by each of the Random Forests' trees.

        TODO: Parameterize the aggregation function and/or allow for feedback for tree weighting
        """
        summed_distributions = None
        for tree in self.trees:
            distribution = tree.predict_for_forest(data)
            if summed_distributions is None:
                summed_distributions = distribution
            else:
                summed_distributions = np.add(summed_distributions,
                                              distribution)
        argmaxed = np.argmax(summed_distributions, axis=1)
        return np.argmax(summed_distributions, axis=1).reshape((len(data), 1))
