import numpy as np
from math import exp, log
from model import Model
from tree_node import TreeNode
from split_rule import SplitRule

DEFAULT_MAX_DEPTH = 15
DEFAULT_MAX_NUM_THRESHOLDS = 5


class DecisionTree(Model):
    """
    Implements a Decision Tree model in which each node has two children.
    """

    def __init__(self, max_depth = DEFAULT_MAX_DEPTH, ignore_feature_indices = set(), \
                 cat_feature_indices = set(), max_num_thresholds = DEFAULT_MAX_NUM_THRESHOLDS):
        """
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of any leaf node in the DecisionTree.
        ignore_feature_indices : set(int), optional
            The set of indices of features this DecisionTree should not consider splitting on.
        cat_feature_indices : set(int), optional
            The set of indices of features that are categorical (as opposed to numerical or real-valued).
        max_num_thresholds : int, optional
            The maximum number of possible thresholds to consider splitting on for any given feature.
        """
        self.max_depth = max_depth
        self.max_num_thresholds = max_num_thresholds
        self.cat_feature_indices = cat_feature_indices
        self.ignore_feature_indices = ignore_feature_indices
        self.root_node = None
        self.possible_threshold_values = None

    def calculate_information_gain(self, left_label_hist, right_label_hist):
        """
        Calculates and outputs a scalar value representing the information gain
        of the specified split on the data.

        Parameters
        ----------
        left_label_hist : dict
            The label histogram of datapoints on the left side of a split.
        right_label_hist : dict
            The label histogram of datapoints on the right side of a split.
        """

        n_l = float(sum(left_label_hist.values()))
        n_r = float(sum(right_label_hist.values()))

        # Calculate entropies of both sides of the split
        H_l = sum(
            map(lambda value: (value / n_l) * log(value / n_l),
                left_label_hist.values()))
        H_r = sum(
            map(lambda value: (value / n_r) * log(value / n_r),
                right_label_hist.values()))

        # Return the information gain
        return (n_l * H_l + n_r * H_r) / (n_l + n_r)

    def find_best_split_rule(self, data, labels):
        """
        Attempts to finds a maximal information gain split rule for a Node.
        Exhaustively tries lots of different (threshold, feature index) values.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.
        labels : np.array
            An (n,) numpy array. Each entry represents the label for its corresponding datapoint.

        Returns
        -------
        SplitRule
            The rule corresponding to the split that yielded maximal information gain.
        """
        _, num_features = data.shape

        max_info_gain = None
        best_split_rule = None

        valid_features_indices = set(
                range(num_features)) - self.ignore_feature_indices

        # Consider all (candidate feature index, candidate threshold value) combinations for a split
        for feature_index in valid_features_indices:
            for threshold in self.possible_threshold_values[feature_index]:
                is_categorical = feature_index in self.cat_feature_indices
                candidate_split_rule = SplitRule(feature_index, threshold,
                                                 is_categorical)

                left_label_hist, right_label_hist = self.get_histograms(
                    data, labels, candidate_split_rule)
                info_gain = self.calculate_information_gain(left_label_hist,
                    right_label_hist)
                if (max_info_gain is None) or (
                        info_gain > max_info_gain):
                    max_info_gain = info_gain
                    best_split_rule = candidate_split_rule

        return best_split_rule

    def set_possible_threshold_values(self, data):
        """
        Sets the possible threshold values for each feature in the data. Will store in the format:
        {Threshold value i => [all but largest candidate threshold value for feature i], ...}.

        If a given feature is numerical (real-valued), the candidate threshold values are uniformly
        distributed across the range its unique feature values found in the data.

        If it is instead categorical, candidate threshold values are randomly sampled.

        TODO: Both of these cases could be handled more intelligently by considering the distribution
        of feature values and setting threshold values to minimize the expected variance in the
        corresponding buckets' counts.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.
        """
        _, f = data.shape
        self.possible_threshold_values = {}

        for i in range(f):
            possible_threshold_values = []
            feature_is_numerical = i not in self.cat_feature_indices

            # Identify the unique values of this feature present in the data
            unique_threshold_values = set()
            for x in data:
                unique_threshold_values.add(x[i])

            # If there is only one unique value, splitting on this feature will never be useful
            if len(unique_threshold_values) <= 1:
                self.ignore_feature_indices.add(i)
                continue

            # If we can afford to consider all unique values as candidate threshold values
            if len(unique_threshold_values) <= self.max_num_thresholds:
                if feature_is_numerical:
                    unique_threshold_values.remove(
                        max(unique_threshold_values))
                possible_threshold_values = list(unique_threshold_values)

            # If there are more unique values to consider as candidate threshold values than allowed
            else:
                max_thresh, min_thresh = max(unique_threshold_values), min(
                    unique_threshold_values)

                # If the feature is numerical, evenly distribute the new candidate threshold values
                # through feature's range of unique values
                if feature_is_numerical:
                    increment = (max_thresh - min_thresh
                                 ) / float(self.max_num_thresholds)
                    possible_threshold_values = \
                        [min_thresh + j * increment for j in range(int(self.max_num_thresholds))]
                else:
                    # TODO: Should randomly sample here instead
                    possible_threshold_values = list(unique_threshold_values)

            self.possible_threshold_values[i] = possible_threshold_values

    def get_histograms(self, data, labels, split_rule):
        """
        Returns two histograms: one for the right side of the split and one for the left.
        Both are of the form { label_1 => count_1, label_2 => count_2, ... }, where count_i is the number
        of datapoints with label label_i that would appear on the appropriate side of the split.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.
        labels : np.array
            An (n,) numpy array. Each entry represents the label for its corresponding datapoint.
        split_rule : SplitRule
            The SplitRule used to determine whether datapoints go right or left.

        Returns
        -------
        tuple(left_label_hist, right_label_hist)
            left_label_hist : dict
                Represents the label histogram of datapoints that would split left with the given rule.
            right_label_hist : dict
                Represents the label histogram of datapoints that would split right with the given rule.
        """

        n, _ = data.shape
        assert (labels.shape[0] == n)

        left_label_hist, right_label_hist = {}, {}
        for i in range(n):
            label = labels[i]
            datapoint = data[i]

            if split_rule.split_right(datapoint):
                corresponding_dictionary = right_label_hist
            else:
                corresponding_dictionary = left_label_hist

            if label not in corresponding_dictionary:
                corresponding_dictionary[label] = 0
            corresponding_dictionary[label] += 1

        return left_label_hist, right_label_hist

    def train(self, data, labels, remaining_depth=None, k=None):
        """
        Grows a decision tree by constructing nodes. Greedily sets split rules for each node by maximizing
        information gain.

        Stops growing (either when no training datapoints are left to separate or when max depth is reached)
        and inserts a leaf node.

        Stores the root node of the resulting tree to use as a starting point for classification later.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.
        labels : np.array
            An (n,) numpy array. Each entry represents the label for its corresponding datapoint.
        remaining_depth : int, optional
            The number of children that can be created underneath nodes at this level of recursion.
            Should be None on the first call.
        k : int, optional
            The number of possible labels. If set to None, the first call to this function will assign it.
            TODO: change the assumption that labels only take on values 0, 1, ..., k - 1.
        """

        node = TreeNode(None, None, None, None)

        if remaining_depth is None:
            # Bookkeeping for first call
            if k is None:
                k = np.max(labels) + 1
            remaining_depth = self.max_depth
            self.root_node = node
            self.set_possible_threshold_values(data)

        if remaining_depth > 0:
            # Non-leaf case
            node.split_rule = self.find_best_split_rule(data, labels)

            left_data, left_labels, right_data, right_labels = self.split_data_and_labels(
                data, labels, node.split_rule)

            if len(left_data) == 0 or len(right_data) == 0:
                # If there would be no more data to split in one of the child nodes, make a leaf
                node.convert_to_leaf(labels, k)
            else:
                node.left = self.train(left_data, left_labels, remaining_depth - 1, k)
                node.right = self.train(right_data, right_labels, remaining_depth - 1, k)

        else:
            # If we've reached our max depth, make a the node a leaf
            node.convert_to_leaf(labels, k)
        return node


    def split_data_and_labels(self, data, labels, split_rule):
        """
        Separates datapoints (and corresponding labels) into right and left buckets
        by evaluating them with the split rule.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.
        labels : np.array
            An (n,) numpy array. Each entry represents the label for its corresponding datapoint.
        split_rule : SplitRule
            The SplitRule to be used for separating the data and labels.

        Returns
        -------
        tuple (left_data, left_labels, right_data, right_labels)
            left_data : np.array
                An (l, d) numpy matrix, where l is the number of datapoints that split left.
                Represents the datapoints that will split left.
            left_labels : np.array
                An (l,) numpy array. Represents the labels of the datapoints that will split left.
            right_data : np.array
                An (n - l, d) numpy matrix. Represents the datapoints that will split right.
            left_labels : np.array
                An (n - l,) numpy array. Represents the labels of the datapoints that will split right.
        """
        n, _ = data.shape
        left_indices, right_indices = [], []

        for data_index in range(n):
            if split_rule.split_right(data[data_index]):
                right_indices.append(data_index)
            else:
                left_indices.append(data_index)

        left_data = data[left_indices]
        right_data = data[right_indices]
        left_labels = labels[left_indices]
        right_labels = labels[right_indices]

        return left_data, left_labels, right_data, right_labels

    def predict(self, data):
        """
        Starts at the root node. For each datapoint: traverses the tree, choosing directions
        by evaluating split rules. Sets datapoint classification to the the label of the leaf
        node at which the datapoint arrives.

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
        assert(self.root_node is not None)

        predictions = np.array([])
        for d in data:
            node = self.root_node
            while not node.is_leaf_node():
                if node.split_rule.split_right(d):
                    node = node.right
                else:
                    node = node.left
            predictions = np.append(predictions, node.label)
        return np.asarray(predictions).reshape((len(data), 1))

    def predict_distribution(self, data):
        """
        A Random Forest that contains this Tree may want to know about the belief distribution over the
        correct classification for each datapoint. It may use these distributions (in conjuction with those
        generated by other trees) to generate its own predictions.

        This function returns such a distribution for each datapoint.

        Parameters
        ----------
        data : np.array
            An (n, d) numpy matrix with numerical (float or int) entries. Each row represents a datapoint.

        Returns
        -------
        np.array
            An (n, k) numpy array, where k is the number of possible labels. Each row represents
            this DecisionTree's belief distribution over the correct classification of labels for
            the corresponding datapoint.
        """
        assert(self.root_node is not None)

        dist_pred = None

        for d in data:
            node = self.root_node
            while not node.is_leaf_node():
                if node.split_rule.split_right(d):
                    node = node.right
                else:
                    node = node.left
            if dist_pred is None:
                dist_pred = node.distribution
            else:
                dist_pred = np.row_stack((dist_pred, node.distribution))

        return dist_pred
