import numpy as np
from math import exp, log
from model import Model

DEFAULT_MAX_HEIGHT = 15
MAX_NUM_THRESHOLDS = 5


class DecisionTree(Model):

    def __init__(self, max_height = DEFAULT_MAX_HEIGHT, feature_indices = None, \
                 cat_feature_indices = [], max_num_thresholds = MAX_NUM_THRESHOLDS):
        self.max_height = max_height
        self.root_node = None
        self.feature_indices = feature_indices  # The subset of features to consider (only assigned when used in a Forest)
        self.possible_threshold_values = None  # {threshold value 1 => [all but largest candidate threshold value for feature 1], ...}
        self.max_num_thresholds = max_num_thresholds
        self.cat_feature_indices = cat_feature_indices  # Indices of features that are categorical (as opposed to numerical)
        self.ignore_feature_indices = set()

    def impurity(self, left_label_hist, right_label_hist):
        """
        Calculates and outputs a scalar value representing the impurity
        (negative information gain) of the specified split on the data.
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

        # Return the impurity
        return -1 * (n_l * H_l + n_r * H_r) / (n_l + n_r)

    def find_best_split_rule(self, data, labels):
        """
        Attempts to finds a minimal inpurity split rule for a Node.
        Exhaustively tries lots of different (threshold, feature index) values.
        """
        _, num_features = data.shape

        min_impurity_value = None
        best_split_rule = None

        if self.feature_indices is None:
            # Either consider all features as candidates for the split rule
            valid_features_indices = set(
                range(num_features)) - self.ignore_feature_indices
        else:
            # Or only the subset that's permitted by the owning Random Forest
            valid_features_indices = self.feature_indices - self.ignore_feature_indices

        # Consider all (candidate feature index, candidate threshold value) combinations for a split
        # and return the one with the lowest impurity
        for feature_index in valid_features_indices:
            for threshold in self.possible_threshold_values[feature_index]:
                is_categorical = feature_index in self.cat_feature_indices
                candidate_split_rule = SplitRule(feature_index, threshold,
                                                 is_categorical)

                left_label_hist, right_label_hist = self.get_histograms(
                    data, labels, candidate_split_rule)
                impurity_value = self.impurity(left_label_hist,
                                               right_label_hist)
                if (min_impurity_value is None) or (
                        impurity_value < min_impurity_value):
                    min_impurity_value = impurity_value
                    best_split_rule = candidate_split_rule

        return best_split_rule

    def set_possible_threshold_values(self, data):
        """
        Sets the possible threshold values for each feature in the data.

        If a given feature is numerical (real-valued), the possible threshold values are uniformly
        distributed across the range its unique feature values found in the data.

        If it is instead categorical, randomly sample `max_num_thresholds` values.

        TODO: Both of these cases could be handled more intelligently by considering the distribution
        of feature values and setting threshold values to minimize the expected variance in the
        corresponding buckets' counts.
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
                    # Should randomly sample here instead
                    possible_threshold_values = list(unique_threshold_values)

            self.possible_threshold_values[i] = possible_threshold_values

    def get_histograms(self, data, labels, split_rule):
        """
        Returns two histograms: one for the right side of the split and one for the left.
        Both are of the form { label_1 => count_1, label_2 => count_2, ... }.
        count_i is the number of datapoints with label label_i that appear on the left/right side of the split.

        The split is specified by the input feature index and threshold value.
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

    def train(self, data, labels, max_label=None, remaining_height=None):
        """
        Grows a decision tree by constructing nodes. Greedily sets split rules for each node by minimizing
        impurity.

        Stops growing (either when no training datapoints are left to separate or when max height is reached)
        and inserts a leaf node.

        Stores the root node of the resulting tree to use as a starting point for classification later.
        """

        if remaining_height is None:
            remaining_height = self.max_height

        node = TreeNode(None, None, None, None)

        if self.root_node is None:  # Some bookkeeping for the first pass
            self.root_node = node
            self.set_possible_threshold_values(data)

        if remaining_height > 0:
            # Non-leaf case
            node.split_rule = self.find_best_split_rule(data, labels)

            left_data, left_labels, right_data, right_labels = self.split_data_and_labels(
                data, labels, node.split_rule)

            if len(left_data) == 0 or len(right_data) == 0:
                # If there would be no more data to split in one of the child nodes, make a leaf
                self.make_leaf(node, labels, max_label)
            else:
                node.left = self.train(left_data, left_labels, max_label,
                                       remaining_height - 1)
                node.right = self.train(right_data, right_labels, max_label,
                                        remaining_height - 1)

        else:
            # If we've reached our max height, make a leaf
            self.make_leaf(node, labels, max_label)
        return node

    def make_leaf(self, node, labels, max_label):
        """
        Makes input node a leaf node.
        """

        k = len(labels)
        if max_label is None:
            max_label = np.max(labels)
        distribution_length = max_label + 1
        counts = np.zeros(distribution_length)

        max_count = 0.0
        mode_label = None
        for l in labels:
            counts[l] += 1.0
            if counts[l] > max_count:
                max_count = counts[l]
                mode_label = l
        node.distribution = np.multiply(float(1) / k, counts).reshape(
            (1, distribution_length))
        node.label = mode_label

    def split_data_and_labels(self, data, labels, split_rule):
        """
        Separates datapoints (and corresponding labels) into right and left buckets
        by evaluating them with the split rule.
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
        """
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

    def predict_and_print(self, d):
        """
        Prints out the path of a datapoint as it traverses the Decision Tree and
        returns its classification label.
        """
        node = self.root_node
        while not node.is_leaf_node():
            split_rule = node.split_rule
            feature_index = split_rule.feature_index
            threshold = split_rule.threshold

            if split_rule.split_right(d):
                operator = " = " if split_rule.is_categorical else " > "
                node = node.right
            else:
                operator = " != " if split_rule.is_categorical else " <= "
                node = node.left
            print(str("Feature #" + str(feature_index) + operator +
                      str(threshold)))
        return node.label

    def predict_for_forest(self, data, report_common_splits=False, tree_num=0):
        """
        A Random Forest that contains this Tree may want to know about the beleif distribution over the
        correct classification for each datapoint. It may use these distributions (in conjuction with those
        generated by other trees) to generate its own predictions.

        This function returns such a distribution for each datapoint.

        Optionally prints out the most commmon split for the tree.
        """
        num_pass = 0
        dist_pred = None
        for d in data:
            node = self.root_node
            while not node.is_leaf_node():
                if node.split_rule.split_right(d):
                    if node == self.root_node:
                        num_pass += 1
                    node = node.right
                else:
                    node = node.left
            if dist_pred is None:
                dist_pred = node.distribution
            else:
                dist_pred = np.row_stack((dist_pred, node.distribution))

        if report_common_splits:
            split_rule = self.root_node.split_rule
            passed = num_pass > len(data) - num_pass

            if split_rule.is_categorical:
                operator = "=" if passed else "!="
            else:
                operator = ">" if passed else "<="
            print("Most common split at tree #" + str(tree_num) + ": feature #"
                  + str(split_rule.feature_index) + " " + operator + " " +
                  str(split_rule.threshold))
        return dist_pred


class TreeNode:
    def __init__(self, left, right, split_rule, label):
        self.left = left  # The left child of the current node.
        self.right = right  # The right child of the current node.
        self.split_rule = split_rule  # Of the form [feature index, threshold value]
        self.label = label  # If set, the Node is a leaf node. If datapoints arrive at this node at inference
        # time, assign them this label.
        # This should be set to argmax_{label_i}(self.distribution[label_i])
        self.distribution = {
        }  # Belief distribution over the correct classification of labels that arrive
        # at this node.
        # Of the form {label_1 => p_1, label_2 => p_2, ...} such that sum(p_i) = 1

    def is_leaf_node(self):
        return self.label is not None


class SplitRule:
    def __init__(self, feature_index, threshold, is_categorical=False):
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_categorical = is_categorical  # True iff this node's split rule is on a categorical feature

    def split_right(self, datapoint):
        """
        Returns true if the datapoint should split right according to this split rule
        """

        # If the feature is categorical, datapoint belongs in right child iff its feature value
        # matches the threshold value exactly.
        if self.is_categorical:
            return datapoint[self.feature_index] == self.threshold
        else:
            # If the feature is real-valued, datapoint belongs in right child iff its
            # feature value > threshold value.
            return datapoint[self.feature_index] > self.threshold
