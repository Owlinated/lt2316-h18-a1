# Module file for implementation of ID3 algorithm.
from math import log2
from pickle import dump, load
from typing import IO, List, Union, Iterator

from numpy import int64, float64
from pandas import DataFrame, Series
from pandas.core.groupby import GroupBy
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

from helper import pairwise
from matcher import DiscreteMatcher, BaseMatcher, ContinuousMatcher
from node import BaseNode, Leaf, Node


class DecisionTree:
    root: BaseNode = None

    def __init__(self, load_from: Union[IO, None] = None):
        """
        Initializes empty decision tree.
        :param load_from: Restore previously saved instance from file.
        """
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.root = load(load_from)

    @staticmethod
    def __build_tree(X: DataFrame, y: Series, attributes: List[str], matcher: BaseMatcher = None) -> BaseNode:
        """
        Recursively builds a tree by greedily creating nodes based on the attribute with the highest information gain.
        :param X: The remaining values
        :param y: Classes corresponding to X
        :param attributes: The remaining attributes to choose from
        :param matcher: Matcher that checks the parent nodes attribute
        :return: Root of subtree that classifies values based on the remaining attributes
        """
        assert X.shape[0] == y.shape[0] and X.shape[0] > 0

        # Check for leaf: Only one class remains
        class_groups: GroupBy = y.groupby(y)
        classes = list(class_groups.groups.keys())
        if len(class_groups.groups) == 1:
            return Leaf(matcher, classes[0])

        # Check for no attributes left
        popular_class = sorted(class_groups, key=lambda key_value: key_value[1].size)[-1][0]
        if not any(True for _ in attributes):
            return Leaf(matcher, popular_class)

        def entropy(set: Series) -> float:
            """
            Measures the uncertainty in set
            """
            probs = map(lambda label: len(list(filter(lambda entry: entry == label, set))) / set.size, classes)
            return -sum(map(lambda prob: prob * log2(prob) if prob > 0 else 0, probs))

        def gain(subsets: Iterator[Series]):
            """
            Measures the information gain of splitting into subsets
            """
            return entropy(y) - sum(map(lambda subset: subset.size / X.size * entropy(subset), subsets))

        def discrete_gain(attr: str) -> float:
            """
            Measures the information gain of splitting into all values at an attribute
            """
            return gain(map(lambda val: y[X[attr] == val], set(X[attr])))

        def continuous_gain(attr: str, threshold: float) -> float:
            """
            Measure the information gain of splitting along a threshold at an attribute
            """
            return gain([y[X[attr] <= threshold], y[X[attr] > threshold]])

        def attribute_gain(attr: str) -> float:
            """
            Measures the information gain based on the type of attribute
            """
            attr_type = X.dtypes[attr]
            if attr_type == int64:
                # Find highest gain achievable by splitting at all values
                return discrete_gain(attr)
            if attr_type == float64:
                # Find highest gain achievable with any threshold
                thresholds = map(lambda value: 0.5 * (value[0] + value[1]), pairwise(sorted(set(X[attr]))))
                return max(map(lambda threshold: continuous_gain(attr, threshold), thresholds))
            raise ValueError("Unknown type of attribute: {}".format(attr_type))

        # Find highest information gain attribute
        attributes = sorted(attributes, key=attribute_gain)
        attribute = attributes[-1]
        child_attributes = attributes[:-1]

        def generate_matchers() -> Iterator[BaseMatcher]:
            """
            Generates matchers for the chosen attribute
            """
            if X.dtypes[attribute] == int64:
                # Create matchers for all values in discrete case
                for child_value in set(X[attribute]):
                    yield DiscreteMatcher(child_value)
            else:
                # Find threshold with highest information gain
                thresholds = map(lambda value: 0.5 * (value[0] + value[1]), pairwise(sorted(set(X[attribute]))))
                threshold = sorted(thresholds, key=lambda threshold: continuous_gain(attribute, threshold))[-1]

                # Generate matchers for <= and > than threshold
                for threshold_direction in [False, True]:
                    yield ContinuousMatcher(threshold, threshold_direction)

        def match_to_node(matcher: BaseMatcher) -> BaseNode:
            """
            Converts a matcher into a node
            """
            child_mask = matcher.is_match(X[attribute])
            child_X = X[child_mask]
            child_y = y[child_mask]
            if child_X.size == 0:
                return Leaf(matcher, popular_class)
            else:
                return DecisionTree.__build_tree(child_X, child_y, child_attributes, matcher)

        return Node(matcher, attribute, list(map(match_to_node, generate_matchers())), popular_class)

    def train(self, X: DataFrame, y: Series, attrs: List[str], prune: bool = False) -> None:
        """
        Uses ID3 to train a decision tree on the supplied data.
        :param X: Training data
        :param y: Classes corresponding to X
        :param attrs: Names of attributes in X
        :param prune: Indicates whether the tree should be pruned
        """
        self.root = DecisionTree.__build_tree(X, y, attrs)
        if prune:
            self.prune(X, y)

    def prune(self, X: DataFrame, y: Series) -> None:
        """
        Prunes nodes, than replace pruned nodes with leafs
        :param X: Set for testing accuracy
        :param y: Labels for testing accuracy
        """
        self.root.mark_pruned(lambda: accuracy_score(y, Series(self.predict(X))))
        self.root = self.root.get_pruned()

    def predict(self, instance: DataFrame) -> Iterator[object]:
        """
        Classifies the given instance based on the learned decision tree.
        :param instance: Data instance to classify
        :return: Predicted class of instance
        :raises ValueError if the class is not trained
        """
        for index, row in instance.iterrows():
            current_node = self.root
            while not current_node.is_leaf():
                next_node = next(filter(
                    lambda child: child.matcher.is_match(row[current_node.attribute]),
                    current_node.children), None)

                # Could not find value, break loop early
                if next_node is None:
                    break

                current_node = next_node

            yield current_node.get_label()

    def test(self, X: DataFrame, y: Series, display: bool = False) -> dict:
        """
        Runs statistical tests on data,
        :param X: Test data
        :param y: Classes of test data
        :param display: Indicates whether the results should be printed
        :return: Dictionary of statistical test results
        """
        predictions = Series(self.predict(X))

        result = {'precision': precision_score(y, predictions, average=None),
                  'recall': recall_score(y, predictions, average=None),
                  'accuracy': accuracy_score(y, predictions),
                  'F1': f1_score(y, predictions, average=None),
                  'confusion-matrix': confusion_matrix(y, predictions)}
        if display:
            print(result)
        return result

    def __str__(self):
        """
        Creates human readable representation of the decision tree,
        :return: Readable string representation or "ID3 untrained" if the model is not trained.
        """
        if self.root is None:
            return "ID3 untrained"
        else:
            return self.root.to_string()

    def save(self, output: IO) -> None:
        """
        Dumps decision tree to a file. This file may later be loaded as a new tree.
        :param output: File to write decision tree into
        """
        dump(self.root, output)
