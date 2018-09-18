# Module file for implementation of ID3 algorithm.
from math import log2
from pickle import dump, load
from typing import IO, List, Union, Iterator
from pandas import DataFrame, Series
from pandas.core.groupby import GroupBy
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

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
    def build_tree(X: DataFrame, y: Series, attributes: List[str], value: object = None) -> BaseNode:
        assert X.shape[0] == y.shape[0] and X.shape[0] > 0

        # Check for leaf: Only one class remains
        class_groups: GroupBy = y.groupby(y)
        classes = list(class_groups.groups.keys())
        if len(class_groups.groups) == 1:
            return Leaf(value, classes[0])

        # Check for no attributes left
        popular_class = sorted(class_groups, key=lambda key_value: key_value[1].size)[-1][0]
        if not any(True for _ in attributes):
            return Leaf(value, popular_class)

        def entropy(set: Series) -> float:
            """
            Measures the uncertainty in set
            """
            probs = map(lambda label: len(list(filter(lambda entry: entry == label, set))) / set.size, classes)
            return -sum(map(lambda prob: prob * log2(prob) if prob > 0 else 0, probs))

        def gain(attr: str) -> float:
            """
            Measures the information gain of an attribute
            """
            subsets = map(lambda val: y[X[attr] == val], set(X[attr]))
            return entropy(y) - sum(map(lambda subset: subset.size / X.size * entropy(subset), subsets))

        # Find highest information gain attribute
        attributes = sorted(attributes, key=gain)
        attribute = attributes[-1]
        child_attributes = attributes[:-1]

        # Create children for all values
        children: List[BaseNode] = []
        for child_value in set(X[attribute]):
            child_mask = X[attribute] == child_value
            child_X = X[child_mask]
            child_y = y[child_mask]
            if child_X.size == 0:
                children.append(Leaf(child_value, popular_class))
            else:
                children.append(DecisionTree.build_tree(child_X, child_y, child_attributes, child_value))

        return Node(value, attribute, children)

    def train(self, X: DataFrame, y: Series, attrs: List[str], prune: bool=False) -> None:
        """
        Uses ID3 to train a decision tree on the supplied data.
        :param X: Training data (to classify)
        :param y: Classes of training data
        :param attrs: Names of attributes in X
        :param prune: Indicates whether the tree should be pruned
        """
        self.root = DecisionTree.build_tree(X, y, attrs)

    def predict(self, instance: DataFrame) -> Iterator[object]:
        """
        Classifies the given instance based on the learned decision tree.
        :param instance: Data instance to classify
        :return: Predicted class of instance
        :raises ValueError if the class is not trained
        """
        for index, row in instance.iterrows():
            current_node = self.root
            while isinstance(current_node, Node):
                current_node = filter(
                    lambda child: child.value == row[current_node.attribute],
                    current_node.children).__next__()

            assert isinstance(current_node, Leaf)
            yield current_node.label

    def test(self, X: DataFrame, y: Series, display: bool=False) -> dict:
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
