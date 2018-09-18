# Module file for implementation of ID3 algorithm.

from typing import IO, List, Union, Iterator
from pandas import DataFrame, Series
from pandas.core.groupby import GroupBy

from node import BaseNode, Leaf, Node


class DecisionTree:
    root: BaseNode = None

    def __init__(self, load_from: Union[str, IO, None] = None):
        """
        Initializes empty decision tree.
        :param load_from: Restore previously saved instance from file.
        """
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")

    @staticmethod
    def build_tree(X: DataFrame, y: Series, attributes: List[str], value: object = None) -> BaseNode:
        assert X.shape[0] == y.shape[0] and X.shape[0] > 0

        # Check for leaf: Only one class remains
        class_groups: GroupBy = y.groupby(y)
        if len(class_groups.groups) == 1:
            return Leaf(value, list(class_groups.groups.keys())[0])

        # Check for no attributes left
        popular_class = sorted(class_groups, key=lambda key_value: key_value[1].size)[-1][0]
        if not any(True for _ in attributes):
            return Leaf(value, popular_class)

        # Find lowest entropy attribute
        # TODO sort attributes by entropy
        attribute = attributes[0]
        child_attributes = attributes[1:]

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

        result = {'precision': None,
                  'recall': None,
                  'accuracy': None,
                  'F1': None,
                  'confusion-matrix': None}
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


    def save(self, output: Union[str, IO]) -> None:
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pass
