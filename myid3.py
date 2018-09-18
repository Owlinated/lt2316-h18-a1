# Module file for implementation of ID3 algorithm.
from math import log2
from pickle import dump, load
from typing import IO, List, Union, Iterator
from pandas import DataFrame, Series, concat
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
    def __build_tree(X: DataFrame, y: Series, attributes: List[str], value: object = None) -> BaseNode:
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
                children.append(DecisionTree.__build_tree(child_X, child_y, child_attributes, child_value))

        return Node(value, attribute, children, popular_class)

    def train(self, X: DataFrame, y: Series, attrs: List[str], prune: bool = False) -> None:
        """
        Uses ID3 to train a decision tree on the supplied data.
        :param X: Training data (to classify)
        :param y: Classes of training data
        :param attrs: Names of attributes in X
        :param prune: Indicates whether the tree should be pruned
        """
        self.root = DecisionTree.__build_tree(X, y, attrs)

    def __prune(self, X: DataFrame, y: Series, node: BaseNode) -> bool:
        """
        Recursively try to prune nodes. If they do not affect accuracy keep the change.
        :param X: Set for testing accuracy
        :param y: Labels for testing accuracy
        :param node: The current node
        :return: Value indicating whether node has been pruned
        """
        if isinstance(node, Leaf):
            return True

        assert isinstance(node, Node)
        if all(map(lambda child: self.__prune(X, y, child), node.children)):
            pre_accuracy = accuracy_score(y, Series(self.predict(X)))
            node.pruned = True
            post_accuracy = accuracy_score(y, Series(self.predict(X)))

            if post_accuracy >= pre_accuracy:
                return True

            node.pruned = False
            return False

    def __rebuild(self, node: BaseNode):
        """
        Replace pruned nodes with leafs
        :param node: The current node
        """
        if isinstance(node, Leaf):
            return

        assert isinstance(node, Node)
        if node == self.root and node.pruned:
            self.root = Leaf(node.value, node.fallback_label)

        pruned_children = list(filter(lambda child: isinstance(child, Node) and child.pruned, node.children))
        node.children = list(filter(lambda child: child not in pruned_children, node.children)) \
            + list(map(lambda child: Leaf(child.value, child.fallback_label), pruned_children))

    def prune(self, X: DataFrame, y: Series) -> None:
        """
        :param X: Set for testing accuracy
        :param y: Labels for testing accuracy
        Prune nodes, than replace pruned nodes with leafs
        """
        self.__prune(X, y, self.root)
        self.__rebuild(self.root)

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
                # Stop early if the node has been pruned
                if current_node.pruned:
                    break

                next_node = next(filter(
                    lambda child: child.value == row[current_node.attribute],
                    current_node.children), None)

                # Could not find value, break loop early
                if next_node is None:
                    break

                current_node = next_node

            if isinstance(current_node, Leaf):
                # Reached leaf, using stored label
                yield current_node.label
            else:
                # Encountered unknown value or pruned node, using fallback label
                assert isinstance(current_node, Node)
                yield current_node.fallback_label

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
