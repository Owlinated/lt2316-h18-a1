from abc import abstractmethod
from typing import List, Callable

from matcher import BaseMatcher


class BaseNode:
    """
    Base class for nodes, stores function to decide if value matches this node.
    """
    matcher: BaseMatcher = None

    def __init__(self, matcher: BaseMatcher):
        self.matcher = matcher

    @abstractmethod
    def is_leaf(self) -> bool:
        """
        Gets a value indicating whether the current node is a leaf or a pruned node.
        :return:
        """
        pass

    @abstractmethod
    def get_label(self) -> str:
        """
        Gets the class label of the current node.
        """
        pass

    @abstractmethod
    def mark_pruned(self, accuracy: Callable[[], float]) -> bool:
        """
        Recursively tries to prune nodes. Keeps the change, if they do not affect accuracy.
        :param accuracy: Function to determine tree's accuracy
        :return: Value indicating whether node has been pruned
        """
        pass

    @abstractmethod
    def get_pruned(self) -> "BaseNode":
        """
        Simplifies tree by replacing pruned nodes with leafs.
        This is not a pure function!
        :return: Root of simplified tree
        """
        pass

    @abstractmethod
    def to_string(self, indent: int = 0) -> str:
        pass


class Node(BaseNode):
    """
    Inner node of tree, stores its children and what attribute those select
    """
    attribute: object = None
    children: List[BaseNode] = None
    fallback_label: object = None
    pruned: bool = False

    def __init__(self, matcher: BaseMatcher, attribute: object, children: List[BaseNode], fallback_label: object):
        super(Node, self).__init__(matcher)
        self.attribute = attribute
        self.children = children
        self.fallback_label = fallback_label

    def is_leaf(self):
        return self.pruned

    def get_label(self):
        return self.fallback_label

    def mark_pruned(self, accuracy: Callable[[], float]) -> bool:
        if all(map(lambda child: child.mark_pruned(accuracy), self.children)):
            pre_accuracy = accuracy()
            self.pruned = True
            if accuracy() >= pre_accuracy:
                return True

        self.pruned = False
        return False

    def get_pruned(self) -> BaseNode:
        if self.pruned:
            return Leaf(self.matcher, self.fallback_label)

        self.children = list(map(lambda child: child.get_pruned(), self.children))
        return self

    def to_string(self, indent: int = 0) -> str:
        result = "{}{} -> switch on {}, fallback to {}"\
            .format("\t" * indent, self.matcher, self.attribute, self.fallback_label)
        child_results = map(lambda child: child.to_string(indent + 1), self.children)
        return "\n".join([result, *child_results])


class Leaf(BaseNode):
    """
    Leaf of tree, stores the class label
    """
    label: object = None

    def __init__(self, matcher: BaseMatcher, label: object):
        super(Leaf, self).__init__(matcher)
        self.label = label

    def is_leaf(self):
        return True

    def get_label(self):
        return self.label

    def mark_pruned(self, accuracy: Callable[[], float]) -> bool:
        return True

    def get_pruned(self) -> BaseNode:
        return self

    def to_string(self, indent: int = 0) -> str:
        return "{}{} -> {}".format("\t" * indent, self.matcher, self.label)
