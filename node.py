from abc import abstractmethod
from typing import List

from matcher import BaseMatcher


class BaseNode:
    """
    Base class for nodes, stores function to decide if value matches this node.
    """
    matcher: BaseMatcher = None

    def __init__(self, matcher: BaseMatcher):
        self.matcher = matcher

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

    def to_string(self, indent: int = 0) -> str:
        return "{}{} -> {}".format("\t" * indent, self.matcher, self.label)
