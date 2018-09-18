from abc import abstractmethod
from typing import List


class BaseNode:
    """
    Base class for nodes, stores value that leads to this node.
    """
    value: object = None

    def __init__(self, value: object):
        self.value = value

    @abstractmethod
    def to_string(self, indent: int = 0) -> str:
        pass


class Node(BaseNode):
    """
    Inner node of tree, stores its children and what attribute those select
    """
    attribute: object = None
    children: List[BaseNode] = None

    def __init__(self, value: object, attribute: object, children: List[BaseNode]):
        super(Node, self).__init__(value)
        self.attribute = attribute
        self.children = children

    def to_string(self, indent: int = 0) -> str:
        result = "{}{} -> switch on {}".format("\t" * indent, self.value, self.attribute)
        child_results = map(lambda child: child.to_string(indent + 1), self.children)
        return "\n".join([result, *child_results])


class Leaf(BaseNode):
    """
    Leaf of tree, stores the class label
    """
    label: object = None

    def __init__(self, value: object, label: object):
        super(Leaf, self).__init__(value)
        self.label = label

    def to_string(self, indent: int = 0) -> str:
        return "{}{} -> {}".format("\t" * indent, self.value, self.label)