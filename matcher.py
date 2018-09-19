from abc import abstractmethod


class BaseMatcher:
    """
    Base class for checking if a value matches a node
    """
    @abstractmethod
    def is_match(self, value: object):
        pass


class DiscreteMatcher(BaseMatcher):
    """
    Class for checking if a value matches a discrete node
    """
    value: object

    def __init__(self, value: object):
        self.value = value

    def is_match(self, value: object) -> bool:
        return value == self.value

    def __str__(self):
        return "= {}".format(self.value)


class ContinuousMatcher(BaseMatcher):
    """
    Class for checking if a value matches a continuous node
    """
    value: float
    greater: bool

    def __init__(self, value: float, greater: bool):
        self.value = value
        self.greater = greater

    def is_match(self, value: object) -> bool:
        return value > self.value if self.greater else value <= self.value

    def __str__(self):
        return ("> {}" if self.greater else "<= {}").format(self.value)
