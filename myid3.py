# Module file for implementation of ID3 algorithm.

from typing import IO, List, Union
from pandas import DataFrame, Series


class DecisionTree:
    def __init__(self, load_from: Union[str, IO, None] = None):
        """
        Initializes empty decision tree.
        :param load_from: Restore previously saved instance from file.
        """
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")

    def train(self, X: DataFrame, y: Series, attrs: List[str], prune: bool=False) -> None:
        """
        Uses ID3 to train a decision tree on the supplied data.
        :param X: Training data (to classify)
        :param y: Classes of training data
        :param attrs: Names of attributes in X
        :param prune: Indicates whether the tree should be pruned
        """
        pass

    def predict(self, instance: DataFrame) -> List[object]:
        """
        Classifies the given instance based on the learned decision tree,
        :param instance: Data instance to classify
        :return: Predicted class of instance
        :raises ValueError if the class is not trained
        """
        pass

    def test(self, X: DataFrame, y: Series, display: bool=False) -> dict:
        """
        Runs statistical tests on data,
        :param X: Test data
        :param y: Classes of test data
        :param display: Indicates whether the results should be printed
        :return: Dictionary of statistical test results
        """
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
        return "ID3 untrained"

    def save(self, output: Union[str, IO]) -> None:
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pass
