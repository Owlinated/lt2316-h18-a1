# Testing file for additional tests
from pathlib import Path
from pandas import read_csv, DataFrame, Series
from myid3 import DecisionTree


def test_set(train_X: DataFrame, train_y: Series, test_X: DataFrame, test_y: Series, print_model: bool):
    tree = DecisionTree()
    tree.train(train_X, train_y, list(train_X))

    if print_model:
        print("\n### Model:")
        print(tree)

    tree.prune(train_X, train_y)

    if print_model:
        print("\n### Pruned Model:")
        print(tree)

    print("\n### Test:")
    tree.test(test_X, test_y, display=True)


def test_file(file: str, print_model: bool):
    print("\n########################"
          "\n# Set: {}".format(file))

    data = read_csv(Path("data").joinpath(file)).sample(frac=1)
    y = data[data.columns[0]]
    X = data[data.columns[1:]]

    train_length = int(len(X) * 0.8)
    train_X = X[:train_length]
    test_X = X[train_length:]
    train_y = y[:train_length]
    test_y = y[train_length:]

    # Train on all values
    print("\n## Complete training")
    test_set(X, y, test_X, test_y, print_model)

    # Train only on training set
    print("\n## Subset training")
    test_set(train_X, train_y, test_X, test_y, print_model)


def test_all():
    # This model is too large to be easily readable, don't print it
    for file in ["balance-scale.data"]:
        test_file(file, False)

    # Print these smaller models
    # easy.data, contains an attribute att_b which decides between classes
    # impossible.data, has no correlation between attribute values and classes
    # prunable.data, is like easy but has an outlier to allow pruning
    # easy_continuous.data contains a continuous attribute att_b which decides between classes
    # easy_string.data contains a categorical string attribute att_b which decides between classes
    for file in ["easy.data", "impossible.data", "prunable.data", "easy_continuous.data", "easy_string.data"]:
        test_file(file, True)


if __name__ == "__main__":
    test_all()