# LT2316 H18 Assignment 1

Git project for implementing assignment 1 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.

The included dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Balance+Scale).

## Your notes

This is an Implementation of the [ID3 algorithm](https://en.wikipedia.org/wiki/ID3_algorithm).
It builds a decision tree based on training data, which can later be used to classify similar data.

### Extensions

Let's look at the extensions this implementation contains:

- Handling of previously unencountered values
- [Reduced Error Pruning](https://en.wikipedia.org/wiki/Pruning_(decision_trees)#Reduced_error_pruning)
- Support for continuously valued attributes

#### Unencountered Values

In its textbook version ID3 only stores label information in leafs. This means it will not have enough information to guess a label when none of a node's leafs match a value.
We can fix this by storing the most popular label of all values that are active when the node is being built. This value is used as a fallback when no child can be found.

#### Reduced Error Pruning

Pruning can help against overfitting and poor generalization. Reduced Error Pruning traverses the tree from bottom to top and tries to replace nodes with leafs. This uses the fallback value from the previous extension as the label for nodes. Nodes are only pruned when all their children are leafs and their absence does not cause a drop in accuracy.

Nodes are initially not replaced, but only marked as pruned. A separate traversal of the tree replaces pruned nodes with leafs. This avoids a lot of tree rewriting when it is being tested.

#### Continuous Attributes

ID3 creates children for all values it encounters in the training pass. This means that it will mostly encounter unknown values when continuous values are used. Instead we are going to split values along a threshold.

To determine this threshold we make a list of all values that the training set encounters for the specific attribute at the current node. We will sort the list of values and then consider all means of pairs of adjacent values as potential thresholds.

Next we find the threshold with the highest information gain. This will then be the threshold we use to split the data at this node. To find the information gain for a given attribute and threshold we use this slightly modified version of the discrete case:

![IG(A,S) = H(S) - SUM over t in T p(t)H(t)](images/IGdef.svg)  
Only now ![T](images/T.svg) is split by a threshold, such that ![T = {{x in S | x.A <= threshold}, {x in S | x.A > threshold}}](images/Tdef.svg).  
It still holds that ![S = Union over t in T](images/Sass.svg).

---
Remark: Continuous attributes are detected by checking the data type. Float64 is considered continuous, while Int64 is considered discrete.