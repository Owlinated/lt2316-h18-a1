# Testing file for decision tree induction.
# This file is intended to help you figure out what is expected
# for you to pass the assignment. I don't recommend you change it, but
# rather copy it and make your own tests or write tests afresh.
# We can/will test your code using other data.

from pandas import read_csv
from myid3 import DecisionTree # What you will write in the myid3.py file.

# Loading the data.
balancedata = read_csv("balance-scale.data", header=None)
balancedata = balancedata.sample(frac=1).reset_index(drop=True) #shuffla shuffla

# Making the data usable.
cols = ["class", "left-weight", "left-distance", "right-weight", "right-distance"]
balancedata.columns = cols
y = balancedata["class"]
X = balancedata[balancedata.columns[1:]]

trainlen = int(len(X)*0.8)

train_X = X[:trainlen]
test_X = X[trainlen:]
train_y = y[:trainlen]
test_y = y[trainlen:]

print("A little taste of the training data.")
print(train_X[:10])
print(train_y[:10])

# Train the model using the basic features of DecisionTree
dt = DecisionTree()
dt.train(X, y, cols[1:])
print("The model looks like:")
print(dt)
print("Testing it out.")
dt.test(test_X, test_y, display=True)

# Demonstrate saving and loading the model.
with open("whatever.model", "w") as modelfile:
    dt.save(modelfile)
with open("whatever.model", "r") as modelfile:
    dt2 = DecisionTree(load_from=modelfile)
    print(dt2)

print("All done!")
