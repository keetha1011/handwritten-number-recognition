import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits


x, y = load_digits()["data"] , load_digits()["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

def prediction(data):
    return tree.predict(np.array(data).reshape(1,-1))[0]
