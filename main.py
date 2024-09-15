import flet as ft

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def damn():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main(page: ft.Page):
    acc = damn()
    page.add(ft.SafeArea(ft.Text("Accuracy_Score: " + str(acc))))


ft.app(main)
