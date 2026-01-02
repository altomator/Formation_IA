import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

print(iris.feature_names)
print(class_names)

# Scatter plot of the first two features
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(class_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], label=class_name)

plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title("Scatter Plot of Iris Dataset")
plt.legend()
plt.show()