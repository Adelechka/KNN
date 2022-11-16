from sklearn import datasets
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import pylab as pl
from collections import Counter

def most_common(lst):
  count = Counter(lst).most_common(1)[0]
  return count[0]
def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric =euclidean):
        self.k = k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy

def _normalize_data(dataset):
    num_features = len(dataset)
    for i in range(4):
        column_values = [row[i] for row in dataset]
        column_min = np.min(column_values)
        column_max = np.max(column_values)

        for row in dataset:
            row[i] = (row[i] - column_min) / (column_max - column_min)

iris = datasets.load_iris()
targets = iris.target
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = targets

x = iris.get('data')
sns.pairplot(df, hue="Species", size=3)
plt.show()

_normalize_data(x)

df.data = x
sns.pairplot(df, hue="Species", size=3)
plt.show()

y = iris.get('target')
n = len(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

k_need = 0

accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)

fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()

max = 0
for i in range(len(accuracies)):
  if (accuracies[i] > max):
    max = accuracies[i]
    k_need = ks[i]

new_raw = [[0.2586, 0.31239, 0.643722, 0.10043932]]

knn = KNeighborsClassifier(k=k_need)
knn.fit(x, y)
result = knn.predict(new_raw)
print(result)

# new_class = predict_classification(x, new_raw, k)
# print(new_class)
