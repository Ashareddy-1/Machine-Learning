import pandas as pd
from sklearn.datasets import load_digits

dataset = load_digits()
print(dataset.keys())

print(dataset.data.shape)
print(dataset.data[0])
print(dataset.data[0].reshape(8,8))

from matplotlib import pyplot as plt
plt.gray()
plt.matshow(dataset.data[0].reshape(8,8))
plt.show()

print(dataset.target)

import numpy as np
print(np.unique(dataset.target))

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df)

print(df.describe())

x = df
y = dataset.target

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=30)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(x.shape)

from sklearn.decomposition import PCA
pca = PCA(0.95)

x_pca = pca.fit_transform(x)
print(x_pca.shape)

print(x_pca)
print(pca.explained_variance_ratio_)
print(pca.n_components_)

x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)
model = LogisticRegression()
model.fit(x_train_pca, y_train)
score = model.score(x_test_pca, y_test)
print(score)
