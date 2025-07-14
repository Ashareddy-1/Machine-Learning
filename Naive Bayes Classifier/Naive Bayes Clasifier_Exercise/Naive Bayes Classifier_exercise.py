#Use wine dataset from sklearn.datasets to classify wines into 3 categories.
#Load the dataset and split it into test and train.
#After that train the model using Gaussian and Multinominal classifier and post which model performs better.
#Use the trained model to perform some predictions on test data.

from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
print(dir(wine))
print(wine.target_names)
df = pd.DataFrame(wine.data, columns = wine.feature_names)
print(df)

target = wine.target
print(target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)


