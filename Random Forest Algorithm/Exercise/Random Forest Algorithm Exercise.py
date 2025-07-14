#Use iris flower dataset to predict flower species using random forest classifier
#1.Measure prediction score using default n_estimators(10)
#2.Now fine tune your model by changing number of tress in your classifier and find the best score you can get using how many tress

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

print(dir(iris))
print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)

df['target'] = iris.target
print(df)

print(iris.target_names)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis = 'columns'), df.target, test_size = 0.2)
print(len(x_test))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

#let's tune this model by giving n_estimators = 10 i.e., 10 random trace and see the score
model = RandomForestClassifier(n_estimators = 10)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

#let's tune this model by giving n_estimators = 20 i.e., 20 random trace and see the score
model = RandomForestClassifier(n_estimators = 20)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

#let's tune this model by giving n_estimators = 30 i.e., 30 random trace and see the score
model = RandomForestClassifier(n_estimators = 30)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

#let's tune this model by giving n_estimators = 5 i.e., 5 random trace and see the score
model = RandomForestClassifier(n_estimators = 5)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

#let's tune this model by giving n_estimators = 5 i.e., 5 random trace and see the score
model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()
