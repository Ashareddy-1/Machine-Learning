#Use iris flower dataset from sklearn library and use cross_val_score against following models to measure the performance of each. In the end figure out the model with best performance,
#Logistic Regression
#SVM
#Decision Tree
#Random Forest

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

iris = load_iris()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3)
print(len(x_train))

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_score = lr.score(x_test, y_test)
print(lr_score)

svm = SVC()
svm.fit(x_train, y_train)
svm_score = svm.score(x_test, y_test)
print(svm_score)

T = DecisionTreeClassifier()
T.fit(x_train, y_train)
T_score = T.score(x_test, y_test)
print(T_score)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_score = rf.score(x_test, y_test)
print(rf_score)

lr = cross_val_score(LogisticRegression(), iris.data, iris.target)
svm = cross_val_score(SVC(), iris.data, iris.target)
T = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
rf = cross_val_score(RandomForestClassifier(), iris.data, iris.target)

print(np.average(lr))
print(np.average(svm))
print(np.average(T))
print(np.average(rf))
