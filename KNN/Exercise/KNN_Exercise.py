#From sklearn.datasets load digits dataset and do following

#Classify digits (0 to 9) using KNN classifier. You can use different values
#for k neighbors and need to figure out a value of K that gives you a maximum score.
#You can manually try different values of K or use gridsearchcv
#Plot confusion matrix
#Plot classification report

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



digits = load_digits()
print(dir(digits))

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)


clf = GridSearchCV(KNeighborsClassifier(), {
    'n_neighbors': list(range(1, 21))
}, cv=5, return_train_score=False)

clf.fit(digits.data,digits.target)
print(clf.cv_results_)

from sklearn.metrics import confusion_matrix
y_predicted = clf.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)

print(clf.best_score_)
print(clf.best_params_)


import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted))
