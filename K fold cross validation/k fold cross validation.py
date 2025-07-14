from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3)
print(len(x_train))

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_score = lr.score(x_test, y_test)
print(lr_score)

svm = SVC()
svm.fit(x_train, y_train)
svm_score = svm.score(x_test, y_test)
print(svm_score)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_score = rf.score(x_test, y_test)
print(rf_score)

#Now let's try k-fold
#Example to understand k-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits = 3)
print(kf)

for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

#Now let's try k-fold on digits.data

def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

scores_1 = []
scores_svm = []
scores_rf = []

for train_index, test_index in kf.split(digits.data):
    x_train, x_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scores_1.append(get_score(LogisticRegression(), x_train, x_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), x_train, x_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(), x_train, x_test, y_train, y_test))
    print(scores_1)
    print(scores_svm)
    print(scores_rf)

#The complete code what you have written till now is messy so using sklearn cross val score you can write it in a simple way

from sklearn.model_selection import cross_val_score
lr = cross_val_score(LogisticRegression(), digits.data, digits.target)
svm = cross_val_score(SVC(), digits.data, digits.target)
rf = cross_val_score(RandomForestClassifier(), digits.data, digits.target)

print(lr)
print(svm)
print(rf)
