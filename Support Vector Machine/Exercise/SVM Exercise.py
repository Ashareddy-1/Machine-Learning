#Train SVM classifier using sklearn digits dataset(i.e. from sklearn.datasets import load_digits) and the,
#1. Measure accuracy of your model using different kernels such as rbf and linear
#2. Tune your model further using regularization and gamma parameters and try to come up with highest accuracy score
#3. Use 80% of samples as training data size

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

digits = load_digits()
print(digits.data)
print(digits.target)


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
print(len(x_train))
print(len(x_test))

rbf_model = SVC(kernel='rbf', C = 4, gamma = 'scale')
print(rbf_model)

print(rbf_model.fit(x_train, y_train))
score = rbf_model.score(x_test, y_test)
print(score)

linear_model = SVC(kernel='linear', C = 4, gamma = 'scale')
print(linear_model)

print(linear_model.fit(x_train, y_train))
score = linear_model.score(x_test, y_test)
print(score)
