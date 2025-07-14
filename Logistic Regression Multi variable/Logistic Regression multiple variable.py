import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))
print(digits.data[0])

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    plt.show()

print(digits.target[0:5])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(len(x_train))
print(len(x_test))

from sklearn.linear_model import LogisticRegression

# Create and train the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

plt.matshow(digits.images[67])
plt.show()
print(digits.target[67])
prediction = model.predict([digits.data[67]])
print(prediction)

y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
