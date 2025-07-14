import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()

print(dir(digits))

from matplotlib import pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
    plt.show()

df = pd.DataFrame(digits.data)
print(df)

print(digits.target)

df['target'] = digits.target
print(df)

df['target'] = digits.target
print(df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis = 'columns'), digits.target, test_size = 0.2)
print(len(x_test))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

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

