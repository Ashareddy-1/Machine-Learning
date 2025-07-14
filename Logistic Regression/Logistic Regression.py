import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\Machine Learning\Logistic Regression\insurance_data.csv')
print(df)

plt.scatter(df.age, df.bought_insurance, marker = '+', color='red')
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size = 0.9)
print(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(prediction)

score = model.score(x_test, y_test)
print(score)

probability = model.predict_proba(x_test)
print(probability)
