import pandas as pd
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\Machine Learning\Splitting dataset into Training and Test\carprices.csv')
print(df)

import matplotlib.pyplot as plt
plt.scatter(df['Mileage'], df['Sell Price($)'])
plt.show()

plt.scatter(df['Age(yrs)'], df['Sell Price($)'])
plt.show()

x = df[['Mileage', 'Age(yrs)']]
print(x)
y = df['Sell Price($)']
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print(prediction)
print(y_test)
score = clf.score(x_test, y_test)
print(score)
