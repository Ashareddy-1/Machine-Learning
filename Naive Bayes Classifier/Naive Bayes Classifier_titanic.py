import pandas as pd
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Naive Bayes Classifier\titanic.csv')
print(df)

df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace = True)
print(df)

target = df.Survived
inputs = df.drop('Survived', axis = 'columns')

dummies = pd.get_dummies(inputs.Sex)
print(dummies)

inputs = pd.concat([inputs, dummies], axis = 'columns')
print(inputs)

inputs.drop('Sex', axis = 'columns', inplace = True)
print(inputs)

inputs.columns[inputs.isna().any()]
print(inputs)

inputs.Age[:10]
print(inputs)

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)

print(x_test[:10])
print(y_test[:10])

y_predict = model.predict(x_test[:10])
print(y_predict)

probability = model.predict_proba(x_test[:10])
print(probability)
