import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Decision Tree\Exercise\titanic.csv')
print(df)

#No need 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked' so dropping and Survived is the output so dropping
inputs = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Survived'], axis = 'columns')
print(inputs)
target = df['Survived']
print(target)

dummies = pd.get_dummies(inputs)
print(dummies)
                 
merged = pd.concat([inputs, dummies], axis = 'columns')
print(merged)

final = merged.drop('Sex', axis = 'columns')
print(final)

x_train, x_test, y_train, y_test = train_test_split(final, target, test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)
