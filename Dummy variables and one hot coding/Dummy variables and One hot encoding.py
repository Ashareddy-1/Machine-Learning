import pandas as pd
df = pd.read_excel(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Dummy variables and one hot coding\homeprices.xlsx')
print(df)
#Using dummies
dummies = pd.get_dummies(df.town)
print(dummies)

merged = pd.concat([df, dummies], axis = 'columns')
print(merged)

final = merged.drop(['town', 'west windsor'], axis = 'columns')
print(final)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

x = final.drop('price', axis = 'columns')
print(x)
y = final.price
print(y)

training = model.fit(x,y)
print(training)

new_data = pd.DataFrame({'area': [2800], 'monroe township': [0], 'robinsville': [1]})
prediction = model.predict(new_data)
print(prediction)

score = model.score(x,y)
print(score)

#Using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df
label_encoding = le.fit_transform(dfle.town)
print(label_encoding)

dfle.town = le.fit_transform(dfle.town)
print(dfle)

x = dfle[['town', 'area']].values
print(x)

y = dfle.price
print(y)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Apply OneHotEncoder to the first column
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# Transform the data
x = ct.fit_transform(x)

print(x)

x = x[:, 1:]
print(x)

training = model.fit(x,y)
print(training)

prediction = model.predict([[1,0,2800]])
print(prediction)



