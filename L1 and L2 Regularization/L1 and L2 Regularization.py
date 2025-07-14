import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\L1 and L2 Regularization\Melbourne_housing_FULL.csv')
print(dataset)

print(dataset.nunique())

# let's use limited columns which makes more sense for serving our purpose
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
print(dataset)
print(dataset.isna().sum())

cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
print(dataset.isna().sum())

dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
print(dataset.isna().sum())
                      
dataset.dropna(inplace=True)
print(dataset.isna().sum())

dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset)

X = dataset.drop('Price', axis=1)
y = dataset['Price']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_X, train_y)
score = reg.score(test_X, test_y)
print(score)

score = reg.score(train_X, train_y)
print(score)

from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)
score_lasso_test = lasso_reg.score(test_X, test_y)
print(score_lasso_test)
score_lasso_train =lasso_reg.score(train_X, train_y)
print(score_lasso_train)

from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)
score_ridge_test = ridge_reg.score(test_X, test_y)
print(score_ridge_test)
score_ridge_train =ridge_reg.score(train_X, train_y)
print(score_ridge_train)
