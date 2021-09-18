import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df_test = pd.read_csv('hw1/p2_titanic/test.csv')
print(df_test['Cabin'])
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

x = df_test.drop(['PassengerId','Name','SibSp','Ticket','Cabin','Embarked'],axis = 1)
x = x.fillna(x.mean())

print(x.head())

logistic_regression = pickle.load(open('hw1/p2_titanic/logistic_regression_model.sav', 'rb'))
y_pred = logistic_regression.predict(x)

df = pd.DataFrame(y_pred, columns=["predictions"])
df.to_csv('predictions_titanic.csv', index=False)
