import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df_train = pd.read_csv('hw1/p2_titanic/train.csv')
print(df_train['Cabin'])
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
# df_train = df_train[df_train['Survived','Pclass','Sex','Age','Fare'].notna()]
# df_train = df_train[df_train['Pclass'].notna()]
# df_train = df_train[df_train['Sex'].notna()]
# df_train = df_train[df_train['Age'].notna()]
# df_train = df_train[df_train['Fare'].notna()]
# df_train = df_train[df_train['Cabin'].notna()]

x = df_train.drop(['Survived','PassengerId','Name','SibSp','Ticket','Cabin','Embarked'],axis = 1)
x = x.fillna(x.mean())
y = df_train.Survived

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
logistic_regression = LogisticRegression().fit(x_train, y_train)

# filename = 'logistic_regression_model.sav'
# pickle.dump(logistic_regression, open(filename, 'wb'))

y_pred = logistic_regression.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy

print(accuracy_percentage)