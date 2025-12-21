
# 学習とスコアの出力

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('./Survived.csv')

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df['Embarked'].fillna(df['Embarked'].mode()[0])

x = df[['Pclass','Age','SibSp','Parch','Fare']]
t = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,t,
test_size = 0.2,random_state = 0)

model = tree.DecisionTreeClassifier(max_depth = 5,
 random_state = 0,class_weight ='balanced')

model.fit(x_train,y_train)
model.score(X = x_test,y = y_test)

'''
Out[4]: 0.7374301675977654
'''

