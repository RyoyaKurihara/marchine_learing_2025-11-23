
# 木の深さを変える。

def learn(x,t,depth=3):
    x_train,x_test,y_train,y_test = train_test_split(x,
        t,test_size = 0.2,random_state = 0)
    model = tree.DecisionTreeClassifier(max_depth =depth,random_state = 0,class_weight="balanced")
    model.fit(x_train,y_train)

    score=model.score(X=x_train,y=y_train)
    score2=model.score(X=x_test,y=y_test)
    return round(score,3),round(score2,3),model


import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df2 = pd.read_csv('./Survived.csv')

df2.groupby('Survived')['Age'].mean()
df2.groupby('Pclass')['Age'].mean()

pd.pivot_table(df2,index = 'Survived',columns = 'Pclass', values = 'Age')


'''
Out[3]: 
Pclass            1          2          3
Survived                                 
0         43.695312  33.544444  26.555556
1         35.368197  25.901566  20.646118
'''



