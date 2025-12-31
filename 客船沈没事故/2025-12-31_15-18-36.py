
# ピポットテーブルによる集計
# 
# グループに分けして穴埋めしたあとの出力はデフォルトでは平均値。
# オプションで最大値にも変えられる。


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

pd.pivot_table(df2,index = 'Survived',columns = 'Pclass', values = 'Age',aggfunc='max')

'''
Out[2]: 
Pclass       1     2     3
Survived                  
0         71.0  70.0  74.0
1         80.0  62.0  63.0
'''