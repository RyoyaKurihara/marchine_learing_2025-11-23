
# 欠損値を埋める
# 
# ピポットテーブルにより、取得した値で各欠損値を埋める。
# fillnaを使うと一括で埋めてしまうため、それぞれ手動で埋める必要がある。

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


is_null = df2['Age'].isnull()

# Pclass 1　に関する埋め込み
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 0)
    &(is_null),'Age'] = 43
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 1)
    &(is_null),'Age'] = 35

# Pclass 2　に関する埋め込み
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 0)
    &(is_null),'Age'] = 33
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 1)
    &(is_null),'Age'] = 25

# Pclass 3　に関する埋め込み
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 0)
    &(is_null),'Age'] = 26
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 1)
    &(is_null),'Age'] = 20


df2

'''
Out[10]: 
     PassengerId  Survived  Pclass  ...     Fare  Cabin  Embarked
0              1         0       3  ...   7.2500    NaN         S
1              2         1       1  ...  71.2833    C85         C
2              3         1       3  ...   7.9250    NaN         S
3              4         1       1  ...  53.1000   C123         S
4              5         0       3  ...   8.0500    NaN         S
..           ...       ...     ...  ...      ...    ...       ...
886          887         0       2  ...  13.0000    NaN         S
887          888         1       1  ...  30.0000    B42         S
888          889         0       3  ...  23.4500    NaN         S
889          890         1       1  ...  30.0000   C148         C
890          891         0       3  ...   7.7500    NaN         Q

[891 rows x 11 columns]
'''
