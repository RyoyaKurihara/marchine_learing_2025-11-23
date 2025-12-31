
# ダミー変数の追加
#
# 機械学習で使える値は数値だけのため、文字列があるときは数値に置き換えることで機械学習に使える様になる。
# 
# `pd.get_dummies(df2['Embarked'],drop_first = False, dtype=int)`で
# `drop_first = False`とすると、変数の数だけダミー変数の列を作ってくれる。

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

male = pd.get_dummies(df2['Sex'],drop_first = True, dtype=int)
embarked = pd.get_dummies(df2['Embarked'],drop_first = False, dtype=int)

x = df2[['Pclass','Age','SibSp','Parch','Fare']]
t = df2['Survived']

x_new = pd.concat([x,male],axis=1)

#x_new = x_tmp.drop("Sex",axis=1)
for j in range(1,6): # jは木の深さ
 # xは特徴量、tは目的変数
    s1,s2,m = learn(x_new,t,depth = j)
    s='深さ{}:訓練データの精度{}::テストデータの精度{}'
    print(s.format(j,s1,s2))

'''
深さ1:訓練データの精度0.787::テストデータの精度0.788
深さ2:訓練データの精度0.792::テストデータの精度0.782
深さ3:訓練データの精度0.847::テストデータの精度0.81
深さ4:訓練データの精度0.854::テストデータの精度0.849
深さ5:訓練データの精度0.865::テストデータの精度0.86
'''






