
# スコアの改善
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
df = pd.read_csv('./Survived.csv')

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df['Embarked'].fillna(df['Embarked'].mode()[0])

x = df[['Pclass','Age','SibSp','Parch','Fare']]
t = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,t,
test_size = 0.2,random_state = 0)

for j in range(1,15): # jは木の深さ jには1～14が入る
    # xは特徴量、tは正解データ
    train_score,test_score,model = learn(x,t,depth = j)
    sentence="訓練データの正解率{}"
    sentence2="訓練データの正解率{}"
    total_sentence='深さ{}:'+sentence+sentence2
    print(total_sentence.format(j,
    train_score,test_score))

'''
深さ1:訓練データの正解率0.659訓練データの正解率0.704
深さ2:訓練データの正解率0.699訓練データの正解率0.732
深さ3:訓練データの正解率0.704訓練データの正解率0.737
深さ4:訓練データの正解率0.698訓練データの正解率0.726
深さ5:訓練データの正解率0.722訓練データの正解率0.737
深さ6:訓練データの正解率0.77訓練データの正解率0.698
深さ7:訓練データの正解率0.771訓練データの正解率0.648
深さ8:訓練データの正解率0.781訓練データの正解率0.631
深さ9:訓練データの正解率0.83訓練データの正解率0.704
深さ10:訓練データの正解率0.851訓練データの正解率0.687
深さ11:訓練データの正解率0.878訓練データの正解率0.676
深さ12:訓練データの正解率0.892訓練データの正解率0.654
深さ13:訓練データの正解率0.909訓練データの正解率0.654
深さ14:訓練データの正解率0.92訓練データの正解率0.654
'''
