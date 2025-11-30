
import pandas as pd
from sklearn import tree

df = pd.read_csv('./iris.csv')
df2 = df.dropna(how = 'any', axis = 0)

x = df2[['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']]
t = df2['種類']

model = tree.DecisionTreeClassifier(max_depth = 2, random_state=0)
model.fit(x, t)
model.score(x, t)

