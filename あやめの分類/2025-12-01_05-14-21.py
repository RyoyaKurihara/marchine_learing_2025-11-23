
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('./iris.csv')
df2 = df.dropna(how = 'any', axis = 0)

x = df2[['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']]
t = df2['種類']

x_train, x_test, y_train, y_test = train_test_split(x, t, test_size = 0.3, random_state = 0)

# print(x_train.shape)
# print(x_test.shape) 

model = tree.DecisionTreeClassifier(max_depth = 2, random_state=0)
model.fit(x_train, y_train)
model.score(x_test, y_test)
